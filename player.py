# player.py
"""
Lyria Realtime player — full parameter support + transition modes.
Polls Flask /player/config for prompt + parameter changes.

Transition modes (set via config['transition_mode']):
  instant  — hard cut (reset_context) on every prompt change
  smooth   — set the new prompt at full weight; context is kept, so Lyria drifts
             toward it on its own. Used for nudges and steady state, and is the
             fallback when no mode is given.
  blend    — two weighted prompts; blend_weight (0-1) controls the mix live.
             Used for the timed crossfade when the reader crosses into a new
             segment (the reader ramps blend_weight 0->1 over ~10s).
"""

import os
import sys
import asyncio
import numpy as np
import sounddevice as sd
import aiohttp
from google import genai
from google.genai import types

MODEL_ID = "models/lyria-realtime-exp"
SAMPLE_RATE = 48000
POLL_URL = "http://localhost:5000/player/config"
HEARTBEAT_URL = "http://localhost:5000/player/heartbeat"
POLL_INTERVAL = 0.1
HEARTBEAT_INTERVAL = 1.0  # seconds between heartbeat posts

current_volume = 0.2  # shared between play_audio and poll_config


# ============================================================
# Audio playback
# ============================================================

async def play_audio(session):
    """Receive audio chunks from Lyria and play via sounddevice."""
    print("Audio started.")
    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=2, dtype="float32")
    stream.start()
    chunks_received = 0
    messages_received = 0
    exit_reason = "iterator ended (server closed stream)"
    last_heartbeat = 0.0
    loop = asyncio.get_event_loop()
    heartbeat_http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1.0))
    try:
        async for message in session.receive():
            messages_received += 1
            if not message.server_content or not message.server_content.audio_chunks:
                continue
            audio_bytes = message.server_content.audio_chunks[0].data
            if not audio_bytes:
                continue
            chunks_received += 1
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if samples.ndim == 1:
                if len(samples) % 2 != 0:
                    samples = np.pad(samples, (0, 1))
                samples = np.reshape(samples, (-1, 2))
            stream.write(samples * current_volume)

            now = loop.time()
            if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                last_heartbeat = now
                try:
                    async with heartbeat_http.post(HEARTBEAT_URL) as r:
                        await r.read()
                except Exception:
                    pass
    except asyncio.CancelledError:
        exit_reason = "cancelled"
    except Exception as e:
        exit_reason = f"{type(e).__name__}: {e}"
    finally:
        stream.stop()
        stream.close()
        await heartbeat_http.close()
        print(f"Audio ended. reason={exit_reason} messages={messages_received} audio_chunks={chunks_received}")


# ============================================================
# Enum helpers
# ============================================================

def _scale(name):
    """Convert scale name string → types.Scale, or None."""
    if not name:
        return None
    try:
        return getattr(types.Scale, name)
    except AttributeError:
        print(f"Unknown scale: {name}")
        return None


def _gen_mode(name):
    """Convert mode name string → types.MusicGenerationMode, or None."""
    if not name:
        return None
    try:
        return getattr(types.MusicGenerationMode, name)
    except AttributeError:
        print(f"Unknown generation mode: {name}")
        return None


# ============================================================
# Config polling
# ============================================================

async def poll_config(session):
    """Poll /player/config and apply changes to the Lyria session."""
    last_prompt_a = None
    last_prompt_b = None
    last_negative = None
    last_blend_weight = -1.0
    last_transition = None
    last_params = {}
    last_paused = False
    last_weighted_prompts = None

    async with aiohttp.ClientSession() as http:
        while True:
            try:
                async with http.get(POLL_URL) as resp:
                    config = await resp.json()

                if not config:
                    await asyncio.sleep(POLL_INTERVAL)
                    continue

                # --- Pause / Resume ---
                paused = bool(config.get("_paused", False))
                if paused != last_paused:
                    if paused:
                        await session.pause()
                        print("Paused.")
                    else:
                        await session.play()
                        print("Resumed.")
                    last_paused = paused

                if paused:
                    await asyncio.sleep(POLL_INTERVAL)
                    continue

                # --- Prompt + transition ---
                weighted_prompts = config.get("weighted_prompts")  # [{text, weight}, ...]

                if weighted_prompts:
                    # Component-based weighted mode
                    if weighted_prompts != last_weighted_prompts:
                        negative = config.get("negative_prompt", "")
                        neg_suffix = f", without {negative}" if negative else ""
                        # Append negative to first component only
                        wps = []
                        for i, wp in enumerate(weighted_prompts):
                            text = wp["text"] + (neg_suffix if i == 0 else "")
                            wps.append(types.WeightedPrompt(text=text, weight=float(wp["weight"])))
                        await session.set_weighted_prompts(prompts=wps)
                        last_weighted_prompts = weighted_prompts
                        labels = " + ".join(f"{wp['weight']:.2f}×{wp['text'][:25]}" for wp in weighted_prompts)
                        print(f"Weighted components: {labels}")
                    # Reset single-prompt tracking so switching back works
                    last_prompt_a = ""
                else:
                    last_weighted_prompts = None
                    prompt_a    = config.get("prompt", "")
                    prompt_b    = config.get("prompt_b", "")
                    negative    = config.get("negative_prompt", "")
                    blend_w     = float(config.get("blend_weight", 0.0))
                    transition  = config.get("transition_mode", "smooth")
                    neg_suffix  = f", without {negative}" if negative else ""

                    prompt_changed = (
                        prompt_a    != last_prompt_a or
                        prompt_b    != last_prompt_b or
                        negative    != last_negative or
                        blend_w     != last_blend_weight or
                        transition  != last_transition
                    )

                    if prompt_changed and prompt_a:
                        if transition == "blend" and prompt_b:
                            w_a = max(0.0, 1.0 - blend_w)
                            w_b = max(0.0, blend_w)
                            await session.set_weighted_prompts(prompts=[
                                types.WeightedPrompt(text=prompt_a + neg_suffix, weight=w_a),
                                types.WeightedPrompt(text=prompt_b,              weight=w_b),
                            ])
                            print(f"Blend {w_a:.2f}×A + {w_b:.2f}×B")
                        else:
                            await session.set_weighted_prompts(prompts=[
                                types.WeightedPrompt(text=prompt_a + neg_suffix, weight=1.0),
                            ])
                            if transition == "instant" and prompt_a != last_prompt_a:
                                await session.reset_context()
                                print(f"Instant cut → {prompt_a}")
                                if negative:
                                    print(f"  negative → {negative}")
                            else:
                                print(f"Smooth update → {prompt_a}")
                                if negative:
                                    print(f"  negative → {negative}")

                        last_prompt_a   = prompt_a
                        last_prompt_b   = prompt_b
                        last_negative   = negative
                        last_blend_weight = blend_w
                        last_transition = transition

                # --- Music generation config (continuous params) ---
                new_params = {}

                for key in ["density", "brightness", "guidance", "temperature"]:
                    if key in config:
                        new_params[key] = float(config[key])

                for key in ["mute_drums", "mute_bass", "only_bass_and_drums"]:
                    if key in config:
                        new_params[key] = bool(config[key])

                if config.get("top_k"):
                    new_params["top_k"] = int(config["top_k"])

                seed = config.get("seed")
                if seed is not None and seed != "" and seed is not None:
                    try:
                        new_params["seed"] = int(seed)
                    except (ValueError, TypeError):
                        pass

                scale = _scale(config.get("scale", ""))
                if scale is not None:
                    new_params["scale"] = scale

                mode = _gen_mode(config.get("music_generation_mode", ""))
                if mode is not None:
                    new_params["music_generation_mode"] = mode

                changed = {k: v for k, v in new_params.items() if last_params.get(k) != v}
                if changed:
                    await session.set_music_generation_config(
                        config=types.LiveMusicGenerationConfig(**changed)
                    )
                    last_params.update(changed)
                    print(f"Params: {changed}")

                # --- Volume (client-side multiplier) ---
                if "volume" in config:
                    global current_volume
                    current_volume = max(0.0, min(1.0, float(config["volume"])))

                # --- BPM ---
                if "bpm" in config:
                    new_bpm = int(config["bpm"])
                    if last_params.get("_bpm") != new_bpm:
                        await session.set_music_generation_config(
                            config=types.LiveMusicGenerationConfig(bpm=new_bpm)
                        )
                        last_params["_bpm"] = new_bpm
                        print(f"BPM: {new_bpm}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                msg = str(e)
                # Session died — bail out so run_session exits and the outer
                # retry loop establishes a fresh connection. Otherwise we
                # loop forever firing prompts at a dead websocket.
                if "1011" in msg or "ConnectionClosed" in type(e).__name__ or "closed" in msg.lower():
                    print(f"Poll error (session dead): {e} — exiting poll loop")
                    raise
                print(f"Poll error: {e}")

            await asyncio.sleep(POLL_INTERVAL)


# ============================================================
# Main
# ============================================================

async def run_session(client):
    """Start one Lyria session. Returns when the session ends."""
    print("Connecting to Lyria Realtime…")
    async with client.aio.live.music.connect(model=MODEL_ID) as session:
        print("Connected.")

        # Read current config so we restore prompt/params after a reconnect
        init_prompt = "gentle piano arpeggios, soft strings, warm ambient"
        init_config = {}
        try:
            async with aiohttp.ClientSession() as http:
                async with http.get(POLL_URL) as resp:
                    init_config = await resp.json()
        except Exception:
            pass

        prompt_text = init_config.get("prompt") or init_prompt
        await session.set_weighted_prompts(prompts=[
            types.WeightedPrompt(text=prompt_text, weight=1.0),
        ])

        cfg_kwargs = dict(
            density=float(init_config.get("density", 0.2)),
            brightness=float(init_config.get("brightness", 0.25)),
            guidance=float(init_config.get("guidance", 4.7)),
            temperature=float(init_config.get("temperature", 2.4)),
            mute_drums=bool(init_config.get("mute_drums", True)),
            mute_bass=bool(init_config.get("mute_bass", False)),
        )
        if init_config.get("bpm"):
            cfg_kwargs["bpm"] = int(init_config["bpm"])

        await session.set_music_generation_config(
            config=types.LiveMusicGenerationConfig(**cfg_kwargs)
        )
        await session.play()
        print("Playing — polling for config changes…")

        tasks = [
            asyncio.create_task(play_audio(session), name="play_audio"),
            asyncio.create_task(poll_config(session), name="poll_config"),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        for task in done:
            exc = task.exception()
            if exc:
                raise exc

        ended = ", ".join(task.get_name() for task in done)
        raise RuntimeError(f"Lyria session task ended: {ended}")

    print("Session closed.")


async def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not set")
        sys.exit(1)

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})

    retry_delay = 3
    while True:
        try:
            await run_session(client)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Session error: {e} — reconnecting in {retry_delay}s…")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 30)
            continue
        # Clean exit (CancelledError not raised) — don't reconnect
        break


if __name__ == "__main__":
    asyncio.run(main())

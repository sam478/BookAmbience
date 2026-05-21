# app.py
"""Lyria Study — Flask server for PDF segmentation, prompt generation, and Lyria player control."""

import os
import re
import json
import subprocess
import sys
import pathlib
import time
import uuid
from datetime import datetime, timezone
from flask import Flask, request, jsonify, send_from_directory
from pydantic import BaseModel, Field

app = Flask(__name__, static_folder="static")

BASE_DIR = pathlib.Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
SEGMENTS_DIR = BASE_DIR / "segments"
PROMPTS_DIR = BASE_DIR / "prompts"

for d in [UPLOADS_DIR, SEGMENTS_DIR, PROMPTS_DIR]:
    d.mkdir(exist_ok=True)

# --- Global state ---
player_process = None
player_config = {}
player_last_chunk_ts = 0.0  # epoch seconds; updated by player.py heartbeat
player_last_restart_ts = 0.0


# --- AI Prompt Template ---
STYLE_HINTS = {
    "classical":      "Classical / Orchestral — strings, woodwinds, brass, piano, harp, timpani",
    "celtic":         "Celtic / Folk — tin whistle, uilleann pipes, fiddle, bodhran, bouzouki, Celtic harp",
    "jazz":           "Jazz / Noir — piano, upright bass, brushed drums, muted trumpet, saxophone",
    "ambient":        "Ambient / Minimalist — sparse piano, sustained strings, long tones, silence, slow pads",
    "horror":         "Horror / Gothic — organ, harpsichord, music box, dissonant strings, bass clarinet",
    "japanese":       "Japanese / East Asian — koto, shakuhachi, taiko, shamisen, biwa",
    "middle_eastern": "Middle Eastern — oud, qanun, frame drum, ney flute, darbuka",
    "scifi":          "Sci-fi / Electronic — theremin, analog synth, glass harmonica, sustained strings, electronic pulses",
}

AI_PROMPT_TEMPLATE = """You are creating a music prompt for Lyria, a real-time AI music generator used as adaptive ambience for literary reading.

Analyze this text and write a vivid, cue-like music prompt that captures the scene's mood, period, setting, motion, stakes, and emotional character. Choose instruments that fit the exact scene.
{style_line}
RULES:
- Write a compact cinematic / ambient music cue, not literary analysis
- Include an exact BPM that fits the scene, a genre/tradition or reference style, and a clear musical character
- Name at least 4 specific instruments, textures, or production details
- Describe what the music sounds like RIGHT NOW; avoid long-form structure or multi-part composition plans
- Prefer instrumental, lyric-free music suitable for reading, but allow intensity when the scene demands it
- Be concrete; avoid vague words like "atmosphere" or "soundscape" unless paired with specific instruments/textures

GOOD prompts:
- "Extremely slow funeral dirge at 50 BPM. Solo muted cello long bow, sparse low timpani heartbeats, distant church bell. Vast empty space between notes. Stillness, dread, military formality."
- "Warm Southern pastoral, 80 BPM waltz. Fingerpicked parlor guitar, gentle front-porch banjo, solo fiddle melody, soft upright bass. Nostalgic Americana, deceptive calm, dusty sunlight."
- "Dark orchestral horror score, 90 BPM. Sustained low brass drone, thunderous timpani rolls, aggressive low string ostinato, shrill violin tremolos, gong hits. Massive, panicked, claustrophobic."
- "Dreamlike drifting, 70 BPM. Ethereal aeolian harp glissandi, reversed orchestral swells, bowed vibraphone, glass harmonica, breathy wordless choir. Floating, hallucinatory, time-dilated."

BAD prompts:
- "Tense and mysterious atmosphere" (no instruments)
- "Dark soundscape building to a climax" (too vague, describes structure)
- "Beautiful emotional music for this passage" (generic)
- "A song about escape and fear" (lyrics/story summary instead of musical description)

TEXT:
\"\"\"{text}\"\"\"

Write a 45-75 word prompt with exact BPM, specific instruments/textures, genre or reference style, and the scene's emotional quality. Your negative_prompt should list concrete musical elements to avoid."""


class MusicPrompt(BaseModel):
    prompt: str = Field(description="Music prompt (45-75 words) with exact BPM, specific instruments/textures, style, and mood", max_length=600)
    negative_prompt: str = Field(description="Concrete musical elements to AVOID", max_length=300)


# ============================================================
# Routes: Static pages
# ============================================================

@app.route("/")
@app.route("/prepare")
def prepare_page():
    return send_from_directory("static", "prepare.html")


@app.route("/reader")
def reader_page():
    return send_from_directory("static", "reader.html")


# ============================================================
# Routes: PDF & Segments
# ============================================================

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files supported"}), 400

    save_path = UPLOADS_DIR / f.filename
    f.save(str(save_path))

    import fitz
    doc = fitz.open(str(save_path))
    page_count = len(doc)
    doc.close()

    return jsonify({"filename": f.filename, "page_count": page_count})


@app.route("/split", methods=["POST"])
def split_pdf():
    data = request.json or {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "No filename"}), 400

    pdf_path = UPLOADS_DIR / filename
    if not pdf_path.exists():
        return jsonify({"error": "File not found"}), 404

    book_slug = re.sub(r'\W+', '_', pdf_path.stem).lower()
    output_dir = SEGMENTS_DIR / book_slug

    # Clear stale segment files from previous runs
    if output_dir.exists():
        for old in output_dir.glob("segment*.txt"):
            old.unlink()
        meta = output_dir / "_segments.json"
        if meta.exists():
            meta.unlink()

    from pdf_splitter import split_pdf_to_segments

    result = split_pdf_to_segments(
        pdf_path=str(pdf_path),
        output_dir=str(output_dir),
        book_title=data.get("book_title"),
        chapter_limit=int(data["chapter_limit"]) if data.get("chapter_limit") else None,
        page_start=int(data["page_start"]) if data.get("page_start") else None,
        page_end=int(data["page_end"]) if data.get("page_end") else None,
        provider_name=data.get("provider") or None,
        model=data.get("model") or None,
    )

    meta_path = output_dir / "_segments.json"
    meta_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return jsonify(result)


@app.route("/segments")
def list_books():
    books = []
    if SEGMENTS_DIR.exists():
        for d in sorted(SEGMENTS_DIR.iterdir()):
            if d.is_dir() and not d.name.startswith("_"):
                books.append(d.name)
    return jsonify(books)


@app.route("/segments/<book>")
def list_segments(book):
    book_dir = SEGMENTS_DIR / book
    if not book_dir.exists():
        return jsonify({"error": "Book not found"}), 404

    meta_path = book_dir / "_segments.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        return jsonify(meta.get("segments", []))

    segments = []
    for f in sorted(book_dir.glob("segment*.txt")):
        match = re.search(r"segment(\d+)", f.stem)
        num = int(match.group(1)) if match else 0
        segments.append({"segment_number": num, "txt_path": str(f)})
    return jsonify(segments)


@app.route("/segments/<book>/<int:num>/text")
def get_segment_text(book, num):
    book_dir = SEGMENTS_DIR / book
    txt_path = book_dir / f"segment{num:02d}.txt"
    if not txt_path.exists():
        return jsonify({"error": "Segment not found"}), 404
    from pdf_splitter import _clean_text
    text = _clean_text(txt_path.read_text(encoding="utf-8"))
    return jsonify({"text": text, "char_count": len(text)})


# ============================================================
# Routes: Prompt Generation
# ============================================================

@app.route("/generate-prompt", methods=["POST"])
def generate_prompt():
    data = request.json or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    provider_name = data.get("provider")
    model = data.get("model")
    style = data.get("style", "").strip()

    style_line = (
        f"STYLE PREFERENCE: {STYLE_HINTS[style]}. "
        f"Use these instruments to express the mood — keep the emotional analysis "
        f"from the text but anchor the sound world to this style."
        if style and style in STYLE_HINTS else ""
    )
    ai_prompt = AI_PROMPT_TEMPLATE.replace("{text}", text[:4000]).replace("{style_line}", style_line)

    from providers import get_provider
    try:
        provider = get_provider(provider_name, model)
        result = provider.generate_structured(ai_prompt, MusicPrompt)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "prompt": result.get("prompt", ""),
        "negative_prompt": result.get("negative_prompt", ""),
    })


@app.route("/generate-all", methods=["POST"])
def generate_all():
    """Generate prompts for all segments of a book sequentially with rolling arc context."""
    data = request.json or {}
    book = data.get("book")
    if not book:
        return jsonify({"error": "No book specified"}), 400

    book_dir = SEGMENTS_DIR / book
    if not book_dir.exists():
        return jsonify({"error": "Book not found"}), 404

    provider_name = data.get("provider")
    model = data.get("model")
    style = data.get("style", "").strip()
    style_line = (
        f"STYLE PREFERENCE: {STYLE_HINTS[style]}. "
        f"Use these instruments to express the mood — keep the emotional analysis "
        f"from the text but anchor the sound world to this style."
        if style and style in STYLE_HINTS else ""
    )

    from providers import get_provider

    prompt_dir = PROMPTS_DIR / book
    prompt_dir.mkdir(parents=True, exist_ok=True)

    regenerate = data.get("regenerate", False)
    if regenerate:
        for old in prompt_dir.glob("segment*.json"):
            old.unlink()

    segment_files = sorted(book_dir.glob("segment*.txt"))
    total = len(segment_files)
    results = []
    previous_prompt = ""
    skipped = 0

    for i, seg_file in enumerate(segment_files):
        match = re.search(r"segment(\d+)", seg_file.stem)
        seg_num = int(match.group(1)) if match else 0

        # Skip segments that already have a non-empty prompt
        out_path = prompt_dir / f"segment{seg_num:02d}.json"
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if existing.get("prompt", "").strip():
                    previous_prompt = existing["prompt"]
                    results.append(existing)
                    skipped += 1
                    print(f"  Segment {seg_num}: [kept existing]")
                    continue
            except Exception:
                pass

        from pdf_splitter import _clean_text
        text = _clean_text(seg_file.read_text(encoding="utf-8"))
        ai_prompt = AI_PROMPT_TEMPLATE.replace("{text}", text[:4000]).replace("{style_line}", style_line)

        if previous_prompt:
            arc_lines = [
                "ARC CONTEXT (read before writing the prompt):",
                f"- This is segment {i + 1} of {total}.",
                f"- Previous segment's music prompt: \"{previous_prompt}\"",
                "- Stay in the same sonic world. Progress naturally — evolve the palette, do NOT reinvent it.",
            ]
            ai_prompt = "\n".join(arc_lines) + "\n\n" + ai_prompt

        last_error = None
        for attempt in range(2):
            try:
                provider = get_provider(provider_name, model)
                result = provider.generate_structured(ai_prompt, MusicPrompt)

                prompt_data = {
                    "segment_number": seg_num,
                    "prompt": result.get("prompt", ""),
                    "negative_prompt": result.get("negative_prompt", ""),
                }

                out_path.write_text(json.dumps(prompt_data, indent=2), encoding="utf-8")

                previous_prompt = prompt_data["prompt"]
                results.append(prompt_data)
                print(f"  Segment {seg_num}: {prompt_data['prompt'][:60]}…")
                last_error = None
                break
            except Exception as e:
                last_error = e
                print(f"  Attempt {attempt + 1} failed for segment {seg_num}: {e}")
                if attempt == 0:
                    time.sleep(2)

        if last_error is not None:
            print(f"  Giving up on segment {seg_num}: {last_error}")
            results.append({"segment_number": seg_num, "error": str(last_error)})

        time.sleep(0.5)

    if skipped:
        print(f"  Kept {skipped} existing prompts, generated {total - skipped - len([r for r in results if 'error' in r])} new")

    return jsonify({"book": book, "count": len(results), "results": results})


@app.route("/prompts/<book>")
def list_prompts(book):
    prompt_dir = PROMPTS_DIR / book
    if not prompt_dir.exists():
        return jsonify([])
    prompts = []
    for f in sorted(prompt_dir.glob("segment*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        prompts.append(data)
    return jsonify(prompts)


@app.route("/prompts/<book>/<int:num>")
def get_prompt(book, num):
    prompt_path = PROMPTS_DIR / book / f"segment{num:02d}.json"
    if not prompt_path.exists():
        return jsonify({"error": "Prompt not found"}), 404
    return jsonify(json.loads(prompt_path.read_text(encoding="utf-8")))


@app.route("/refine-prompt", methods=["POST"])
def refine_prompt():
    data = request.json or {}
    prompt = data.get("prompt", "").strip()
    instruction = data.get("instruction", "").strip()
    if not prompt or not instruction:
        return jsonify({"error": "prompt and instruction required"}), 400

    negative = data.get("negative_prompt", "").strip()
    # Nudges have their own provider/model settings (separate from the prompt-
    # generation provider) so we can pin this to something fast like Groq.
    from providers import load_settings
    settings = load_settings()
    provider_name = data.get("provider") or settings.get("nudge_provider") or "groq"
    model = data.get("model") or settings.get("nudge_model") or "llama-3.1-8b-instant"

    refine_text = f"""You are NUDGING a music prompt for Lyria, a real-time AI music generator. This is a small adjustment, not a rewrite.

Current prompt: "{prompt}"
Current negative prompt: "{negative}"
User instruction: "{instruction}"

CRITICAL: Make the SMALLEST CHANGE possible to follow the instruction. Keep the overall scene, instrumentation, key/BPM, and stylistic references intact. Change only the adjectives or a couple of words that relate directly to the instruction. The result should sound like the SAME PIECE with a slight shift in character - not a different composition.

Examples:
- "make it more peaceful" -> swap "rapid/agitated/frantic" for "gentle/flowing/unhurried"; do NOT remove instruments, do NOT change the scene description.
- "add drums" -> add "soft brushed snare" (or similar) to the existing prompt; do not remove anything.
- "darker" -> adjust mood words ("warm" -> "shadowed", "hopeful" -> "uneasy"); keep instruments the same.

Stay in the same word-count range as the original (20-40 words). Describe the sound RIGHT NOW, not a progression.

Rules for the negative_prompt:
- If the user asks to ADD something currently in the negative prompt, you MUST REMOVE it from the negative prompt - otherwise Lyria will keep avoiding it.
- If the user asks to REMOVE/reduce something, add that thing to the negative prompt.
- Otherwise leave the negative prompt ALONE. Do not add new items just because the prompt changed tone.
- Never include anything in both the positive and negative prompt."""

    from providers import get_provider
    from pydantic import BaseModel, Field as PField

    class RefinedPrompt(BaseModel):
        prompt: str = PField(description="15-40 word music prompt with specific instruments and mood")
        negative_prompt: str = PField(description="Elements to avoid, comma-separated", max_length=300)

    print(f"\n[nudge] provider={provider_name} model={model}")
    print(f"[nudge] instruction: {instruction}")
    print(f"[nudge] before prompt: {prompt}")
    print(f"[nudge] before negative: {negative}")

    try:
        provider = get_provider(provider_name, model)
        result = provider.generate_structured(refine_text, RefinedPrompt)
    except Exception as e:
        print(f"[nudge] ERROR: {type(e).__name__}: {e}")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    print(f"[nudge] after prompt: {result.get('prompt', '')}")
    print(f"[nudge] after negative: {result.get('negative_prompt', '')}\n")

    return jsonify({
        "prompt": result.get("prompt", ""),
        "negative_prompt": result.get("negative_prompt", ""),
    })


@app.route("/save-prompt", methods=["POST"])
def save_prompt():
    data = request.json or {}
    book = data.get("book")
    num = data.get("segment_number")
    if not book or num is None:
        return jsonify({"error": "book and segment_number required"}), 400

    prompt_dir = PROMPTS_DIR / book
    prompt_dir.mkdir(parents=True, exist_ok=True)

    out_path = prompt_dir / f"segment{int(num):02d}.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return jsonify({"saved": str(out_path)})


# ============================================================
# Routes: Provider / Settings
# ============================================================

@app.route("/providers")
def get_providers():
    from providers import get_available_providers, load_settings
    available = get_available_providers()
    settings = load_settings()

    providers = []
    if available.get("gemini"):
        from providers.gemini_provider import get_gemini_models
        providers.append({
            "id": "gemini",
            "name": "Google Gemini",
            "has_key": bool(os.environ.get("GOOGLE_API_KEY")),
            "models": get_gemini_models() or ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
            "current_model": settings.get("gemini_model", "gemini-2.5-flash"),
        })
    if available.get("groq"):
        from providers.groq_provider import get_groq_models
        providers.append({
            "id": "groq",
            "name": "Groq",
            "has_key": bool(os.environ.get("GROQ_API_KEY")),
            "models": get_groq_models() or ["moonshotai/kimi-k2-instruct-0905"],
            "current_model": settings.get("groq_model", "moonshotai/kimi-k2-instruct-0905"),
        })
    if available.get("anthropic"):
        from providers.anthropic_provider import get_anthropic_models
        providers.append({
            "id": "anthropic",
            "name": "Anthropic (Claude)",
            "has_key": bool(os.environ.get("CLAUDE_API") or os.environ.get("ANTHROPIC_API_KEY")),
            "models": get_anthropic_models(),
            "current_model": settings.get("anthropic_model", "claude-sonnet-4-6"),
        })
    if available.get("lmstudio"):
        providers.append({
            "id": "lmstudio",
            "name": "LM Studio",
            "has_key": True,
            "models": [settings.get("lmstudio_model", "local-model")],
            "current_model": settings.get("lmstudio_model", "local-model"),
        })
    return jsonify(providers)


@app.route("/settings", methods=["GET"])
def get_settings():
    from providers import load_settings
    return jsonify(load_settings())


@app.route("/settings", methods=["POST"])
def update_settings():
    from providers import load_settings, save_settings
    current = load_settings()
    current.update(request.json or {})
    save_settings(current)
    return jsonify(current)


# ============================================================
# Routes: Lyria Player
# ============================================================

@app.route("/player/start", methods=["POST"])
def player_start():
    global player_process, player_config

    if player_process and player_process.poll() is None:
        return jsonify({"error": "Player already running"}), 400

    data = request.json or {}
    if data:
        player_config = data

    global player_last_chunk_ts
    player_last_chunk_ts = 0.0

    player_process = subprocess.Popen(
        [sys.executable, str(BASE_DIR / "player.py")],
        cwd=str(BASE_DIR),
    )
    return jsonify({"status": "started", "pid": player_process.pid})


@app.route("/player/stop", methods=["POST"])
def player_stop():
    global player_process, player_last_chunk_ts
    player_last_chunk_ts = 0.0
    if player_process and player_process.poll() is None:
        player_process.terminate()
        try:
            player_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            player_process.kill()
        player_process = None
        return jsonify({"status": "stopped"})
    player_process = None
    return jsonify({"status": "not running"})


@app.route("/player/restart", methods=["POST"])
def player_restart():
    """Restart the Lyria subprocess while preserving the current config."""
    global player_process, player_last_chunk_ts, player_last_restart_ts

    now = time.time()
    if now - player_last_restart_ts < 5:
        return jsonify({"status": "restart throttled"}), 429
    player_last_restart_ts = now
    player_last_chunk_ts = 0.0

    if player_process and player_process.poll() is None:
        player_process.terminate()
        try:
            player_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            player_process.kill()

    player_process = subprocess.Popen(
        [sys.executable, str(BASE_DIR / "player.py")],
        cwd=str(BASE_DIR),
    )
    return jsonify({"status": "restarted", "pid": player_process.pid})


@app.route("/player/config", methods=["GET"])
def get_player_config():
    return jsonify(player_config)


@app.route("/player/config", methods=["POST"])
def set_player_config():
    global player_config
    player_config.update(request.json or {})
    return jsonify({"status": "updated"})


@app.route("/player/status")
def player_status():
    running = player_process is not None and player_process.poll() is None
    seconds_since_chunk = (time.time() - player_last_chunk_ts) if player_last_chunk_ts else None
    audio_alive = seconds_since_chunk is not None and seconds_since_chunk < 3.0
    return jsonify({
        "running": running,
        "audio_alive": audio_alive,
        "seconds_since_chunk": seconds_since_chunk,
    })


@app.route("/player/heartbeat", methods=["POST"])
def player_heartbeat():
    global player_last_chunk_ts
    player_last_chunk_ts = time.time()
    return ("", 204)


@app.route("/player/pause", methods=["POST"])
def player_pause():
    global player_config
    player_config["_paused"] = True
    return jsonify({"status": "paused"})


@app.route("/player/resume", methods=["POST"])
def player_resume():
    global player_config
    player_config["_paused"] = False
    return jsonify({"status": "resumed"})


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import logging
    # Suppress polling spam — only log non-polling requests
    wlog = logging.getLogger("werkzeug")
    _poll_paths = {"/player/config", "/player/status", "/player/heartbeat"}

    class _PollFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return not any(p in msg for p in _poll_paths)

    wlog.addFilter(_PollFilter())

    app.run(debug=True, port=5000)

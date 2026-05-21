# BookAmbience

BookAmbience is a prototype for adaptive AI-generated music during narrative reading. It takes a PDF, segments the text around narrative shifts, generates music prompts for the resulting segments, and plays continuous music through Google Lyria RealTime while changing prompt as each segment gets read. 

The reader interface also includes controls for adjusting the generated music through preset options or free-text instructions.

## Setup

Install the Python dependencies:

```powershell
pip install -r requirements.txt
```

Music generation requires a Google API key with access to Lyria RealTime. Set it before running the application:

```powershell
$env:GOOGLE_API_KEY="your_google_api_key"
```

Segmentation and prompt generation also require an LLM provider. Supported providers include Google Gemini, Anthropic Claude, Groq, and a local OpenAI-compatible model endpoint.

For Anthropic:

```powershell
$env:ANTHROPIC_API_KEY="your_anthropic_api_key"
```

For Groq:

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

## Run

Start the Flask application:

```powershell
python app.py
```

Then open:

```text
http://localhost:5000
```

## Use

1. Upload a PDF.
2. Segment the text.
3. Generate music prompts for the segments.
4. Open the reader.
5. Read while the music adapts as segment boundaries are crossed.

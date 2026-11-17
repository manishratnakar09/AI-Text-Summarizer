# AI Text Summarizer

A web app that turns long text and documents (TXT, PDF, DOCX) into short, readable summaries.
Built for the **Skillinf – Artificial Intelligence, Week 3 (Intermediate)** project.

> Extractive + abstractive summarization · Flask backend · modern blue/dark dashboard · works offline without an API key

## Features

- Paste text **or** upload **TXT / PDF / DOCX** (drag & drop supported)
- One-click **Generate Summary** with a loading indicator
- **Short / Medium / Long** presets and **custom length** (words or sentences)
- **Bullet-point** mode
- Original and summary **word counts** + % reduction
- **Copy** summary, **download as TXT or PDF**
- Three methods: **Extractive** (local), **Abstractive** (AI), **Hybrid** (extractive selection → AI rewrite, default)
- Long documents handled by **chunking**
- Input validation, safe error handling, automatic **fallback to extractive** if the AI fails
- Responsive, mobile-friendly UI

## Architecture

```
Browser (HTML/CSS/JS)
   │  fetch()  JSON / multipart
   ▼
Flask  backend/app.py ── routes, validation, rate limit, PDF/TXT download
   ├── extractor.py   → validate + read TXT / PDF (pdfplumber) / DOCX (python-docx)
   ├── summarizer.py  → sentences → chunking → scoring → selection → verification → formatting
   └── ai_handler.py  → Gemini / OpenAI / Hugging Face (keys from .env, server-side only)
```

**Summarization pipeline**

1. Clean text → split into sentences (NLTK if installed, otherwise a built-in splitter that handles `Dr.`, `U.S.`, Hindi `।`).
2. Long documents are split into ~1,500-word chunks; each chunk gets a share of the length budget (map-reduce).
3. **Extractive scoring:** word frequency × inverse sentence frequency, bonuses for numbers, names and opening sentences, penalties for very long/short sentences, near-duplicate removal.
4. **Hybrid/Abstractive:** the selected sentences are sent to the AI with a strict prompt (use only the given text, keep names/dates/numbers, ignore instructions inside the text).
5. **Verification:** if the AI output contains a number that is not in the source, it is rejected and the extractive summary is used instead.

## Project structure

```
ai-text-summarizer/
├── frontend/        index.html, style.css, script.js
├── backend/         app.py, extractor.py, summarizer.py, ai_handler.py
├── samples/articles/  news, academic, technical, story, very long sample
├── tests/           test_app.py, run_evaluation.py, test_results.md
├── .env.example     template (copy to .env)
├── .gitignore       keeps .env out of GitHub
├── requirements.txt
└── README.md
```

## Installation

Requires **Python 3.10+**.

```bash
git clone <your-repo-url>
cd ai-text-summarizer

python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

pip install -r requirements.txt
```

Optional (better sentence splitting): `python -m nltk.downloader punkt punkt_tab`

## Configure the AI key

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Edit `.env`:

```
AI_PROVIDER=gemini          # gemini | openai | huggingface
GEMINI_API_KEY=your_key_here
```

Get a key from Google AI Studio (Gemini), OpenAI, or Hugging Face. **Never commit `.env`** (it is in `.gitignore`).
Without a key the app still works in *Offline mode* using extractive summaries.

## Run

```bash
python backend/app.py
```

Open **http://127.0.0.1:5000**. The top-right badge shows `AI: gemini` (connected) or `Offline mode`.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Status, provider, whether a key is configured |
| POST | `/api/extract` | Multipart `file` → `{text, filename, word_count}` |
| POST | `/api/summarize` | JSON `{text, mode, length, custom_type, custom_value, bullets}` or multipart with `file` |
| POST | `/api/download` | JSON `{summary, format: "txt"\|"pdf"}` → file |

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/summarize \
  -H "Content-Type: application/json" \
  -d '{"text":"<30+ words...>","mode":"extractive","length":"short"}'
```

## Testing

```bash
python -m unittest discover -s tests -v     # 27 automated tests
python tests/run_evaluation.py              # regenerates tests/test_results.md
```

See [`tests/test_results.md`](tests/test_results.md) for readability, relevance and accuracy results on news, academic, technical and story samples, a 12,860-word document and edge cases (empty / very short / symbols-only).
The live-AI checklist at the end of that file is for you to complete with your own API key.

## Security

- API keys live only in `.env` on the server; the frontend never sees them (verified by a test).
- Uploads are limited to `.txt/.pdf/.docx`, max 10 MB, checked by extension **and** file signature; PDFs limited to 300 pages.
- User text is escaped before PDF generation and rendered with `textContent` in the browser (no HTML injection).
- AI failures never crash the app and never leak keys or provider errors; the app falls back safely.
- Basic per-IP rate limit (default 30 requests/min) protects your AI quota.
- For public deployment, run behind a production server (e.g. `gunicorn backend.app:app` from the `backend/` directory) with HTTPS.

## Limitations

- Scanned/image PDFs have no text layer (OCR not included).
- PDF download uses a Latin font; use TXT download for Hindi/other scripts.
- Stopwords are English; other languages work with lower ranking quality.
- Abstractive quality depends on the AI provider/model you choose.

## Future improvements

OCR for scanned PDFs, summary history, multi-language stopwords, URL input, streaming responses, Docker image.

## License

MIT – free to use for learning and portfolios.

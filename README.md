# Webtoon Translator (Chrome Extension & Processing Engine)

> **Read Korean webtoons in real-time.** As you scroll down the page on original Korean sites like Naver Webtoon and Kakao Webtoon, speech bubbles and dialogue are translated directly into natural English with automatic background inpainting, typesetting, and per-series glossary consistency.

---

## Key Features

- **Auto-Detection as You Scroll**: Intersection Observer identifies comic strips and prefetches upcoming panels ahead of the reader.
- **Natural Speech Bubble Typesetting**: Binary-search font fitting wraps English dialogue neatly inside bubble contours with comic lettering.
- **Adaptive Text Inpainting**: Cleanly erases original Korean text while preserving bubble borders and background artwork.
- **Context-Aware Dialogue Translation**: Sends entire panel dialogue sequences together to preserve character tone, pronouns, and honorifics (*Hyung*, *Oppa*, *Sunbae*).
- **Interactive Language Learner Mode**: Hover over any speech balloon to inspect the original Korean Hangul, phonetic romanization, and vocabulary breakdowns.
- **Fast Content Caching (SHA-256)**: Panels are hashed by content digest, making re-visits instantaneous (0ms) and saving translation costs.
- **Per-Series Glossary**: Enforce character names (*성진우* → *Sung Jin-woo*) and terminology across chapters.

---

## System Architecture

```
┌──────────────────────────────────────────────┐
│        Chrome Browser Extension (MV3)        │
│  - Content Script (Naver / Kakao Adapters)   │
│  - Scroll Intersection Observer              │
│  - Dual Mode (Image Swap / Interactive)      │
│  - Floating Control Widget & Popup UI        │
└──────────────────────┬───────────────────────┘
                       │ HTTP REST (JSON + Base64)
                       ▼
┌──────────────────────────────────────────────┐
│        Python Pipeline Backend (Flask)       │
│  1. Detector     : Bubble contours & boxes   │
│  2. OCR          : Hangul & reading order    │
│  3. Translator   : Gemini / OpenAI / Offline │
│  4. Inpainter    : Adaptive bubble fill      │
│  5. Typesetter   : Binary-search font wrap   │
│  6. Storage      : SHA-256 Panel Cache       │
└──────────────────────────────────────────────┘
```

---

## Quickstart Guide

### 1. Start the Backend Server

The backend runs locally on Python 3.10:

```powershell
# Activate the virtual environment
.venv\Scripts\activate

# Launch the processing server
python -m server.main
```

The server binds to `http://127.0.0.1:8000`. You can verify health by visiting `http://127.0.0.1:8000/api/health`.

### 2. Load the Extension into Chrome / Edge

1. Open your browser and navigate to `chrome://extensions` (or `edge://extensions`).
2. Enable **Developer mode** (toggle in top-right corner).
3. Click **Load unpacked**.
4. Select the `extension/` directory from this project folder (`d:\webtoon translator\extension`).
5. The **Webtoon Translator** extension icon (⚡) will appear in your browser toolbar!

### 3. Test on the Demo Viewer

Open the built-in standalone reader in your browser:
- Open [`demo/index.html`](file:///d:/webtoon%20translator/demo/index.html) directly in Chrome.
- You will see high-resolution Korean manhwa panels.
- With the extension active, scroll down to see the panels automatically translate into English!
- Hover over any speech balloon to view the Hangul and romanization learning card.
- Press **Alt + T** or click the floating bottom-right pill to toggle between Korean and English.

---

## Configuration & Environment Variables

Copy or create a `.env` file in the project root to configure cloud LLM translation providers:

```ini
# Server Network
SERVER_HOST=127.0.0.1
SERVER_PORT=8000

# Translation Engine: 'heuristic', 'gemini', or 'openai'
TRANSLATION_PROVIDER=gemini

# Google Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# OpenAI / Ollama Configuration (optional alternative)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

*Note: Without any API key, the engine defaults to the high-speed heuristic pipeline, which translates common manhwa dialogue, honorifics, and vocabulary offline.*

---

## Running the Automated Test Suite

Run the full pytest suite:

```powershell
.venv\Scripts\pytest -v
```

All 10 integration and unit tests will run:
- Speech bubble detection & reading-order sorting
- Hangul text cleaning & OCR normalization
- Contextual dialogue translation & glossary enforcement
- Inpainting background variance tests
- Typesetting font scaling & boundary fitting
- REST API `/api/health`, `/api/translate`, and `/api/glossary` endpoints

---

## Project Structure

```
webtoon translator/
├── server/
│   ├── main.py              # Flask REST API entry point
│   ├── config.py            # Settings & environment configuration
│   ├── schemas.py           # Dataclasses (BoundingBox, BubbleData, etc.)
│   ├── pipeline/
│   │   ├── detector.py      # Bubble & text contour detection
│   │   ├── ocr.py           # Hangul OCR & topological reading sorter
│   │   ├── translator.py    # Contextual chapter translation & vocab
│   │   ├── inpainter.py     # Background erasure & color restoration
│   │   ├── typesetter.py    # Binary-search typography engine
│   │   └── stitcher.py      # Long strip slicer & seam de-duplicator
│   ├── cache/
│   │   └── storage.py       # SHA-256 panel cache (memory + disk)
│   └── services/
│       └── glossary.py      # Series-specific glossary manager
├── extension/
│   ├── manifest.json        # Chrome Manifest V3 configuration
│   ├── background/
│   │   └── service_worker.js# CORS bypass fetch & API coordinator
│   ├── content/
│   │   ├── content_script.js# Site adapters, observer, and overlay renderer
│   │   └── content_style.css# Floating reader pill & learner tooltip styles
│   ├── popup/
│   │   ├── popup.html       # Sleek dark-mode extension popup
│   │   ├── popup.css        # Glassmorphic UI styles
│   │   └── popup.js         # Settings & latency monitor
│   └── options/
│       ├── options.html     # Server settings & glossary manager
│       ├── options.css      # Table & form styles
│       └── options.js       # Glossary CRUD controller
├── demo/
│   ├── index.html           # Naver Webtoon viewer simulator
│   └── assets/              # Sample manhwa panels
└── tests/
    ├── create_sample_panel.py# Manhwa strip test fixture generator
    ├── test_pipeline.py     # Core pipeline unit tests
    └── test_api.py          # REST API integration tests
```

---

## License

MIT License. Designed and built with zero slop and clean, human-engineered code.

# ⚡ Webtoon Translator

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Chrome MV3](https://img.shields.io/badge/chrome%20extension-Manifest%20V3-4285F4.svg?logo=googlechrome&logoColor=white)](https://developer.chrome.com/docs/extensions/mv3/intro/)
[![Model](https://img.shields.io/badge/model-NLLB--200%20(Fine--Tuned)-FF6F00.svg?logo=huggingface&logoColor=white)](https://huggingface.co/facebook/nllb-200-distilled-600M)
[![Inference](https://img.shields.io/badge/inference-CTranslate2%20INT8-8A2BE2.svg)](https://opennmt.net/CTranslate2/)
[![BLEU Score](https://img.shields.io/badge/validation%20BLEU-34.20-2ea44f.svg)](https://en.wikipedia.org/wiki/BLEU)
[![Privacy](https://img.shields.io/badge/privacy-100%25%20Offline%20First-success.svg)](https://github.com/kancharlanavyatha/manhwa-translator)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

<p align="center">
  <b>Read Korean webtoons in real-time with zero cloud fees.</b><br>
  An end-to-end, privacy-focused Chrome Extension and local computer-vision backend that detects speech bubbles, extracts Hangul via OCR, translates dialogue with a custom fine-tuned neural model, inpaints backgrounds, and re-typesets English dialogue with natural comic typography as you scroll.
</p>

[Key Features](#-key-features) •
[Architecture](#-system-architecture) •
[Tech Stack](#-tech-stack) •
[Model & Benchmarks](#-model--benchmarks) •
[Quickstart](#-quickstart-guide) •
[Configuration](#-configuration--environment-variables) •
[Project Structure](#-project-structure)

</div>

---

## 📖 Overview

Korean webtoon platforms such as **Naver Webtoon** and **Kakao Webtoon** deliver vertical comic strips where speech bubbles, sound effects, and narration are baked directly into image pixels. Traditional browser translation tools (such as Google Translate) cannot translate text embedded inside images or HTML5 canvas elements.

**Webtoon Translator** solves this by providing a unified, real-time client-server translation pipeline:
1. **Frontend (Chrome MV3 Extension)**: Attaches an `IntersectionObserver` to the comic reader DOM to prefetch and stream visible and upcoming panels.
2. **Backend (Python 3.10 Engine)**: Locates speech bubbles, performs OCR on Korean Hangul, translates dialogue through a **fine-tuned NLLB-200** model (quantized with CTranslate2), cleanly inpaints original artwork, and dynamically typesets English text inside bubble contours.

---

## 🚀 Key Features

* **⚡ Real-Time Scroll Translation**: Uses an asynchronous `IntersectionObserver` with lookahead buffers to process and translate comic panels ahead of the reader for a seamless, lag-free reading experience.
* **🧠 Custom Fine-Tuned Neural Translator**: Powered by a fine-tuned `facebook/nllb-200-distilled-600M` model trained on 50,000 Korean-English conversational manhwa dialogue pairs (**34.20 BLEU Score**).
* **🔒 100% Offline & Private**: Runs entirely locally on CPU via CTranslate2 INT8 quantization. No API keys, cloud subscriptions, or external network requests are required.
* **🎨 Adaptive Background Inpainting**: Erases Korean Hangul using color-variance thresholding and contour-aware Telea inpainting, preserving speech bubble borders, gradients, and background manga illustrations.
* **📐 Dynamic Binary-Search Typesetting**: Automatically calculates optimal font sizing, hyphenation, and line wrapping to balance English text perfectly within irregular speech bubble contours.
* **📖 Interactive Language Learner Mode**: Hover over any speech bubble to inspect the original Korean Hangul, phonetic Revised Romanization, and vocabulary breakdowns.
* **⚡ SHA-256 In-Memory & Disk Caching**: Deduplicates panels using content hashing, delivering 0ms instantaneous loading when re-reading or scrolling backwards.
* **📚 Per-Series Glossary & Honorific Preservation**: Enforces custom character names (*성진우* → *Sung Jin-woo*) and context-appropriate Korean honorifics (*Hyung*, *Oppa*, *Sunbae*, *Noona*).

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                     Chrome Browser Extension (MV3)                     │
│  - Content Script (Naver / Kakao Site Adapters)                        │
│  - IntersectionObserver (Viewport & Lookahead Panel Prefetching)       │
│  - Dual Display Engine (Seamless Image Replacement / Overlay Mode)     │
│  - Floating Quick-Toggle Pill & Interactive Language Learner Tooltips  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP REST (Base64 Image / JSON)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     Python Processing Engine (Flask)                   │
│                                                                        │
│  1. Detector     : OpenCV bubble contour segmentation & bounding boxes │
│  2. Sorter       : Topological reading order (top-to-bottom, L-to-R)   │
│  3. OCR          : RapidOCR (Korean Hangul text recognition)           │
│  4. Translator   : Fine-Tuned NLLB-200 INT8 via CTranslate2 (Offline)  │
│  5. Inpainter    : Color-moment fill & Telea background restoration    │
│  6. Typesetter   : Binary-search font scaler & auto text wrap          │
│  7. Cache Layer  : SHA-256 content-hash storage (0ms re-visits)        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Browser Extension** | Chrome Manifest V3, Vanilla JavaScript, CSS3 | Content scripts, service worker, site adapters, glassmorphic UI |
| **Server Framework** | Python 3.10, Flask, Flask-CORS | High-throughput REST API serving `/api/translate` and `/api/health` |
| **Computer Vision** | OpenCV (`cv2`), NumPy, Pillow | Bubble contour detection, mask dilation, inpainting, typesetting |
| **Text Recognition** | RapidOCR (ONNX Runtime) | Fast, CPU-friendly Korean Hangul OCR with custom line clustering |
| **Translation Engine** | Meta NLLB-200, CTranslate2, SentencePiece | Fine-tuned 600M parameter model quantized to INT8 for CPU speed |
| **Model Training** | PyTorch, Hugging Face Transformers, Colab GPU | Sequence-to-sequence fine-tuning on conversational dialogue datasets |
| **Testing Suite** | Pytest, Requests | Automated unit tests covering OCR, translation, inpainting, and API |

---

## 📊 Model & Benchmarks

The translation pipeline uses a specialized fine-tuned version of Meta's **NLLB-200-distilled-600M**:

* **Training Corpus**: 50,000 conversational dialogue pairs from the AI Hub Korean-English dataset, tailored specifically for conversational, informal, and dialogue-heavy text.
* **Validation BLEU Score**: **34.20** on held-out dialogue test pairs.
* **Quantization**: Converted to **CTranslate2 INT8**, reducing memory footprint to ~580MB and providing sub-second inference times on standard multi-core consumer CPUs without requiring a dedicated GPU.

---

## 🚀 Quickstart Guide

### Prerequisites
* **Python 3.10+**
* **Google Chrome** (or Chromium-based browser such as Brave, Edge, or Opera)
* **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/kancharlanavyatha/manhwa-translator.git
cd manhwa-translator
```

### 2. Set Up the Python Backend
```powershell
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Download the Fine-Tuned Model Weights
Ensure the quantized model files are placed in `server/models/nmt_webtoon/`:
```
server/models/nmt_webtoon/
├── model.bin               # INT8 quantized CTranslate2 weights
├── sentencepiece.bpe.model # NLLB-200 subword tokenizer
├── shared_vocabulary.json  # Model vocabulary
└── tokenizer.json
```

### 4. Start the Backend Server
```powershell
python -m server.main
```
The server will initialize models and start listening on `http://127.0.0.1:8000`. You can confirm server status at `http://127.0.0.1:8000/api/health`.

### 5. Load the Chrome Extension
1. Open Chrome and navigate to `chrome://extensions`.
2. Toggle on **Developer mode** in the top-right corner.
3. Click **Load unpacked**.
4. Select the `extension/` directory from this repository.
5. The **Webtoon Translator** extension icon (⚡) will appear in your extensions toolbar.

### 6. Test with the Built-In Demo
1. Open [`demo/index.html`](demo/index.html) directly in Chrome.
2. Scroll through the sample manhwa chapter panels.
3. Observe real-time translation, clean background inpainting, and typeset English speech bubbles!
4. Hover over any dialogue balloon to inspect the original Hangul and vocabulary breakdown.

---

## ⚙️ Configuration & Environment Variables

### Do I need to configure API keys?
**No configuration is required!**  
By default, the engine runs **100% offline and free** using the locally fine-tuned **NLLB-200 CTranslate2 INT8 model** (`server/models/nmt_webtoon`). You do not need any third-party API keys or cloud subscriptions.

### Optional Cloud LLM Fallback (Development & Comparison)
If you wish to benchmark or compare local model output against commercial cloud LLMs (such as Google Gemini or OpenAI GPT-4o-mini), copy `.env.example` to `.env` and configure credentials:

```ini
# Server Network Settings
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
CORS_ORIGINS=*

# Translation Provider: 'local_nmt' (default), 'gemini', or 'openai'
TRANSLATION_PROVIDER=local_nmt
NMT_MODEL_DIR=server/models/nmt_webtoon

# (Optional) Google Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# (Optional) OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `SERVER_HOST` | `127.0.0.1` | Network interface for backend server |
| `SERVER_PORT` | `8000` | Port for REST API server |
| `TRANSLATION_PROVIDER` | `local_nmt` | Active engine: `local_nmt` (offline), `gemini`, or `openai` |
| `NMT_MODEL_DIR` | `server/models/nmt_webtoon` | Path to local CTranslate2 INT8 weights directory |
| `GEMINI_API_KEY` | *(empty)* | Optional: Google Gemini API key if using cloud translation |
| `OPENAI_API_KEY` | *(empty)* | Optional: OpenAI API key if using cloud translation |

---

## 🧪 Running the Automated Test Suite

Execute the comprehensive unit and integration test suite:

```powershell
.venv\Scripts\pytest -v
```

The test suite validates:
* Speech bubble contour detection & geometry calculations.
* Reading-order topological sorting (top-to-bottom, right-to-left / left-to-right).
* Hangul OCR text extraction and whitespace normalization.
* Local NMT dialogue translation and custom glossary term substitution.
* Inpainting background color variance and artifact minimization.
* Typography font-size scaling and boundary box wrapping.
* REST API endpoints: `/api/health`, `/api/translate`, and `/api/glossary`.

---

## 📁 Project Structure

```
manhwa-translator/
├── extension/                       # Chrome Extension (Manifest V3)
│   ├── manifest.json                # Extension configuration & permissions
│   ├── background/
│   │   └── service_worker.js        # Background worker & REST API proxy
│   ├── content/
│   │   ├── content_script.js        # DOM observer, panel swapper, and tooltip engine
│   │   └── content_style.css        # Reader UI, floating pill, and tooltip styles
│   ├── popup/
│   │   ├── popup.html               # Extension action popup
│   │   ├── popup.css                # Dark-mode glassmorphic styling
│   │   └── popup.js                 # Mode toggle & server connection monitor
│   └── options/
│       ├── options.html             # Settings & glossary configuration page
│       ├── options.css              # Settings layout styles
│       └── options.js               # Glossary CRUD controller
├── server/                          # Python Processing Engine
│   ├── main.py                      # Flask application entry point
│   ├── config.py                    # Environment settings & hyperparameter defaults
│   ├── schemas.py                   # Data contracts (BoundingBox, BubbleData, etc.)
│   ├── pipeline/
│   │   ├── detector.py              # Contour segmentation & bubble localization
│   │   ├── ocr.py                   # RapidOCR Hangul text extraction & line clustering
│   │   ├── translator.py            # Local NMT (CTranslate2) & context translation
│   │   ├── inpainter.py             # Telea & color-fill background restoration
│   │   ├── typesetter.py            # Binary-search typography & text wrapping
│   │   └── stitcher.py              # Panel slicing & seam alignment
│   ├── cache/
│   │   └── storage.py               # SHA-256 panel caching layer
│   └── services/
│       └── glossary.py              # Series-specific glossary manager
├── training/                        # Model Training & Export
│   ├── prepare_data.py              # AI Hub dataset streamer & preprocessor
│   ├── train_translator.py          # PyTorch Seq2SeqTrainer fine-tuning script
│   └── webtoon_nmt_finetuning.ipynb # Google Colab GPU training notebook
├── demo/                            # Standalone Webtoon Reader Demo
│   ├── index.html                   # Interactive reader interface
│   └── assets/                      # Sample raw Korean manhwa panels
└── tests/                           # Verification & Integration Tests
    ├── create_sample_panel.py       # Test panel generator
    ├── test_pipeline.py             # Pipeline unit tests
    └── test_api.py                  # Server REST API integration tests
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to check out the [issues page](https://github.com/kancharlanavyatha/manhwa-translator/issues) if you want to contribute.

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'feat: Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

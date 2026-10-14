# Webtoon Translator: Architectural Deep Dive

This document outlines the engineering principles, algorithms, and design choices implemented in the Webtoon Translator system.

---

## 1. Pipeline Stages

```
   [Raw Comic Slice]
           │
           ▼
┌───────────────────────┐
│ 1. Detection Engine   │  Thresholding, contour geometry filtering (solidity,
│ (Heuristic / ML)      │  aspect ratio), text gradient density.
└──────────┬────────────┘
           │ Bounding Boxes [x, y, w, h]
           ▼
┌───────────────────────┐
│ 2. Reading Order      │  Topological sorter clusters boxes by vertical Y bands
│    Sorter             │  (80px rows) and orders left-to-right.
└──────────┬────────────┘
           │ Ordered Boxes
           ▼
┌───────────────────────┐
│ 3. Hangul OCR         │  EasyOCR / Tesseract / Heuristic with NFKC unicode
│                       │  normalization and quote standardization.
└──────────┬────────────┘
           │ Raw Dialogue Strings
           ▼
┌───────────────────────┐
│ 4. Contextual         │  Single prompt containing entire panel dialogue flow
│    Translation        │  + series glossary. Preserves honorifics & tone.
└──────────┬────────────┘
           │ Translated English + Vocab Breakdown
           ▼
┌───────────────────────┐
│ 5. Inpainting Engine  │  Adaptive Bubble Fill: Samples border pixels to deduce
│                       │  median fill color + Telea for textured art.
└──────────┬────────────┘
           │ Cleaned Canvas
           ▼
┌───────────────────────┐
│ 6. Typesetting Engine │  Binary-search font fitting + greedy word wrapping
│                       │  + anti-aliased PIL rendering with stroke halos.
└──────────┬────────────┘
           │
           ▼
   [Translated Panel]
```

---

## 2. Key Technical Innovations

### Adaptive Speech Bubble Inpainting
Conventional inpainting methods (such as pure Navier-Stokes or Telea) frequently smudge text or create blurry gray patches inside white speech balloons because they propagate neighboring dark edge pixels into the center.

Our `BubbleInpainter` solves this with a two-tier strategy:
1. **Border Color Sampling**: Samples the four perimeter borders of the bounding box to determine the true background color (using `np.median` across BGR channels).
2. **Variance Test**: Checks if the variance along the perimeter is low (`border_variance < 180.0`). If low, the region is an enclosed balloon.
3. **Adaptive Color Replacement & Blending**: Dilates the text stroke mask by 3px, fills the masked area with the sampled background color, and blends the edges with a Gaussian-blurred alpha mask to eliminate pixelated fringes.

### Binary-Search Typography & Typesetting
Speech bubbles have curved, non-rectangular boundaries. Setting font sizes arbitrarily causes text to spill out of the balloon or shrink to illegible micro-text.

The `Typesetter` implements:
- **Inner Padding**: Insets usable area by 15% from the bounding box edge to avoid the curved balloon corners.
- **Binary Search Font Fitting**: Searches between `min_font_size` (11px) and `max_font_size` (42px). For each candidate size, a greedy word wrapper breaks dialogue into lines and verifies both total block height and maximum line width.
- **Centering & Stroke**: Centers the text block vertically and horizontally, rendering text with a subtle 1px white halo (`stroke_width=1`) to guarantee readability against any background.

### Vertical Strip Stitching & Bisected Bubbles
Webtoons are distributed as continuous strips or chopped into slices (e.g. 800x1600). Speech bubbles that span across the cut between slice $N$ and slice $N+1$ will be cut in half, breaking OCR and causing duplicate fragments.

The `WebtoonStripManager`:
- Slices tall strips with a configurable vertical overlap (e.g. 200px).
- Projects detected boxes back into global strip coordinates using the slice offset $y_{\text{offset}}$.
- Performs Intersection-over-Union (IoU) de-duplication: if two candidate boxes overlap by $\ge 40\%$ at the slice seam, they are merged into a single union bounding box.

---

## 3. Scaling & Deployment

1. **Local Prototype Mode**: Runs on CPU using OpenCV and Flask with minimal resource consumption (<100MB RAM).
2. **Serverless GPU Deployment**: The pipeline can be packaged into a Docker container and deployed to RunPod, Modal, or AWS ECS with GPU acceleration.
3. **Aggressive Content Hashing**: Panels are cached using a SHA-256 digest of image bytes + target language + glossary hash. Because webtoons have fixed chapters read by thousands of users, the cache hit rate on popular series approaches 98%, reducing LLM inference costs to near zero for repeat reads.

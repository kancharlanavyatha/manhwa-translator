/**
 * Webtoon Translator - Content Script
 *
 * Runs inside Korean webtoon reading pages (Naver Webtoon, Kakao, and generic viewers).
 * Automatically detects comic panels as you scroll, requests background translation,
 * and swaps in translated speech bubbles or displays interactive language-learning overlays.
 */

(function () {
  "use strict";

  if (window.__webtoonTranslatorInjected) return;
  window.__webtoonTranslatorInjected = true;

  console.log("%c[Webtoon Translator]%c Initializing extension...", "color: #6366f1; font-weight: bold;", "");

  const CONFIG = {
    serverUrl: "http://127.0.0.1:8000",
    targetLang: "en",
    renderMode: "swap", // 'swap' or 'overlay'
    showHoverVocab: true,
  };

  let isTranslatedView = true;
  const enqueuedImages = new WeakSet();
  const panelRegistry = new Map();

  // -------------------------------------------------------------
  // 1. Comic Image Detection
  // -------------------------------------------------------------
  function findComicImages() {
    // Priority 1: Standard webtoon container selectors (Naver Webtoon desktop & mobile, Kakao, Demo)
    const containerImgs = document.querySelectorAll(
      "#comic_view_area img, .wt_viewer img, .comic-panel-img, div[class*='comic'] img, div[class*='viewer'] img, div[id*='comic'] img, div[class*='Viewer'] img, div[id*='view'] img, .toon_view_area img, div[class*='Episode'] img"
    );
    if (containerImgs.length > 0) {
      return Array.from(containerImgs);
    }

    // Priority 2: Generic fallback (tall or wide comic strips)
    return Array.from(document.querySelectorAll("img")).filter((img) => {
      if (img.width > 0 && img.width < 320) return false;
      if (img.height > 0 && img.height < 250) return false;
      const src = (img.src || img.dataset?.src || "").toLowerCase();
      if (src.includes("icon") || src.includes("logo") || src.includes("avatar") || src.includes("banner")) return false;
      return true;
    });
  }

  // -------------------------------------------------------------
  // 2. Sequential Queue & Intersection Observer
  // -------------------------------------------------------------
  const processingQueue = [];
  let isQueuePumping = false;

  function queuePanelImage(img) {
    if (enqueuedImages.has(img) || panelRegistry.has(img)) return;
    enqueuedImages.add(img);
    processingQueue.push(img);
    pumpQueue();
  }

  async function pumpQueue() {
    if (isQueuePumping || processingQueue.length === 0) return;
    isQueuePumping = true;
    const img = processingQueue.shift();
    try {
      await processPanelImage(img);
    } catch (err) {
      console.warn("[Webtoon Translator] Panel processing skipped:", err);
    } finally {
      isQueuePumping = false;
      // 150ms breather between panels to keep the browser 100% smooth and responsive
      setTimeout(pumpQueue, 150);
    }
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          queuePanelImage(entry.target);
        }
      });
    },
    {
      rootMargin: "400px 0px 400px 0px", // Prefetch 400px ahead of viewport
      threshold: 0.01,
    }
  );

  let scanDebounceTimer = null;
  function debouncedScan() {
    if (scanDebounceTimer) clearTimeout(scanDebounceTimer);
    scanDebounceTimer = setTimeout(scanAndObservePanels, 600);
  }

  function isNearViewport(el) {
    const rect = el.getBoundingClientRect();
    const windowH = window.innerHeight || document.documentElement.clientHeight;
    return rect.top < windowH + 600 && rect.bottom > -400;
  }

  function scanAndObservePanels() {
    const images = findComicImages();
    images.forEach((img) => {
      if (!img.dataset.wtObserved) {
        img.dataset.wtObserved = "true";
        observer.observe(img);
      }
      if (isNearViewport(img)) {
        queuePanelImage(img);
      }
    });
    updateWidgetStats();
  }

  // -------------------------------------------------------------
  // 3. High-Performance Base64 Extraction
  // -------------------------------------------------------------
  async function getImageBase64(img) {
    const src = img.currentSrc || img.src || img.dataset?.src || img.dataset?.original;
    if (src && src.startsWith("data:")) {
      return src;
    }

    // Ensure image is loaded before capturing
    if (!img.complete || img.naturalWidth === 0) {
      await new Promise((resolve) => {
        img.addEventListener("load", resolve, { once: true });
        img.addEventListener("error", resolve, { once: true });
        setTimeout(resolve, 800);
      });
    }

    // Method 1: Offscreen Canvas (instant, 0ms latency for same-origin & local)
    try {
      const w = img.naturalWidth || img.width;
      const h = img.naturalHeight || img.height;
      if (w > 0 && h > 0) {
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
        if (dataUrl && dataUrl.length > 500) {
          return dataUrl;
        }
      }
    } catch (_) {
      // Canvas tainted (CORS)
    }

    // Method 2: In-page fetch with native FileReader
    try {
      const resp = await fetch(src);
      if (resp.ok) {
        const blob = await resp.blob();
        return await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result);
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });
      }
    } catch (_) {
      // In-page fetch blocked by CORS
    }

    // Method 3: Background service worker fetch
    const fetchResp = await sendMessageAsync({
      type: "FETCH_IMAGE_B64",
      payload: { url: src },
    });

    if (fetchResp?.success && fetchResp?.base64) {
      return fetchResp.base64;
    }

    throw new Error("Unable to extract panel image data");
  }

  // -------------------------------------------------------------
  // 4. Translation Request & In-Place Rendering
  // -------------------------------------------------------------
  async function processPanelImage(img) {
    if (panelRegistry.has(img)) return;

    const originalSrc = img.currentSrc || img.src || img.dataset?.src || img.dataset?.original;
    if (!originalSrc) return;

    markImageLoading(img, true);

    try {
      const panelIdx = panelRegistry.size;
      const payload = {
        panelIndex: panelIdx,
        targetLang: CONFIG.targetLang,
        mode: "both",
      };

      if (originalSrc.startsWith("http://") || originalSrc.startsWith("https://")) {
        payload.imageUrl = originalSrc;
      } else {
        payload.imageBase64 = await getImageBase64(img);
      }

      // 2. Request translation (Service worker first for CORS/mixed-content bypass, with direct fallback)
      let data = null;
      const swResp = await sendMessageAsync({
        type: "TRANSLATE_IMAGE",
        payload: payload,
      });

      if (swResp?.success && swResp?.data) {
        data = swResp.data;
      } else {
        const bodyObj = {
          panel_index: panelIdx,
          target_lang: CONFIG.targetLang,
          mode: "both",
        };
        if (payload.imageUrl) bodyObj.image_url = payload.imageUrl;
        if (payload.imageBase64) bodyObj.image_base64 = payload.imageBase64;

        const resp = await fetch(`${CONFIG.serverUrl}/api/translate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(bodyObj),
        });

        if (!resp.ok) {
          const errText = await resp.text();
          throw new Error(`Server returned ${resp.status}: ${errText}`);
        }
        data = await resp.json();
      }
      const { translated_image_base64, bubbles, cached, processing_time_ms } = data;

      // 3. Register panel and render
      panelRegistry.set(img, {
        originalSrc: originalSrc,
        translatedSrc: translated_image_base64 || originalSrc,
        bubbles: bubbles || [],
        overlayContainer: null,
      });

      renderPanel(img);

      console.log(
        `%c[Webtoon Translator]%c Panel ${panelIdx + 1} ready (${processing_time_ms}ms, cached: ${cached})`,
        "color: #10b981; font-weight: bold;",
        ""
      );
    } catch (err) {
      console.warn("[Webtoon Translator] Panel translation skipped:", err.message);
      enqueuedImages.delete(img);
    } finally {
      markImageLoading(img, false);
      updateWidgetStats();
    }
  }

  function renderPanel(img) {
    const entry = panelRegistry.get(img);
    if (!entry) return;

    if (CONFIG.renderMode === "swap") {
      if (entry.overlayContainer) {
        entry.overlayContainer.style.display = "none";
      }
      if (isTranslatedView && entry.translatedSrc) {
        if (img.hasAttribute("srcset")) {
          img.dataset.origSrcset = img.getAttribute("srcset");
          img.removeAttribute("srcset");
        }
        img.src = entry.translatedSrc;
      } else {
        if (img.dataset.origSrcset) {
          img.setAttribute("srcset", img.dataset.origSrcset);
        }
        img.src = entry.originalSrc;
      }
    } else {
      img.src = entry.originalSrc;
      ensureBubbleOverlay(img, entry);
      if (entry.overlayContainer) {
        entry.overlayContainer.style.display = isTranslatedView ? "block" : "none";
      }
    }
  }

  function ensureBubbleOverlay(img, entry) {
    if (entry.overlayContainer) return;
    if (!entry.bubbles || entry.bubbles.length === 0) return;

    const parent = img.parentElement;
    if (window.getComputedStyle(parent).position === "static") {
      parent.style.position = "relative";
    }

    const overlay = document.createElement("div");
    overlay.className = "wt-bubble-overlay-container";

    const rect = img.getBoundingClientRect();
    const naturalW = img.naturalWidth || rect.width;
    const naturalH = img.naturalHeight || rect.height;

    const scaleX = (img.clientWidth || rect.width) / naturalW;
    const scaleY = (img.clientHeight || rect.height) / naturalH;

    entry.bubbles.forEach((b) => {
      const bubbleEl = document.createElement("div");
      bubbleEl.className = "wt-speech-bubble-box";

      bubbleEl.style.left = `${b.box.x * scaleX}px`;
      bubbleEl.style.top = `${b.box.y * scaleY}px`;
      bubbleEl.style.width = `${b.box.w * scaleX}px`;
      bubbleEl.style.height = `${b.box.h * scaleY}px`;

      const textSpan = document.createElement("span");
      textSpan.className = "wt-bubble-text";
      textSpan.textContent = b.english_text;
      bubbleEl.appendChild(textSpan);

      if (CONFIG.showHoverVocab) {
        bubbleEl.appendChild(createLearnerTooltip(b));
      }

      overlay.appendChild(bubbleEl);
    });

    parent.appendChild(overlay);
    entry.overlayContainer = overlay;
  }

  function createLearnerTooltip(bubble) {
    const card = document.createElement("div");
    card.className = "wt-learner-card";

    const hangul = document.createElement("div");
    hangul.className = "wt-learner-hangul";
    hangul.textContent = bubble.korean_text;

    const roman = document.createElement("div");
    roman.className = "wt-learner-roman";
    roman.textContent = bubble.romanization || "";

    const meaning = document.createElement("div");
    meaning.className = "wt-learner-english";
    meaning.textContent = `“${bubble.english_text}”`;

    card.appendChild(hangul);
    if (bubble.romanization) card.appendChild(roman);
    card.appendChild(meaning);

    if (bubble.vocab_breakdown && bubble.vocab_breakdown.length > 0) {
      const vocabList = document.createElement("div");
      vocabList.className = "wt-learner-vocab-list";
      bubble.vocab_breakdown.forEach((v) => {
        const item = document.createElement("span");
        item.className = "wt-learner-vocab-pill";
        item.textContent = `${v.korean} (${v.romanization}): ${v.english}`;
        vocabList.appendChild(item);
      });
      card.appendChild(vocabList);
    }

    return card;
  }

  function markImageLoading(img, isLoading) {
    if (isLoading) {
      img.classList.add("wt-translating-pulse");
    } else {
      img.classList.remove("wt-translating-pulse");
      img.classList.add("wt-translated-fade");
    }
  }

  // -------------------------------------------------------------
  // 5. Floating Control Widget
  // -------------------------------------------------------------
  function injectFloatingWidget() {
    if (document.getElementById("wt-floating-widget")) return;

    const widget = document.createElement("div");
    widget.id = "wt-floating-widget";
    widget.innerHTML = `
      <div class="wt-widget-pill">
        <div class="wt-widget-status">
          <span class="wt-status-dot"></span>
          <span id="wt-widget-count">Webtoon Translator</span>
        </div>
        <div class="wt-widget-actions">
          <button id="wt-toggle-btn" class="wt-widget-btn" title="Toggle English/Korean (Alt+T)">
            <span id="wt-toggle-label">EN</span>
          </button>
          <button id="wt-mode-btn" class="wt-widget-btn" title="Toggle Swap/Overlay Mode">
            <span id="wt-mode-label">Mode: Swap</span>
          </button>
        </div>
      </div>
    `;

    document.body.appendChild(widget);

    document.getElementById("wt-toggle-btn").addEventListener("click", toggleTranslation);
    document.getElementById("wt-mode-btn").addEventListener("click", () => {
      CONFIG.renderMode = CONFIG.renderMode === "swap" ? "overlay" : "swap";
      document.getElementById("wt-mode-label").textContent = `Mode: ${
        CONFIG.renderMode === "swap" ? "Swap" : "Overlay"
      }`;
      panelRegistry.forEach((_, img) => renderPanel(img));
    });
  }

  function toggleTranslation() {
    isTranslatedView = !isTranslatedView;
    const label = document.getElementById("wt-toggle-label");
    if (label) {
      label.textContent = isTranslatedView ? "EN" : "KR";
    }
    panelRegistry.forEach((_, img) => renderPanel(img));
  }

  function updateWidgetStats() {
    const countEl = document.getElementById("wt-widget-count");
    if (!countEl) return;
    const total = findComicImages().length;
    const translated = panelRegistry.size;
    const text = `Translated: ${translated}/${total} panels`;
    if (countEl.textContent !== text) {
      countEl.textContent = text;
    }
  }

  window.addEventListener("keydown", (e) => {
    if (e.altKey && (e.key === "t" || e.key === "T")) {
      e.preventDefault();
      toggleTranslation();
    }
  });

  // -------------------------------------------------------------
  // 6. Initialization
  // -------------------------------------------------------------
  function sendMessageAsync(msg) {
    return new Promise((resolve) => {
      try {
        if (typeof chrome === "undefined" || !chrome.runtime || !chrome.runtime.sendMessage) {
          resolve({ success: false, error: "Chrome runtime unavailable" });
          return;
        }
        chrome.runtime.sendMessage(msg, (response) => {
          if (chrome.runtime.lastError) {
            resolve({ success: false, error: chrome.runtime.lastError.message });
          } else {
            resolve(response || { success: false, error: "No response" });
          }
        });
      } catch (err) {
        resolve({ success: false, error: err.message });
      }
    });
  }

  function init() {
    injectFloatingWidget();
    scanAndObservePanels();

    // Safely observe new images inserted by infinite scroll (ignoring extension UI mutations)
    try {
      const mutObs = new MutationObserver((mutations) => {
        let hasNewElements = false;
        for (const m of mutations) {
          if (m.target && m.target.id && m.target.id.startsWith("wt-")) continue;
          if (m.target && m.target.closest && m.target.closest("#webtoon-translator-widget")) continue;
          if (m.addedNodes.length > 0) {
            hasNewElements = true;
            break;
          }
        }
        if (hasNewElements) {
          debouncedScan();
        }
      });
      mutObs.observe(document.body, { childList: true, subtree: true });
    } catch (_) {}

    // Periodic gentle poll
    setInterval(debouncedScan, 3000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

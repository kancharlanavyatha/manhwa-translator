/**
 * Webtoon Translator - Background Service Worker (Manifest V3)
 *
 * Responsibilities:
 * - Coordinates translation requests with the local/remote backend server.
 * - Bypasses CORS restrictions when fetching raw comic images from CDN servers.
 * - Manages extension cache and settings storage.
 */

const DEFAULT_SETTINGS = {
  serverUrl: "http://127.0.0.1:8000",
  enabled: true,
  targetLang: "en",
  renderMode: "swap", // 'swap' (in-place image swap) or 'overlay' (interactive HTML bubbles)
  showHoverVocab: true,
  prefetchAhead: 3,
  fontScale: 100,
};

/**
 * Initialize extension settings on installation.
 */
chrome.runtime.onInstalled.addListener(async () => {
  const current = await chrome.storage.local.get("settings");
  if (!current.settings) {
    await chrome.storage.local.set({ settings: DEFAULT_SETTINGS });
    console.log("[ServiceWorker] Initialized default settings:", DEFAULT_SETTINGS);
  }

  // Register declarativeNetRequest rules to bypass Naver CDN hotlink protection
  try {
    if (chrome.declarativeNetRequest && chrome.declarativeNetRequest.updateDynamicRules) {
      await chrome.declarativeNetRequest.updateDynamicRules({
        removeRuleIds: [1],
        addRules: [
          {
            id: 1,
            priority: 1,
            action: {
              type: "modifyHeaders",
              requestHeaders: [
                {
                  header: "Referer",
                  operation: "set",
                  value: "https://comic.naver.com/",
                },
              ],
            },
            condition: {
              urlFilter: "*://*.pstatic.net/*",
              resourceTypes: ["xmlhttprequest", "image", "other"],
            },
          },
        ],
      });
      console.log("[ServiceWorker] Registered declarativeNetRequest rule for Naver CDN.");
    }
  } catch (err) {
    console.warn("[ServiceWorker] Failed to register declarativeNetRequest rule:", err);
  }
});

/**
 * Handle messages from Content Script and Popup.
 */
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  const { type, payload } = request;

  switch (type) {
    case "TRANSLATE_IMAGE":
      handleTranslateImage(payload)
        .then((res) => sendResponse({ success: true, data: res }))
        .catch((err) => sendResponse({ success: false, error: err.message }));
      return true; // Keep message channel open for async response

    case "CHECK_HEALTH":
      handleCheckHealth(payload?.serverUrl)
        .then((res) => sendResponse({ success: true, data: res }))
        .catch((err) => sendResponse({ success: false, error: err.message }));
      return true;

    case "FETCH_IMAGE_B64":
      handleFetchImageAsBase64(payload?.url)
        .then((b64) => sendResponse({ success: true, base64: b64 }))
        .catch((err) => sendResponse({ success: false, error: err.message }));
      return true;

    case "GET_SETTINGS":
      chrome.storage.local.get("settings", (items) => {
        sendResponse({ settings: items.settings || DEFAULT_SETTINGS });
      });
      return true;

    case "UPDATE_SETTINGS":
      chrome.storage.local.set({ settings: payload }, () => {
        sendResponse({ success: true });
      });
      return true;

    default:
      console.warn("[ServiceWorker] Unknown message type:", type);
      sendResponse({ success: false, error: "Unknown message type" });
      return false;
  }
});

/**
 * Translates an individual comic panel by posting to the Python backend.
 */
async function handleTranslateImage(payload) {
  const { imageBase64, imageUrl, chapterId, panelIndex, targetLang, mode, glossary } = payload;
  const store = await chrome.storage.local.get("settings");
  const settings = store.settings || DEFAULT_SETTINGS;
  const serverUrl = settings.serverUrl || "http://127.0.0.1:8000";

  const requestBody = {
    image_base64: imageBase64,
    image_url: imageUrl,
    chapter_id: chapterId || "default_chapter",
    panel_index: panelIndex || 0,
    target_lang: targetLang || settings.targetLang || "en",
    mode: mode || "both",
    glossary: glossary || {},
  };

  const response = await fetch(`${serverUrl}/api/translate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Server returned ${response.status}: ${errorText}`);
  }

  return await response.json();
}

/**
 * Checks connectivity and health of the backend translation server.
 */
async function handleCheckHealth(customUrl) {
  const store = await chrome.storage.local.get("settings");
  const settings = store.settings || DEFAULT_SETTINGS;
  const serverUrl = customUrl || settings.serverUrl || "http://127.0.0.1:8000";

  const startTime = Date.now();
  const response = await fetch(`${serverUrl}/api/health`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    throw new Error(`Health check failed with status ${response.status}`);
  }

  const data = await response.json();
  const latency = Date.now() - startTime;
  return { ...data, latencyMs: latency };
}

/**
 * Fetches an image URL in the background context and converts it to base64 Data URL.
 * Uses ArrayBuffer and btoa (immune to worker environment restrictions).
 */
async function handleFetchImageAsBase64(url) {
  if (!url) {
    throw new Error("Missing image URL");
  }

  const fetchOptions = {
    headers: {
      Accept: "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    },
  };
  if (url.includes("pstatic.net") || url.includes("naver.com")) {
    fetchOptions.referrer = "https://comic.naver.com/";
  } else if (url.includes("kakao.com")) {
    fetchOptions.referrer = "https://webtoon.kakao.com/";
  }

  const response = await fetch(url, fetchOptions);
  if (!response.ok) {
    throw new Error(`Failed to fetch image ${url} (${response.status})`);
  }

  const blob = await response.blob();
  if (typeof FileReader !== "undefined") {
    return await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("FileReader failed to convert image"));
      reader.readAsDataURL(blob);
    });
  }

  const buffer = await blob.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunkSize = 16384;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode.apply(null, chunk);
  }
  const base64 = btoa(binary);
  const mimeType = blob.type || "image/jpeg";
  return `data:${mimeType};base64,${base64}`;
}

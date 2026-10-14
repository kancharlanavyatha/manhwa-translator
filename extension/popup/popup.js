/**
 * Webtoon Translator - Popup Script
 *
 * Coordinates user settings with chrome.storage.local and monitors backend health.
 */

document.addEventListener("DOMContentLoaded", async () => {
  // DOM Elements
  const toggleEnabled = document.getElementById("toggle-enabled");
  const targetLangSelect = document.getElementById("target-lang");
  const segmentBtns = document.querySelectorAll(".segment-btn");
  const toggleHoverVocab = document.getElementById("toggle-hover-vocab");
  const fontScaleSlider = document.getElementById("font-scale");
  const fontScaleValue = document.getElementById("font-scale-value");
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const statusLatency = document.getElementById("status-latency");
  const btnOpenOptions = document.getElementById("btn-open-options");
  const btnClearCache = document.getElementById("btn-clear-cache");

  // Load current settings
  const { settings } = await chrome.storage.local.get("settings");
  const current = settings || {
    enabled: true,
    targetLang: "en",
    renderMode: "swap",
    showHoverVocab: true,
    fontScale: 100,
    serverUrl: "http://127.0.0.1:8000",
  };

  // Populate UI
  toggleEnabled.checked = current.enabled !== false;
  targetLangSelect.value = current.targetLang || "en";
  toggleHoverVocab.checked = current.showHoverVocab !== false;
  fontScaleSlider.value = current.fontScale || 100;
  fontScaleValue.textContent = `${fontScaleSlider.value}%`;

  segmentBtns.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === (current.renderMode || "swap"));
  });

  // Check backend server health
  checkServerHealth(current.serverUrl);

  // Event Listeners for UI state saves
  toggleEnabled.addEventListener("change", () => {
    saveSetting("enabled", toggleEnabled.checked);
  });

  targetLangSelect.addEventListener("change", () => {
    saveSetting("targetLang", targetLangSelect.value);
  });

  segmentBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      segmentBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      saveSetting("renderMode", btn.dataset.mode);
    });
  });

  toggleHoverVocab.addEventListener("change", () => {
    saveSetting("showHoverVocab", toggleHoverVocab.checked);
  });

  fontScaleSlider.addEventListener("input", () => {
    fontScaleValue.textContent = `${fontScaleSlider.value}%`;
  });

  fontScaleSlider.addEventListener("change", () => {
    saveSetting("fontScale", parseInt(fontScaleSlider.value, 10));
  });

  btnOpenOptions.addEventListener("click", () => {
    chrome.runtime.openOptionsPage();
  });

  btnClearCache.addEventListener("click", async () => {
    btnClearCache.textContent = "Clearing...";
    try {
      const serverUrl = current.serverUrl || "http://127.0.0.1:8000";
      await fetch(`${serverUrl}/api/cache/clear`, { method: "POST" });
      btnClearCache.textContent = "Cleared!";
      setTimeout(() => (btnClearCache.textContent = "🗑️ Clear Cache"), 1500);
    } catch (err) {
      btnClearCache.textContent = "Error";
      setTimeout(() => (btnClearCache.textContent = "🗑️ Clear Cache"), 1500);
    }
  });

  async function saveSetting(key, value) {
    const store = await chrome.storage.local.get("settings");
    const updated = { ...(store.settings || {}), [key]: value };
    await chrome.storage.local.set({ settings: updated });
  }

  async function checkServerHealth(serverUrl) {
    chrome.runtime.sendMessage(
      { type: "CHECK_HEALTH", payload: { serverUrl } },
      (response) => {
        if (response && response.success) {
          statusDot.className = "dot connected";
          statusText.textContent = `Connected (${response.data.translation_provider})`;
          statusLatency.textContent = `${response.data.latencyMs} ms`;
        } else {
          statusDot.className = "dot error";
          statusText.textContent = "Server Offline (127.0.0.1:8000)";
          statusLatency.textContent = "offline";
        }
      }
    );
  }
});

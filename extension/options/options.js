/**
 * Webtoon Translator - Options & Glossary Controller
 */

document.addEventListener("DOMContentLoaded", async () => {
  const settingsForm = document.getElementById("settings-form");
  const serverUrlInput = document.getElementById("server-url");
  const prefetchAheadInput = document.getElementById("prefetch-ahead");
  const saveStatus = document.getElementById("save-status");

  const addTermForm = document.getElementById("add-term-form");
  const termSeriesInput = document.getElementById("term-series");
  const termKoreanInput = document.getElementById("term-korean");
  const termEnglishInput = document.getElementById("term-english");
  const termCategoryInput = document.getElementById("term-category");
  const glossaryTbody = document.getElementById("glossary-tbody");
  const btnSeedPresets = document.getElementById("btn-seed-presets");

  // Load saved settings
  const { settings } = await chrome.storage.local.get("settings");
  if (settings) {
    if (settings.serverUrl) serverUrlInput.value = settings.serverUrl;
    if (settings.prefetchAhead) prefetchAheadInput.value = settings.prefetchAhead;
  }

  // Save Settings Form Handler
  settingsForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const updated = {
      ...(settings || {}),
      serverUrl: serverUrlInput.value.trim(),
      prefetchAhead: parseInt(prefetchAheadInput.value, 10),
    };
    await chrome.storage.local.set({ settings: updated });
    saveStatus.textContent = "Saved successfully!";
    setTimeout(() => (saveStatus.textContent = ""), 2000);
    loadGlossary();
  });

  // Load and render glossary terms
  async function loadGlossary() {
    glossaryTbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:#64748b; padding:16px;">Loading terms...</td></tr>`;
    try {
      const serverUrl = serverUrlInput.value.trim() || "http://127.0.0.1:8000";
      const resp = await fetch(`${serverUrl}/api/glossary?series_id=default`);
      if (!resp.ok) throw new Error("Failed to fetch glossary");
      const items = await resp.json();

      if (!items || items.length === 0) {
        glossaryTbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:#64748b; padding:16px;">No terms found. Add your first term above.</td></tr>`;
        return;
      }

      glossaryTbody.innerHTML = "";
      items.forEach((item) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><code>${escapeHtml(item.series_id)}</code></td>
          <td style="font-weight:600; color:#38bdf8;">${escapeHtml(item.source_term)}</td>
          <td style="font-weight:600;">${escapeHtml(item.target_term)}</td>
          <td><span class="category-tag">${escapeHtml(item.category || "term")}</span></td>
          <td><button class="btn-delete" data-series="${escapeHtml(item.series_id)}" data-term="${escapeHtml(item.source_term)}">Delete</button></td>
        `;
        glossaryTbody.appendChild(tr);
      });

      // Bind delete handlers
      document.querySelectorAll(".btn-delete").forEach((btn) => {
        btn.addEventListener("click", () => {
          deleteTerm(btn.dataset.series, btn.dataset.term);
        });
      });
    } catch (err) {
      glossaryTbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:#ef4444; padding:16px;">Error connecting to backend (${err.message})</td></tr>`;
    }
  }

  // Add term form handler
  addTermForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const serverUrl = serverUrlInput.value.trim() || "http://127.0.0.1:8000";
    const payload = {
      series_id: termSeriesInput.value.trim() || "default",
      source_term: termKoreanInput.value.trim(),
      target_term: termEnglishInput.value.trim(),
      category: termCategoryInput.value,
      notes: "",
    };

    try {
      const resp = await fetch(`${serverUrl}/api/glossary`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) throw new Error("Failed to add term");

      termKoreanInput.value = "";
      termEnglishInput.value = "";
      loadGlossary();
    } catch (err) {
      alert(`Error saving term: ${err.message}`);
    }
  });

  async function deleteTerm(seriesId, sourceTerm) {
    if (!confirm(`Delete translation for '${sourceTerm}'?`)) return;
    const serverUrl = serverUrlInput.value.trim() || "http://127.0.0.1:8000";
    try {
      const resp = await fetch(
        `${serverUrl}/api/glossary?series_id=${encodeURIComponent(seriesId)}&source_term=${encodeURIComponent(sourceTerm)}`,
        { method: "DELETE" }
      );
      if (!resp.ok) throw new Error("Failed to delete term");
      loadGlossary();
    } catch (err) {
      alert(`Error deleting term: ${err.message}`);
    }
  }

  // Preset Seeder
  btnSeedPresets.addEventListener("click", async () => {
    const presets = [
      { series_id: "default", source_term: "형", target_term: "Hyung", category: "honorific" },
      { series_id: "default", source_term: "오빠", target_term: "Oppa", category: "honorific" },
      { series_id: "default", source_term: "선배", target_term: "Sunbae", category: "honorific" },
      { series_id: "default", source_term: "누나", target_term: "Noona", category: "honorific" },
      { series_id: "default", source_term: "성진우", target_term: "Sung Jin-woo", category: "character" },
    ];

    const serverUrl = serverUrlInput.value.trim() || "http://127.0.0.1:8000";
    for (const p of presets) {
      await fetch(`${serverUrl}/api/glossary`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(p),
      });
    }
    loadGlossary();
  });

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  loadGlossary();
});

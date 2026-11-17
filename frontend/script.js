(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const MIN_WORDS = 30, MAX_MB = 10, EXT = [".txt", ".pdf", ".docx"];

  const el = {
    text: $("inputText"), words: $("inWords"), chars: $("inChars"), file: $("fileInput"),
    drop: $("dropzone"), fileInfo: $("fileInfo"), gen: $("generateBtn"), err: $("errorBox"),
    custToggle: $("customToggle"), custBox: $("customBox"), custType: $("customType"), custVal: $("customValue"),
    bullets: $("bullets"), mode: $("mode"), placeholder: $("placeholder"), loading: $("loading"),
    result: $("result"), summary: $("summaryText"), warn: $("warnBox"), method: $("methodBadge"),
    statOrig: $("statOrig"), statSum: $("statSum"), statRed: $("statRed"), toast: $("toast"),
  };
  let currentSummary = "";

  const countWords = (t) => (t.trim().match(/\S+/g) || []).length;

  function toast(msg) {
    el.toast.textContent = msg;
    el.toast.classList.add("show");
    setTimeout(() => el.toast.classList.remove("show"), 2200);
  }
  function showError(msg) { el.err.textContent = msg; el.err.hidden = !msg; }

  function updateCounts() {
    el.words.textContent = countWords(el.text.value).toLocaleString();
    el.chars.textContent = el.text.value.length.toLocaleString();
  }
  el.text.addEventListener("input", () => { updateCounts(); showError(""); });

  // ---- tabs
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => { t.classList.toggle("active", t === tab); t.setAttribute("aria-selected", t === tab); });
    $("paneText").hidden = tab.dataset.tab !== "text";
    $("paneFile").hidden = tab.dataset.tab !== "file";
  }));

  // ---- file upload (extract text on the server, then show it in the editor)
  function checkFile(f) {
    const ext = f.name.slice(f.name.lastIndexOf(".")).toLowerCase();
    if (!EXT.includes(ext)) return "Unsupported file type. Please upload a .txt, .pdf or .docx file.";
    if (f.size === 0) return "The selected file is empty.";
    if (f.size > MAX_MB * 1024 * 1024) return `File is too large. Maximum size is ${MAX_MB} MB.`;
    return "";
  }
  async function handleFile(f) {
    showError("");
    const problem = checkFile(f);
    if (problem) { showError(problem); return; }
    const fd = new FormData();
    fd.append("file", f);
    el.fileInfo.hidden = false;
    el.fileInfo.textContent = `Reading ${f.name}…`;
    try {
      const res = await fetch("/api/extract", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Could not read the file.");
      el.text.value = data.text;
      updateCounts();
      el.fileInfo.textContent = `✓ ${data.filename} loaded (${data.word_count.toLocaleString()} words). You can edit the text in the “Paste text” tab.`;
    } catch (e) {
      el.fileInfo.hidden = true;
      showError(e.message || "Upload failed.");
    } finally { el.file.value = ""; }
  }
  el.file.addEventListener("change", () => el.file.files[0] && handleFile(el.file.files[0]));
  ["dragenter", "dragover"].forEach((ev) => el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.remove("drag"); }));
  el.drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && handleFile(e.dataTransfer.files[0]));

  // ---- options
  el.custToggle.addEventListener("change", () => {
    el.custBox.hidden = !el.custToggle.checked;
    document.querySelectorAll("input[name=length]").forEach((r) => (r.disabled = el.custToggle.checked));
  });
  el.custType.addEventListener("change", () => {
    el.custVal.value = el.custType.value === "words" ? 100 : 5;
    el.custVal.max = el.custType.value === "words" ? 2000 : 50;
  });

  $("clearBtn").addEventListener("click", () => {
    el.text.value = ""; updateCounts(); showError(""); el.fileInfo.hidden = true;
    el.result.hidden = true; el.placeholder.hidden = false; el.method.hidden = true;
  });
  $("sampleBtn").addEventListener("click", () => {
    el.text.value = "Artificial intelligence is changing how people work with information. Every day, students, researchers and professionals face long reports, research papers and news articles that take hours to read. Text summarization tools help by reducing a document to its most important points. There are two main approaches. Extractive summarization selects the most important sentences from the original text and keeps them unchanged. Abstractive summarization writes new sentences that express the same meaning, much like a human would. Modern language models can do both, but they must be guided carefully so that they do not invent facts. A good summary keeps names, dates and numbers accurate, avoids repetition and remains easy to read. In 2024, surveys of university students reported that saving reading time was the most common reason for using AI study tools. Long documents are usually split into smaller chunks, summarized separately and then combined, so that no section is ignored.";
    updateCounts(); $("tabText").click(); showError("");
  });

  // ---- summarize
  function setBusy(busy) {
    el.gen.disabled = busy;
    el.gen.classList.toggle("busy", busy);
    el.gen.querySelector(".btn-label").textContent = busy ? "Summarizing…" : "Generate Summary";
    el.loading.hidden = !busy;
    if (busy) { el.placeholder.hidden = true; el.result.hidden = true; }
  }

  function renderSummary(text, asBullets) {
    el.summary.textContent = "";
    const lines = text.split("\n").filter((l) => l.trim());
    if (asBullets && lines.length) {
      const ul = document.createElement("ul");
      lines.forEach((l) => { const li = document.createElement("li"); li.textContent = l.replace(/^\s*[•\-*]\s*/, ""); ul.appendChild(li); });
      el.summary.appendChild(ul);
    } else { el.summary.textContent = text; }
  }

  el.gen.addEventListener("click", async () => {
    showError("");
    const text = el.text.value.trim();
    if (!text) return showError("Please paste some text or upload a document first.");
    if (countWords(text) < MIN_WORDS) return showError(`Text is too short. Please enter at least ${MIN_WORDS} words.`);

    const body = {
      text, mode: el.mode.value, bullets: el.bullets.checked,
      length: (document.querySelector("input[name=length]:checked") || {}).value || "medium",
    };
    if (el.custToggle.checked) {
      const v = parseInt(el.custVal.value, 10);
      const max = el.custType.value === "words" ? 2000 : 50, min = el.custType.value === "words" ? 10 : 1;
      if (!Number.isInteger(v) || v < min || v > max) return showError(`Custom ${el.custType.value} must be between ${min} and ${max}.`);
      body.custom_type = el.custType.value; body.custom_value = v;
    }

    setBusy(true);
    try {
      const res = await fetch("/api/summarize", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      let data = {};
      try { data = await res.json(); } catch (_) { /* non-JSON error page */ }
      if (!res.ok) throw new Error(data.error || `Server error (${res.status}).`);

      currentSummary = data.summary;
      renderSummary(data.summary, body.bullets);
      el.statOrig.textContent = data.original_words.toLocaleString();
      el.statSum.textContent = data.summary_words.toLocaleString();
      el.statRed.textContent = Math.max(0, data.reduction_percent) + "%";
      el.method.hidden = false; el.method.textContent = data.method;
      el.warn.hidden = !(data.warnings && data.warnings.length);
      el.warn.textContent = (data.warnings || []).join(" ");
      el.result.hidden = false;
      if (window.innerWidth <= 900) el.result.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) {
      el.placeholder.hidden = false;
      showError(e instanceof TypeError ? "Cannot reach the server. Is the backend running?" : e.message);
    } finally { setBusy(false); }
  });

  // ---- copy / download
  $("copyBtn").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(currentSummary); toast("Summary copied"); }
    catch (_) {
      const ta = document.createElement("textarea"); ta.value = currentSummary; document.body.appendChild(ta);
      ta.select(); document.execCommand("copy"); ta.remove(); toast("Summary copied");
    }
  });
  async function download(format) {
    if (!currentSummary) return;
    try {
      const res = await fetch("/api/download", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ summary: currentSummary, format }) });
      if (!res.ok) throw new Error((await res.json()).error || "Download failed.");
      const url = URL.createObjectURL(await res.blob());
      const a = document.createElement("a"); a.href = url; a.download = `summary.${format}`; a.click();
      URL.revokeObjectURL(url);
    } catch (e) { showError(e.message); }
  }
  $("dlTxt").addEventListener("click", () => download("txt"));
  $("dlPdf").addEventListener("click", () => download("pdf"));

  // ---- AI status badge
  fetch("/api/health").then((r) => r.json()).then((h) => {
    const b = $("aiStatus");
    b.textContent = h.ai_configured ? `AI: ${h.provider}` : "Offline mode";
    b.classList.toggle("ok", h.ai_configured);
    b.title = h.ai_configured ? "AI engine connected" : "No API key set — extractive summaries only";
  }).catch(() => { $("aiStatus").textContent = "Server offline"; });

  updateCounts();
})();

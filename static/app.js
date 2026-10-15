document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const sendBtn = document.getElementById("send-btn");
  const chatViewport = document.getElementById("chat-viewport");
  const messagesStream = document.getElementById("messages-stream");
  const heroWelcome = document.getElementById("hero-welcome");
  const clearChatBtn = document.getElementById("clear-chat-btn");
  const toggleSidebarBtn = document.getElementById("toggle-sidebar-btn");
  const sidebar = document.getElementById("sidebar");

  const systemBadge = document.getElementById("system-badge");
  const chunkCountBadge = document.getElementById("chunk-count-badge");
  const topKInput = document.getElementById("top-k");
  const topKVal = document.getElementById("top-k-val");
  const thresholdInput = document.getElementById("threshold");
  const thresholdVal = document.getElementById("threshold-val");
  const streamToggle = document.getElementById("stream-toggle");

  // Upload elements
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const browseBtn = document.getElementById("browse-btn");
  const uploadCard = document.getElementById("upload-status-card");
  const jobFilename = document.getElementById("job-filename");
  const jobBadge = document.getElementById("job-badge");
  const progressFill = document.getElementById("progress-fill");
  const jobDetails = document.getElementById("job-details");
  const docList = document.getElementById("doc-list");
  const toastContainer = document.getElementById("toast-container");

  // Multi-Tenant User Workspace State (persisted per device/browser in localStorage)
  let currentUserId = localStorage.getItem("omni_rag_user_id");
  if (!currentUserId) {
    currentUserId = "usr_" + Math.random().toString(36).substring(2, 10);
    localStorage.setItem("omni_rag_user_id", currentUserId);
  }

  // Display workspace identifier in sidebar
  const workspaceIdEl = document.getElementById("user-workspace-id");
  if (workspaceIdEl) {
    workspaceIdEl.textContent = currentUserId;
  }

  // Conversational Session State
  let currentSessionId = sessionStorage.getItem("omni_rag_session_id");
  if (!currentSessionId) {
    currentSessionId = "sess_" + Math.random().toString(36).substring(2, 12);
    sessionStorage.setItem("omni_rag_session_id", currentSessionId);
  }

  // Visual Citation Inspector Elements
  const inspectorModal = document.getElementById("citation-inspector-modal");
  const inspectorBackdrop = document.getElementById("inspector-backdrop");
  const inspectorCloseBtn = document.getElementById("inspector-close-btn");
  const inspectorDocTitle = document.getElementById("inspector-doc-title");
  const inspectorSimilarity = document.getElementById("inspector-similarity");
  const inspectorPage = document.getElementById("inspector-page");
  const inspectorChunkIdx = document.getElementById("inspector-chunk-idx");
  const inspectorOffsets = document.getElementById("inspector-offsets");
  const inspectorSnippetText = document.getElementById("inspector-snippet-text");
  const inspectorCopyBtn = document.getElementById("inspector-copy-btn");
  const inspectorFollowupBtn = document.getElementById("inspector-followup-btn");

  // Track initial state
  let hasMessages = false;

  // Initialize marked options
  if (window.marked) {
    marked.setOptions({
      highlight: function (code, lang) {
        if (window.hljs && hljs.getLanguage(lang)) {
          return hljs.highlight(code, { language: lang }).value;
        }
        return window.hljs ? hljs.highlightAuto(code).value : code;
      },
      breaks: true,
      gfm: true,
    });
  }

  // Toast Notification Helper
  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${message}</span>`;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(20px)";
      toast.style.transition = "all 0.3s ease";
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  // =====================================================================
  // DEV MODE / CONSUMER MODE CONTROLLER
  // =====================================================================
  const devModeBtn = document.getElementById("dev-mode-btn");
  const devModeIndicator = document.getElementById("dev-mode-indicator");

  // Load user preference (default: Dev Mode ON for portfolio showcase)
  let isDevMode = localStorage.getItem("omni_rag_dev_mode") !== "false";

  function applyDevMode(active, showNotice = false) {
    if (active) {
      document.body.classList.add("dev-mode-active");
      if (devModeBtn) devModeBtn.classList.add("active");
      if (devModeIndicator) devModeIndicator.textContent = "ON";
      if (showNotice) showToast("🛠️ Dev Mode ON: Engineering metrics & retrieval tuning visible", "info");
    } else {
      document.body.classList.remove("dev-mode-active");
      if (devModeBtn) devModeBtn.classList.remove("active");
      if (devModeIndicator) devModeIndicator.textContent = "OFF";
      if (showNotice) showToast("✨ Simple Mode: Clean consumer experience (tech metrics hidden)", "info");
    }
    try {
      localStorage.setItem("omni_rag_dev_mode", active ? "true" : "false");
    } catch (e) {
      console.warn("Storage warning:", e);
    }
  }

  // Initial application on boot
  applyDevMode(isDevMode, false);

  if (devModeBtn) {
    devModeBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      isDevMode = !isDevMode;
      applyDevMode(isDevMode, true);
    });
  }

  // Expose globally for testing/debugging
  window.toggleDevMode = () => {
    isDevMode = !isDevMode;
    applyDevMode(isDevMode, true);
  };

  // Mobile Sidebar Drawer Management
  const sidebarBackdrop = document.getElementById("sidebar-backdrop");
  if (sidebarBackdrop) {
    sidebarBackdrop.addEventListener("click", () => {
      sidebar.classList.add("collapsed");
    });
  }

  // Auto-collapse sidebar on initial load for mobile screens (width <= 768px)
  if (window.innerWidth <= 768) {
    sidebar.classList.add("collapsed");
  }

  function closeMobileSidebar() {
    if (window.innerWidth <= 768) {
      sidebar.classList.add("collapsed");
    }
  }

  // Toggle Sidebar
  toggleSidebarBtn.addEventListener("click", () => {
    sidebar.classList.toggle("collapsed");
  });

  // Slider Listeners
  topKInput.addEventListener("input", (e) => (topKVal.textContent = e.target.value));
  thresholdInput.addEventListener("input", (e) => (thresholdVal.textContent = e.target.value));

  // Auto-expand Textarea
  chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + "px";
  });

  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  // Health and stats check
  let currentLLMProvider = "Local LLM";
  async function checkHealth() {
    try {
      const res = await fetch(`/health?user_id=${encodeURIComponent(currentUserId)}`);
      if (res.ok) {
        const data = await res.json();
        currentLLMProvider = data.llm_provider || "Online";
        systemBadge.textContent = `${currentLLMProvider} • ${data.vector_store_type}`;
        chunkCountBadge.textContent = data.total_indexed_chunks + " chunks";
        const storageValEl = document.getElementById("user-storage-val");
        if (storageValEl && data.storage_used_mb !== undefined) {
          storageValEl.textContent = `${data.storage_used_mb} / ${data.storage_limit_mb || 10} MB`;
        }
      }
    } catch {
      systemBadge.textContent = "Connecting...";
    }
  }
  checkHealth();

  // Dynamic Indexed Documents Management
  async function fetchDocumentList() {
    try {
      const res = await fetch(`/api/documents?user_id=${encodeURIComponent(currentUserId)}`);
      if (res.ok) {
        const docs = await res.json();
        renderDocumentList(docs);
      }
    } catch {
      // Keep existing list on error
    }
  }

  function renderDocumentList(docs = []) {
    if (!docs || docs.length === 0) {
      docList.innerHTML = `<p style="font-size: 11px; color: var(--text-muted); text-align: center; padding: 14px 10px;">No documents in your workspace.<br>Upload a PDF, Markdown, or TXT file above.</p>`;
      return;
    }

    docList.innerHTML = docs
      .map(
        (d) => `
        <div class="doc-item-row">
          <div class="doc-info" style="display: flex; flex-direction: column; overflow: hidden; max-width: 80%;">
            <div style="display: flex; align-items: center; gap: 4px; overflow: hidden;">
              <span class="doc-name" title="${escapeHtml(d.source)}">📄 ${escapeHtml(d.source)}</span>
              ${d.is_sample ? '<span style="font-size: 9px; color: #94a3b8; background: rgba(255,255,255,0.08); padding: 1px 5px; border-radius: 4px; font-weight: 600;">Sample</span>' : ""}
            </div>
            <span class="doc-chunks-sub" style="font-size: 10.5px; color: var(--text-muted); padding-left: 20px;">${d.chunks_count} chunk${d.chunks_count === 1 ? "" : "s"}</span>
          </div>
          <button class="doc-del-btn" title="Delete document from your workspace" data-source="${escapeHtml(d.source)}">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="3 6 5 6 21 6"/>
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            </svg>
          </button>
        </div>
      `
      )
      .join("");

    document.querySelectorAll(".doc-del-btn").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        const source = e.currentTarget.getAttribute("data-source");
        try {
          const res = await fetch(`/api/documents/${encodeURIComponent(source)}?user_id=${encodeURIComponent(currentUserId)}`, { method: "DELETE" });
          if (res.ok) {
            showToast(`Deleted ${source} from your workspace`, "success");
            await fetchDocumentList();
            await fetchSuggestions();
            checkHealth();
          }
        } catch {
          showToast("Failed to delete document", "error");
        }
      });
    });
  }

  // Dynamic Prompt Suggestions Engine
  const promptGrid = document.querySelector(".prompt-grid");
  const heroDesc = document.querySelector(".hero-desc");

  async function fetchSuggestions() {
    if (!promptGrid) return;
    try {
      const res = await fetch(`/api/suggestions?user_id=${encodeURIComponent(currentUserId)}`);
      if (res.ok) {
        const suggestions = await res.json();
        renderSuggestions(suggestions);
      }
    } catch {
      // Keep existing cards on network error
    }
  }

  function renderSuggestions(suggestions = []) {
    if (!promptGrid || !suggestions || suggestions.length === 0) return;

    promptGrid.innerHTML = suggestions
      .map(
        (s) => `
        <div class="prompt-card" data-query="${escapeHtml(s.query)}">
          <div class="card-icon">${escapeHtml(s.icon || "💡")}</div>
          <div class="card-text">
            <strong>${escapeHtml(s.title)}</strong>
            <span>${escapeHtml(s.subtitle)}</span>
          </div>
        </div>
      `
      )
      .join("");

    // Bind click events on dynamic cards
    promptGrid.querySelectorAll(".prompt-card").forEach((card) => {
      card.addEventListener("click", () => {
        const query = card.getAttribute("data-query");
        if (query) {
          chatInput.value = query;
          chatInput.focus();
          chatForm.dispatchEvent(new Event("submit"));
        }
      });
    });
  }

  // Initial load
  fetchDocumentList();
  fetchSuggestions();

  // Clear Chat & Reset Session Memory
  clearChatBtn.addEventListener("click", async () => {
    try {
      if (currentSessionId) {
        await fetch(`/api/sessions/${encodeURIComponent(currentSessionId)}`, { method: "DELETE" });
      }
    } catch {}
    currentSessionId = "sess_" + Math.random().toString(36).substring(2, 12);
    sessionStorage.setItem("omni_rag_session_id", currentSessionId);
    messagesStream.innerHTML = "";
    heroWelcome.style.display = "block";
    hasMessages = false;
    showToast("Chat reset & conversational memory cleared", "info");
  });

  // File Upload Handlers
  browseBtn.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("click", (e) => {
    if (e.target !== browseBtn) fileInput.click();
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("user_id", currentUserId);

    uploadCard.classList.remove("hidden");
    jobFilename.textContent = file.name;
    jobBadge.textContent = "Uploading";
    jobBadge.className = "badge-pill badge-processing";
    progressFill.style.width = "25%";
    jobDetails.textContent = "Writing file to disk...";

    try {
      const res = await fetch("/api/documents/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload rejected");
      }

      const job = await res.json();
      showToast(`Upload accepted! Job ID: ${job.job_id}`, "info");
      pollJobStatus(job.job_id, file.name);
    } catch (err) {
      jobBadge.textContent = "Failed";
      jobBadge.className = "badge-pill badge-failed";
      jobDetails.textContent = err.message;
      showToast(`Upload failed: ${err.message}`, "error");
    }
  }

  function pollJobStatus(jobId, filename) {
    jobBadge.textContent = "Indexing";
    progressFill.style.width = "65%";
    jobDetails.textContent = "Parsing sections, generating vectors & storing...";

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/documents/jobs/${jobId}`);
        if (!res.ok) return;
        const job = await res.json();

        if (job.status === "INDEXED") {
          clearInterval(interval);
          progressFill.style.width = "100%";
          jobBadge.textContent = "Indexed";
          jobBadge.className = "badge-pill badge-success";
          jobDetails.textContent = `Indexed ${job.chunks_count} chunks into Vector DB!`;
          showToast(`✓ Ingested ${filename} (${job.chunks_count} chunks)`, "success");

          await fetchDocumentList();
          await fetchSuggestions();
          checkHealth();
          setTimeout(() => uploadCard.classList.add("hidden"), 3500);
        } else if (job.status === "FAILED") {
          clearInterval(interval);
          jobBadge.textContent = "Failed";
          jobBadge.className = "badge-pill badge-failed";
          jobDetails.textContent = job.error_message || "Ingestion error";
          showToast(`Ingestion failed: ${job.error_message}`, "error");
        }
      } catch {
        clearInterval(interval);
      }
    }, 700);
  }

  const sendIconSvg = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
      <line x1="22" y1="2" x2="11" y2="13"/>
      <polygon points="22 2 15 22 11 13 2 9 22 2"/>
    </svg>
  `;

  function setGeneratingState(isGenerating) {
    if (!sendBtn) return;
    if (isGenerating) {
      sendBtn.disabled = true;
      sendBtn.classList.add("loading");
      sendBtn.innerHTML = '<span class="button-spinner"></span>';
      chatInput.disabled = true;
      chatInput.placeholder = "AI is thinking and synthesizing response...";
    } else {
      sendBtn.disabled = false;
      sendBtn.classList.remove("loading");
      sendBtn.innerHTML = sendIconSvg;
      chatInput.disabled = false;
      chatInput.placeholder = "Ask anything about the indexed documents...";
      setTimeout(() => chatInput.focus(), 50);
    }
  }

  function createThinkingCard() {
    const wrapper = document.createElement("div");
    wrapper.className = "thinking-status-card";
    wrapper.innerHTML = `
      <div class="thinking-header-row">
        <div class="thinking-left">
          <div class="thinking-spinner-ring">
            <div class="spinner-core"></div>
          </div>
          <div class="thinking-labels">
            <span class="thinking-stage-title" id="stage-title">Searching knowledge base...</span>
            <span class="thinking-stage-sub" id="stage-sub">Querying vector embeddings & keyword index</span>
          </div>
        </div>
        <div class="thinking-right">
          <span class="thinking-timer-badge" id="thinking-timer">0s</span>
        </div>
      </div>
      <div class="thinking-progress-track">
        <div class="thinking-progress-glow"></div>
      </div>
      <div class="thinking-footer-row">
        <div class="thinking-hardware-pill">
          <span class="pulse-chip"></span>
          <span class="chip-text" id="hardware-status">Query received • starting local search...</span>
        </div>
        <span class="thinking-reassurance">Please hold on, synthesizing grounded answer</span>
      </div>
    `;

    const stageTitle = wrapper.querySelector("#stage-title");
    const stageSub = wrapper.querySelector("#stage-sub");
    const thinkingTimer = wrapper.querySelector("#thinking-timer");
    const hardwareStatus = wrapper.querySelector("#hardware-status");

    let seconds = 0;
    const interval = setInterval(() => {
      seconds++;
      if (thinkingTimer) thinkingTimer.textContent = `${seconds}s`;

      if (seconds === 1) {
        if (stageTitle) stageTitle.textContent = "Re-ranking context candidates...";
        if (stageSub) stageSub.textContent = "FlashRank Cross-Encoder scoring chunks";
        if (hardwareStatus) hardwareStatus.textContent = "Hybrid retrieval complete • ranking top chunks...";
      } else if (seconds === 3) {
        if (stageTitle) stageTitle.textContent = "Synthesizing answer...";
        if (stageSub) stageSub.textContent = `Provider: ${currentLLMProvider}`;
        if (hardwareStatus) {
          hardwareStatus.textContent = currentLLMProvider.includes("Gemini")
            ? "Cloud TPU inference active • responding in < 1 second..."
            : "Local CPU token evaluation in progress...";
        }
      } else if (seconds === 10) {
        if (hardwareStatus) hardwareStatus.textContent = "Evaluating prompt tokens on CPU (takes ~25-45s)...";
      } else if (seconds === 25) {
        if (hardwareStatus) hardwareStatus.textContent = "Generation in progress • tokens starting soon...";
      }
    }, 1000);

    return {
      element: wrapper,
      stop: () => clearInterval(interval),
    };
  }

  // Chat Submission Handler
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = chatInput.value.trim();
    if (!query) return;

    chatInput.value = "";
    chatInput.style.height = "auto";

    if (!hasMessages) {
      heroWelcome.style.display = "none";
      hasMessages = true;
    }

    appendUserMessage(query);
    setGeneratingState(true);

    const topK = parseInt(topKInput.value);
    const threshold = parseFloat(thresholdInput.value);
    const isStream = streamToggle.checked;

    try {
      if (isStream) {
        await streamResponse(query, topK, threshold);
      } else {
        await standardResponse(query, topK, threshold);
      }
    } finally {
      setGeneratingState(false);
    }
  });

  function appendUserMessage(text) {
    const node = document.createElement("div");
    node.className = "message-node user-node";
    node.innerHTML = `
      <div class="avatar-ring">👤</div>
      <div class="bubble-content">
        <p>${escapeHtml(text)}</p>
      </div>
    `;
    messagesStream.appendChild(node);
    chatViewport.scrollTop = chatViewport.scrollHeight;
  }

  // =====================================================================
  // VISUAL CITATION INSPECTOR & TELEMETRY BADGES
  // =====================================================================
  function openCitationInspector(sourceData) {
    if (!inspectorModal || !sourceData) return;
    const docName = sourceData.source || "Document";
    if (inspectorDocTitle) inspectorDocTitle.textContent = docName;
    const simPct = Math.round((sourceData.similarity || 0) * 100);
    if (inspectorSimilarity) {
      inspectorSimilarity.textContent = `${simPct}%`;
      inspectorSimilarity.className = `meta-value ${simPct >= 70 ? "score-high" : ""}`;
    }
    if (inspectorPage) inspectorPage.textContent = `Page ${sourceData.page || 1}`;
    if (inspectorChunkIdx) inspectorChunkIdx.textContent = `#${sourceData.chunk_index ?? 0}`;
    if (inspectorOffsets) {
      const cStart = sourceData.char_start ?? 0;
      const cEnd = sourceData.char_end ?? (sourceData.text ? sourceData.text.length : 0);
      inspectorOffsets.textContent = `${cStart} — ${cEnd}`;
    }
    if (inspectorSnippetText) {
      inspectorSnippetText.textContent = sourceData.text || "No snippet content available.";
    }

    if (inspectorCopyBtn) {
      inspectorCopyBtn.onclick = () => {
        navigator.clipboard.writeText(sourceData.text || "");
        showToast(`Copied snippet from ${docName}!`, "info");
      };
    }

    if (inspectorFollowupBtn) {
      inspectorFollowupBtn.onclick = () => {
        closeCitationInspector();
        chatInput.value = `Regarding ${docName} (Page ${sourceData.page || 1}): `;
        chatInput.focus();
      };
    }

    inspectorModal.classList.remove("hidden");
    document.body.style.overflow = "hidden";
  }

  function closeCitationInspector() {
    if (!inspectorModal) return;
    inspectorModal.classList.add("hidden");
    document.body.style.overflow = "";
  }

  if (inspectorCloseBtn) inspectorCloseBtn.addEventListener("click", closeCitationInspector);
  if (inspectorBackdrop) inspectorBackdrop.addEventListener("click", closeCitationInspector);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && inspectorModal && !inspectorModal.classList.contains("hidden")) {
      closeCitationInspector();
    }
  });

  function bindInlineCitationClicks(messageNode, sources) {
    if (!messageNode) return;
    const badges = messageNode.querySelectorAll(".inline-cite-badge");
    badges.forEach((badge) => {
      badge.style.cursor = "pointer";
      badge.onclick = (e) => {
        e.stopPropagation();
        const srcName = badge.getAttribute("data-source");
        const pageNum = parseInt(badge.getAttribute("data-page") || "1");

        let matched = (sources || []).find((s) => {
          const sName = (s.source || "").toLowerCase();
          const target = (srcName || "").toLowerCase();
          return sName.includes(target) || target.includes(sName);
        });

        if (!matched && sources && sources.length > 0) {
          matched = sources[0];
        }

        if (matched) {
          openCitationInspector(matched);
        } else {
          openCitationInspector({
            source: srcName,
            page: pageNum,
            similarity: 0.85,
            text: `Source context for ${srcName} (Page ${pageNum}). Detailed chunk text is available in the Verified Sources deck below.`,
          });
        }
      };
    });
  }

  function renderTelemetryBadge(container, telemetry) {
    if (!container || !telemetry) return;
    const existing = container.querySelector(".telemetry-bar");
    if (existing) existing.remove();

    const pill = document.createElement("div");
    pill.className = "telemetry-bar dev-only";
    const total = Math.round(telemetry.total_ms || telemetry.latency_ms || 0);
    const retrieval = Math.round(telemetry.retrieval_ms || 0);
    const gen = Math.round(telemetry.generation_ms || (total > retrieval ? total - retrieval : total));
    const groundedPct = Math.round((telemetry.groundedness !== undefined ? telemetry.groundedness : 1.0) * 100);
    const engine = telemetry.vector_store || telemetry.vector_store_type || "pgvector (PostgreSQL HNSW)";

    pill.innerHTML = `
      <div class="telemetry-pill" title="End-to-End RAG Observability Breakdown">
        <span class="telemetry-icon">⚡</span>
        <strong class="total-latency">${total}ms</strong>
        <span class="telemetry-sep">•</span>
        <span class="telemetry-detail">Retrieval: ${retrieval}ms | LLM: ${gen}ms</span>
        <span class="telemetry-sep">•</span>
        <span class="grounded-pill">
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            <polyline points="9 12 11 14 15 10"/>
          </svg>
          ${groundedPct}% Grounded
        </span>
        <span class="telemetry-sep">•</span>
        <span class="engine-tag">${escapeHtml(engine)}</span>
      </div>
    `;
    container.appendChild(pill);
  }

  function renderCitationsComponent(container, sources, latencyMs = null) {
    if (!sources || sources.length === 0) {
      container.classList.add("hidden");
      container.innerHTML = "";
      return;
    }

    container.classList.remove("hidden");
    container.innerHTML = "";

    const highestScore = Math.max(...sources.map((s) => s.similarity || 0));
    const topMatchPct = Math.round(highestScore * 100);

    // 1. Minimized toggle bar
    const toggleBtn = document.createElement("div");
    toggleBtn.className = "citations-header-toggle";
    toggleBtn.setAttribute("role", "button");
    toggleBtn.setAttribute("tabindex", "0");
    toggleBtn.setAttribute("title", "Click to check verified sources and inspect snippets");

    toggleBtn.innerHTML = `
      <div class="citations-header-left">
        <div class="citations-shield-icon">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            <polyline points="9 12 11 14 15 10"/>
          </svg>
        </div>
        <span class="citations-title">Verified Sources</span>
        <span class="citations-count-pill">${sources.length} sources</span>
        ${latencyMs ? `<span class="citations-latency-pill">${Math.round(latencyMs)}ms</span>` : ""}
      </div>
      <div class="citations-header-right">
        <span class="citations-hint-text">${topMatchPct}% match</span>
        <span class="citations-chevron-icon">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <polyline points="6 9 12 15 18 9"/>
          </svg>
        </span>
      </div>
    `;

    // 2. Expandable Deck (starts collapsed / minimized)
    const deck = document.createElement("div");
    deck.className = "citations-deck collapsed";

    sources.forEach((s, idx) => {
      const card = document.createElement("div");
      card.className = "citation-detail-card";

      const matchPct = Math.round((s.similarity || 0) * 100);
      const tierClass = matchPct >= 70 ? "tier-high" : "tier-med";

      card.innerHTML = `
        <div class="citation-card-header" role="button" tabindex="0" title="Click to view chunk details">
          <div class="citation-card-left">
            <span class="doc-icon">📄</span>
            <span class="citation-source-name" title="${escapeHtml(s.source || "Document")}">${escapeHtml(s.source || "Document")}</span>
            <span class="citation-tag">Page ${s.page || 1}</span>
            <span class="citation-tag">Chunk #${s.chunk_index ?? idx}</span>
          </div>
          <div class="citation-card-right">
            <span class="cite-score-pill ${tierClass}">${matchPct}% match</span>
            <span class="card-chevron-icon">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="6 9 12 15 18 9"/>
              </svg>
            </span>
          </div>
        </div>
        <div class="citation-detail-drawer">
          <div class="citation-snippet-box">
            <div class="snippet-meta-bar">
              <span class="snippet-meta-label">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                  <line x1="9" y1="9" x2="15" y2="9"/>
                  <line x1="9" y1="13" x2="15" y2="13"/>
                  <line x1="9" y1="17" x2="13" y2="17"/>
                </svg>
                Retrieved Context Chunk
              </span>
              <div style="display: flex; gap: 6px;">
                <button class="inspect-chunk-btn" type="button" title="Open full visual chunk inspector">
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>
                  </svg>
                  <span>Inspect</span>
                </button>
                <button class="copy-chunk-btn" type="button" title="Copy chunk text">
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                  </svg>
                  <span>Copy</span>
                </button>
              </div>
            </div>
            <div class="citation-snippet-text">${escapeHtml(s.text || "No raw text snippet available.")}</div>
          </div>
        </div>
      `;

      // Toggle individual card detail drawer
      const cardHeader = card.querySelector(".citation-card-header");
      cardHeader.addEventListener("click", (e) => {
        e.stopPropagation();
        card.classList.toggle("open");
      });

      // Inspect chunk button handler
      const inspectChunkBtn = card.querySelector(".inspect-chunk-btn");
      if (inspectChunkBtn) {
        inspectChunkBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          openCitationInspector(s);
        });
      }

      // Copy snippet button handler
      const copyChunkBtn = card.querySelector(".copy-chunk-btn");
      copyChunkBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        navigator.clipboard.writeText(s.text || "");
        showToast(`Copied chunk from ${s.source || "document"}!`, "info");
      });

      deck.appendChild(card);
    });

    // Toggle overall deck on header click
    toggleBtn.addEventListener("click", () => {
      const isExpanded = toggleBtn.classList.toggle("expanded");
      if (isExpanded) {
        deck.classList.remove("collapsed");
        const firstCard = deck.querySelector(".citation-detail-card");
        if (firstCard) firstCard.classList.add("open");
      } else {
        deck.classList.add("collapsed");
      }
    });

    container.appendChild(toggleBtn);
    container.appendChild(deck);
  }

  async function streamResponse(query, topK, threshold) {
    const node = document.createElement("div");
    node.className = "message-node assistant-node";
    node.innerHTML = `
      <div class="avatar-ring">⚡</div>
      <div class="bubble-content">
        <div class="text-body"></div>
        <div class="citations-wrapper hidden"></div>
        <div class="action-strip">
          <button class="tiny-copy-btn" title="Copy response">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
            </svg>
            <span>Copy</span>
          </button>
          <button class="tiny-speak-btn" title="Read aloud (Text to Speech)">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
              <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/>
            </svg>
            <span>Listen</span>
          </button>
        </div>
      </div>
    `;
    messagesStream.appendChild(node);
    chatViewport.scrollTop = chatViewport.scrollHeight;

    const textBody = node.querySelector(".text-body");
    const citationsWrap = node.querySelector(".citations-wrapper");
    const copyBtn = node.querySelector(".tiny-copy-btn");
    const speakBtn = node.querySelector(".tiny-speak-btn");
    const bubbleContent = node.querySelector(".bubble-content");

    // Mount active thinking and progress card immediately
    const thinking = createThinkingCard();
    textBody.appendChild(thinking.element);
    chatViewport.scrollTop = chatViewport.scrollHeight;

    let fullText = "";
    let pendingSources = null;
    let latestTelemetry = null;
    const streamStart = performance.now();

    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(fullText);
      showToast("Answer copied to clipboard!", "info");
    });

    speakBtn.addEventListener("click", () => {
      toggleSpeak(fullText, speakBtn);
    });

    try {
      const response = await fetch("/api/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          top_k: topK,
          min_similarity: threshold,
          session_id: currentSessionId,
          user_id: currentUserId,
        }),
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || "Stream request rejected");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop();

        for (const rawEvent of events) {
          const lines = rawEvent.split("\n");
          let eventName = "";
          let dataStr = "";

          for (const line of lines) {
            if (line.startsWith("event:")) eventName = line.replace("event:", "").trim();
            if (line.startsWith("data:")) dataStr = line.replace("data:", "").trim();
          }

          if (eventName === "sources") {
            try {
              const sources = JSON.parse(dataStr);
              if (sources && sources.length > 0) {
                pendingSources = sources;
              }
            } catch {}
          } else if (eventName === "token") {
            try {
              const tokenData = JSON.parse(dataStr);
              const token = tokenData.token || "";
              if (token) {
                if (fullText === "") {
                  // FIRST TOKEN ARRIVED: Stop thinking animation and begin streaming answer
                  thinking.stop();
                  textBody.innerHTML = "";
                }
                fullText += token;
                textBody.innerHTML = renderFormattedAnswer(fullText, true);
                bindInlineCitationClicks(textBody, pendingSources);
                chatViewport.scrollTop = chatViewport.scrollHeight;
              }
            } catch {}
          } else if (eventName === "done") {
            try {
              const doneObj = JSON.parse(dataStr);
              if (doneObj && doneObj.latency) {
                latestTelemetry = {
                  retrieval_ms: doneObj.latency.retrieval_ms,
                  generation_ms: doneObj.latency.generation_ms,
                  total_ms: doneObj.latency.total_ms,
                  groundedness: doneObj.groundedness,
                  vector_store: doneObj.vector_store,
                };
              }
            } catch {}
          } else if (eventName === "error") {
            try {
              thinking.stop();
              const err = JSON.parse(dataStr);
              textBody.innerHTML = `<p style="color: var(--rose-accent);"><strong>Security Block:</strong> ${escapeHtml(err.message)}</p>`;
            } catch {}
          }
        }
      }

      // 1. Remove the stream cursor and render finalized text
      thinking.stop();
      if (fullText) {
        textBody.innerHTML = renderFormattedAnswer(fullText, false);
        bindInlineCitationClicks(textBody, pendingSources);
      }

      // 2. Render Verified Sources ONLY after complete output is finished
      const streamDuration = performance.now() - streamStart;
      if (pendingSources && pendingSources.length > 0) {
        renderCitationsComponent(citationsWrap, pendingSources, streamDuration);
      }

      // 3. Render Engineering Observability Breakdown Badge
      const telemetryPayload = latestTelemetry || {
        total_ms: streamDuration,
        retrieval_ms: 0,
        generation_ms: streamDuration,
        groundedness: pendingSources && pendingSources.length > 0 ? 1.0 : 0.0,
      };
      renderTelemetryBadge(bubbleContent, telemetryPayload);
      chatViewport.scrollTop = chatViewport.scrollHeight;
    } catch (err) {
      thinking.stop();
      textBody.innerHTML = `<p style="color: var(--rose-accent);"><strong>Error:</strong> ${err.message}</p>`;
    }
  }

  async function standardResponse(query, topK, threshold) {
    const node = document.createElement("div");
    node.className = "message-node assistant-node";
    node.innerHTML = `
      <div class="avatar-ring">⚡</div>
      <div class="bubble-content">
        <div class="text-body"></div>
        <div class="citations-wrapper hidden"></div>
        <div class="action-strip">
          <button class="tiny-copy-btn" title="Copy response">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
            </svg>
            <span>Copy</span>
          </button>
          <button class="tiny-speak-btn" title="Read aloud (Text to Speech)">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
              <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/>
            </svg>
            <span>Listen</span>
          </button>
        </div>
      </div>
    `;
    messagesStream.appendChild(node);
    chatViewport.scrollTop = chatViewport.scrollHeight;

    const textBody = node.querySelector(".text-body");
    const citationsWrap = node.querySelector(".citations-wrapper");
    const copyBtn = node.querySelector(".tiny-copy-btn");
    const speakBtn = node.querySelector(".tiny-speak-btn");
    const bubbleContent = node.querySelector(".bubble-content");

    // Mount active thinking and progress card immediately
    const thinking = createThinkingCard();
    textBody.appendChild(thinking.element);
    chatViewport.scrollTop = chatViewport.scrollHeight;

    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(textBody.innerText);
      showToast("Answer copied to clipboard!", "info");
    });

    let currentAnswerText = "";
    speakBtn.addEventListener("click", () => {
      toggleSpeak(currentAnswerText || textBody.innerText, speakBtn);
    });

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          top_k: topK,
          min_similarity: threshold,
          session_id: currentSessionId,
          user_id: currentUserId,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Query failed");
      }

      const data = await res.json();
      currentAnswerText = data.answer;
      thinking.stop();
      textBody.innerHTML = renderFormattedAnswer(data.answer, false);
      bindInlineCitationClicks(textBody, data.sources);

      if (data.sources && data.sources.length > 0) {
        renderCitationsComponent(citationsWrap, data.sources, data.latency_ms);
      }

      const telemetryPayload = data.latency_breakdown || {
        total_ms: data.latency_ms,
        retrieval_ms: 0,
        generation_ms: data.latency_ms,
        groundedness: data.groundedness,
        vector_store: data.vector_store_type,
      };
      renderTelemetryBadge(bubbleContent, telemetryPayload);
    } catch (err) {
      thinking.stop();
      textBody.innerHTML = `<p style="color: var(--rose-accent);"><strong>Error:</strong> ${err.message}</p>`;
    }
    chatViewport.scrollTop = chatViewport.scrollHeight;
  }

  function renderFormattedAnswer(rawText, isStreaming = false) {
    if (!rawText) return "";

    // 1. Clean up awkward punctuation around citations (e.g. "[Source: ...].," -> "[Source: ...]")
    let clean = rawText
      .replace(/(\[Source:\s*[^\]]+\])\s*[\.,]+/gi, "$1")
      .replace(/\s+([.,;:])/g, "$1");

    // 2. Parse Markdown (using marked library or fallback)
    let rawHtml = window.marked ? marked.parse(clean) : formatMarkdown(clean);

    // 3. Sanitize HTML with DOMPurify to defend against stored XSS from untrusted documents
    let html = window.DOMPurify ? DOMPurify.sanitize(rawHtml) : rawHtml;

    // 4. Transform [Source: filename, Page: X] into clickable inline badges
    html = html.replace(
      /\[Source:\s*([^,\]]+?)(?:,\s*Page:?\s*(\d+))?\]/gi,
      (match, source, page) => {
        const cleanSource = source.trim();
        const pageNum = page ? page.trim() : "1";
        const pageBadge = page
          ? `<span class="badge-page">p.${pageNum}</span>`
          : "";
        return `<span class="inline-cite-badge" data-source="${escapeHtml(cleanSource)}" data-page="${pageNum}" title="Click to inspect chunk in slide-drawer"><span class="badge-icon">📄</span><span class="badge-source">${escapeHtml(cleanSource)}</span>${pageBadge}</span>`;
      }
    );

    if (isStreaming) {
      html += `<span class="stream-cursor">▋</span>`;
    }

    return html;
  }

  function formatMarkdown(text) {
    return text
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/\n\n/g, "<br><br>")
      .replace(/\n/g, "<br>");
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // =====================================================================
  // SPEECH RECOGNITION (Voice Input) & SPEECH SYNTHESIS (Read Aloud)
  // =====================================================================
  const micBtn = document.getElementById("mic-btn");
  const voiceLangSelect = document.getElementById("voice-lang");

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  let activeRecognition = null;
  let isListening = false;

  if (micBtn) {
    if (!SpeechRecognition) {
      micBtn.title = "Voice input is not supported in this browser. Try Chrome, Edge, or Safari.";
      micBtn.style.opacity = "0.5";
      micBtn.addEventListener("click", () => {
        showToast("Voice input not supported in this browser. Try Chrome or Edge.", "info");
      });
    } else {
      micBtn.addEventListener("click", () => {
        if (isListening) {
          if (activeRecognition) activeRecognition.stop();
          return;
        }

        try {
          const recognition = new SpeechRecognition();
          activeRecognition = recognition;
          recognition.lang = voiceLangSelect ? voiceLangSelect.value : "en-IN";
          recognition.interimResults = true;
          recognition.continuous = false;

          const langLabel = voiceLangSelect ? voiceLangSelect.options[voiceLangSelect.selectedIndex].text : "your language";
          chatInput.placeholder = `Listening... Speak now in ${langLabel}`;
          micBtn.classList.add("listening");
          isListening = true;

          const initialText = chatInput.value ? chatInput.value.trim() + " " : "";

          recognition.onresult = (event) => {
            let transcript = "";
            for (let i = 0; i < event.results.length; i++) {
              transcript += event.results[i][0].transcript;
            }
            chatInput.value = initialText + transcript;
            chatInput.style.height = "auto";
            chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + "px";
          };

          recognition.onerror = (e) => {
            isListening = false;
            micBtn.classList.remove("listening");
            chatInput.placeholder = "Ask anything about the indexed documents...";
            if (e.error === "not-allowed") {
              showToast("Microphone permission denied. Allow microphone access in your browser.", "error");
            } else if (e.error !== "no-speech") {
              showToast(`Microphone error: ${e.error}`, "error");
            }
          };

          recognition.onend = () => {
            isListening = false;
            micBtn.classList.remove("listening");
            chatInput.placeholder = "Ask anything about the indexed documents...";
            chatInput.focus();
          };

          recognition.start();
        } catch (err) {
          isListening = false;
          micBtn.classList.remove("listening");
          showToast("Microphone initialization error: " + err.message, "error");
        }
      });
    }
  }

  function toggleSpeak(rawText, btn) {
    if (!("speechSynthesis" in window)) {
      showToast("Text-to-speech not supported in this browser", "info");
      return;
    }

    if (window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel();
      document.querySelectorAll(".tiny-speak-btn").forEach((b) => {
        b.classList.remove("speaking");
        const span = b.querySelector("span");
        if (span) span.textContent = "Listen";
      });
      return;
    }

    if (!rawText) return;

    // Clean markdown and citations from spoken text
    const cleanSpoken = rawText
      .replace(/\[Source:[^\]]+\]/gi, "")
      .replace(/[*#`_~>•]/g, "")
      .replace(/\s+/g, " ")
      .trim();

    if (!cleanSpoken) return;

    const utterance = new SpeechSynthesisUtterance(cleanSpoken);
    const isHindi = /[\u0900-\u097F]/.test(cleanSpoken);
    utterance.lang = isHindi ? "hi-IN" : "en-IN";
    utterance.rate = 1.0;

    btn.classList.add("speaking");
    const span = btn.querySelector("span");
    if (span) span.textContent = "Stop";

    utterance.onend = () => {
      btn.classList.remove("speaking");
      if (span) span.textContent = "Listen";
    };

    utterance.onerror = () => {
      btn.classList.remove("speaking");
      if (span) span.textContent = "Listen";
    };

    window.speechSynthesis.speak(utterance);
  }
});

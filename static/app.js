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
      const res = await fetch("/health");
      if (res.ok) {
        const data = await res.json();
        currentLLMProvider = data.llm_provider || "Online";
        systemBadge.textContent = `${currentLLMProvider} • ${data.vector_store_type}`;
        chunkCountBadge.textContent = data.total_indexed_chunks + " chunks";
      }
    } catch {
      systemBadge.textContent = "Connecting...";
    }
  }
  checkHealth();

  // Dynamic Indexed Documents Management
  async function fetchDocumentList() {
    try {
      const res = await fetch("/api/documents");
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
      docList.innerHTML = `<p style="font-size: 11px; color: var(--text-muted); text-align: center; padding: 14px 10px;">No documents indexed.<br>Upload a PDF or Markdown file above.</p>`;
      return;
    }

    docList.innerHTML = docs
      .map(
        (d) => `
        <div class="doc-item-row">
          <div class="doc-info" style="display: flex; flex-direction: column; overflow: hidden; max-width: 80%;">
            <span class="doc-name" title="${escapeHtml(d.source)}">📄 ${escapeHtml(d.source)}</span>
            <span class="doc-chunks-sub" style="font-size: 10.5px; color: var(--text-muted); padding-left: 20px;">${d.chunks_count} chunk${d.chunks_count === 1 ? "" : "s"}</span>
          </div>
          <button class="doc-del-btn" title="Delete document" data-source="${escapeHtml(d.source)}">
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
          const res = await fetch(`/api/documents/${encodeURIComponent(source)}`, { method: "DELETE" });
          if (res.ok) {
            showToast(`Deleted ${source} from vector store`, "success");
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
      const res = await fetch("/api/suggestions");
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

  // Clear Chat
  clearChatBtn.addEventListener("click", () => {
    messagesStream.innerHTML = "";
    heroWelcome.style.display = "block";
    hasMessages = false;
    showToast("Chat reset", "info");
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
              <button class="copy-chunk-btn" type="button" title="Copy chunk text">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                </svg>
                <span>Copy Chunk</span>
              </button>
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
        // Open the top match card automatically for convenience
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

    // Mount active thinking and progress card immediately
    const thinking = createThinkingCard();
    textBody.appendChild(thinking.element);
    chatViewport.scrollTop = chatViewport.scrollHeight;

    let fullText = "";
    let pendingSources = null;
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
        body: JSON.stringify({ query, top_k: topK, min_similarity: threshold }),
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
                // Buffer sources so they only show after the full response has finished
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
                chatViewport.scrollTop = chatViewport.scrollHeight;
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
      }

      // 2. Render Verified Sources ONLY after complete output is finished
      if (pendingSources && pendingSources.length > 0) {
        const streamDuration = performance.now() - streamStart;
        renderCitationsComponent(citationsWrap, pendingSources, streamDuration);
        chatViewport.scrollTop = chatViewport.scrollHeight;
      }
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
        body: JSON.stringify({ query, top_k: topK, min_similarity: threshold }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Query failed");
      }

      const data = await res.json();
      currentAnswerText = data.answer;
      thinking.stop();
      textBody.innerHTML = renderFormattedAnswer(data.answer, false);

      if (data.sources && data.sources.length > 0) {
        renderCitationsComponent(citationsWrap, data.sources, data.latency_ms);
      }
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
    let html = window.marked ? marked.parse(clean) : formatMarkdown(clean);

    // 3. Transform [Source: filename, Page: X] into sleek inline badges
    html = html.replace(
      /\[Source:\s*([^,\]]+?)(?:,\s*Page:?\s*(\d+))?\]/gi,
      (match, source, page) => {
        const cleanSource = source.trim();
        const pageBadge = page
          ? `<span class="badge-page">p.${page.trim()}</span>`
          : "";
        return `<span class="inline-cite-badge" title="${escapeHtml(match)}"><span class="badge-icon">📄</span><span class="badge-source">${escapeHtml(cleanSource)}</span>${pageBadge}</span>`;
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

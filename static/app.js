document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
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
  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (res.ok) {
        const data = await res.json();
        systemBadge.textContent = "Online • " + data.vector_store_type;
        chunkCountBadge.textContent = data.total_indexed_chunks + " chunks";
      }
    } catch {
      systemBadge.textContent = "Connecting...";
    }
  }
  checkHealth();

  // Populate Default Document in list
  function renderDocumentList(docName = "company_handbook.md", chunkCount = 4) {
    docList.innerHTML = `
      <div class="doc-item-row">
        <span class="doc-name">📄 ${docName}</span>
        <button class="doc-del-btn" title="Delete document" data-source="${docName}">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="3 6 5 6 21 6"/>
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
          </svg>
        </button>
      </div>
    `;

    document.querySelectorAll(".doc-del-btn").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        const source = e.currentTarget.getAttribute("data-source");
        try {
          const res = await fetch(`/api/documents/${encodeURIComponent(source)}`, { method: "DELETE" });
          if (res.ok) {
            showToast(`Deleted ${source} from vector store`, "success");
            docList.innerHTML = `<p style="font-size: 11px; color: var(--text-muted); text-align: center; padding: 10px;">No documents indexed</p>`;
            checkHealth();
          }
        } catch {
          showToast("Failed to delete document", "error");
        }
      });
    });
  }
  renderDocumentList();

  // Prompt Cards Click
  document.querySelectorAll(".prompt-card").forEach((card) => {
    card.addEventListener("click", () => {
      const query = card.getAttribute("data-query");
      if (query) {
        chatInput.value = query;
        chatInput.focus();
        chatForm.dispatchEvent(new Event("submit"));
      }
    });
  });

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

          renderDocumentList(filename, job.chunks_count);
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

    const topK = parseInt(topKInput.value);
    const threshold = parseFloat(thresholdInput.value);
    const isStream = streamToggle.checked;

    if (isStream) {
      await streamResponse(query, topK, threshold);
    } else {
      await standardResponse(query, topK, threshold);
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
        <div class="text-body"><span class="typing-cursor">▋</span></div>
        <div class="citations-wrapper hidden"></div>
        <div class="action-strip">
          <button class="tiny-copy-btn" title="Copy response">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
            </svg>
            <span>Copy</span>
          </button>
        </div>
      </div>
    `;
    messagesStream.appendChild(node);
    chatViewport.scrollTop = chatViewport.scrollHeight;

    const textBody = node.querySelector(".text-body");
    const citationsWrap = node.querySelector(".citations-wrapper");
    const copyBtn = node.querySelector(".tiny-copy-btn");

    let fullText = "";
    let pendingSources = null;
    const streamStart = performance.now();

    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(fullText);
      showToast("Answer copied to clipboard!", "info");
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
              fullText += tokenData.token || "";
              textBody.innerHTML = window.marked ? marked.parse(fullText + " ▋") : formatMarkdown(fullText + " ▋");
              chatViewport.scrollTop = chatViewport.scrollHeight;
            } catch {}
          } else if (eventName === "error") {
            try {
              const err = JSON.parse(dataStr);
              textBody.innerHTML = `<p style="color: var(--rose-accent);"><strong>Security Block:</strong> ${escapeHtml(err.message)}</p>`;
            } catch {}
          }
        }
      }

      // 1. Remove the stream cursor and render the finalized text
      if (fullText) {
        textBody.innerHTML = window.marked ? marked.parse(fullText) : formatMarkdown(fullText);
      }

      // 2. Render Verified Sources ONLY after the complete output has finished
      if (pendingSources && pendingSources.length > 0) {
        const streamDuration = performance.now() - streamStart;
        renderCitationsComponent(citationsWrap, pendingSources, streamDuration);
        chatViewport.scrollTop = chatViewport.scrollHeight;
      }
    } catch (err) {
      textBody.innerHTML = `<p style="color: var(--rose-accent);"><strong>Error:</strong> ${err.message}</p>`;
    }
  }

  async function standardResponse(query, topK, threshold) {
    const node = document.createElement("div");
    node.className = "message-node assistant-node";
    node.innerHTML = `
      <div class="avatar-ring">⚡</div>
      <div class="bubble-content">
        <div class="text-body"><p>Searching vector index...</p></div>
        <div class="citations-wrapper hidden"></div>
        <div class="action-strip">
          <button class="tiny-copy-btn" title="Copy response">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
            </svg>
            <span>Copy</span>
          </button>
        </div>
      </div>
    `;
    messagesStream.appendChild(node);
    chatViewport.scrollTop = chatViewport.scrollHeight;

    const textBody = node.querySelector(".text-body");
    const citationsWrap = node.querySelector(".citations-wrapper");
    const copyBtn = node.querySelector(".tiny-copy-btn");

    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(textBody.innerText);
      showToast("Answer copied to clipboard!", "info");
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
      const parsedHtml = window.marked ? marked.parse(data.answer) : formatMarkdown(data.answer);
      textBody.innerHTML = parsedHtml;

      if (data.sources && data.sources.length > 0) {
        renderCitationsComponent(citationsWrap, data.sources, data.latency_ms);
      }
    } catch (err) {
      textBody.innerHTML = `<p style="color: var(--rose-accent);"><strong>Error:</strong> ${err.message}</p>`;
    }
    chatViewport.scrollTop = chatViewport.scrollHeight;
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
});

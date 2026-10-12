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

  async function streamResponse(query, topK, threshold) {
    const node = document.createElement("div");
    node.className = "message-node assistant-node";
    node.innerHTML = `
      <div class="avatar-ring">⚡</div>
      <div class="bubble-content">
        <div class="text-body"><span class="typing-cursor">▋</span></div>
        <div class="citations-wrapper hidden">
          <div class="citations-headline">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
              <polyline points="20 6 9 17 4 12"/>
            </svg>
            <span>Verified Sources</span>
          </div>
          <div class="citations-deck"></div>
        </div>
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
    const citationsDeck = node.querySelector(".citations-deck");
    const copyBtn = node.querySelector(".tiny-copy-btn");

    let fullText = "";

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
                citationsWrap.classList.remove("hidden");
                citationsDeck.innerHTML = sources
                  .map(
                    (s) => `
                    <div class="citation-chip">
                      <div class="cite-meta">
                        <span>📄 <strong>${escapeHtml(s.source)}</strong> (Page ${s.page}, Chunk #${s.chunk_index})</span>
                      </div>
                      <span class="cite-score-pill">${Math.round(s.similarity * 100)}% match</span>
                    </div>`
                  )
                  .join("");
              }
            } catch {}
          } else if (eventName === "token") {
            try {
              const tokenData = JSON.parse(dataStr);
              fullText += tokenData.token || "";
              textBody.innerHTML = window.marked ? marked.parse(fullText) : formatMarkdown(fullText);
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
      </div>
    `;
    messagesStream.appendChild(node);
    chatViewport.scrollTop = chatViewport.scrollHeight;

    const textBody = node.querySelector(".text-body");

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

      textBody.innerHTML = `
        <div>${parsedHtml}</div>
        ${
          data.sources.length > 0
            ? `<div class="citations-wrapper">
                <div class="citations-headline">Verified Sources (${data.latency_ms} ms):</div>
                <div class="citations-deck">
                  ${data.sources
                    .map(
                      (s) => `
                    <div class="citation-chip">
                      <div class="cite-meta">
                        <span>📄 <strong>${escapeHtml(s.source)}</strong> (Page ${s.page}, Chunk #${s.chunk_index})</span>
                      </div>
                      <span class="cite-score-pill">${Math.round(s.similarity * 100)}% match</span>
                    </div>`
                    )
                    .join("")}
                </div>
              </div>`
            : ""
        }
      `;
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

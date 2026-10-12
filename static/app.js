document.addEventListener("DOMContentLoaded", () => {
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const messagesContainer = document.getElementById("messages-container");
  const clearChatBtn = document.getElementById("clear-chat-btn");
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

  // Update slider labels
  topKInput.addEventListener("input", (e) => (topKVal.textContent = e.target.value));
  thresholdInput.addEventListener("input", (e) => (thresholdVal.textContent = e.target.value));

  // Auto-resize textarea
  chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
  });

  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  // Load system stats
  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (res.ok) {
        const data = await res.json();
        systemBadge.textContent = "Online (" + data.vector_store_type + ")";
        systemBadge.className = "badge";
        chunkCountBadge.textContent = data.total_indexed_chunks + " chunks";
      }
    } catch {
      systemBadge.textContent = "Offline";
      systemBadge.className = "badge failed";
    }
  }
  checkHealth();

  // Clear chat
  clearChatBtn.addEventListener("click", () => {
    messagesContainer.innerHTML = `
      <div class="message assistant-msg">
        <div class="msg-avatar">🤖</div>
        <div class="msg-content">
          <p>Chat cleared! Ready for your questions.</p>
        </div>
      </div>
    `;
  });

  // Suggested question chips
  document.querySelectorAll(".chip-query").forEach((chip) => {
    chip.addEventListener("click", () => {
      chatInput.value = chip.textContent.replace(/^[^\s]+\s/, "").trim();
      chatInput.focus();
      chatForm.dispatchEvent(new Event("submit"));
    });
  });

  // File Upload Handlers
  browseBtn.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("click", () => fileInput.click());

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
    jobBadge.textContent = "UPLOADING";
    jobBadge.className = "badge processing";
    progressFill.style.width = "25%";
    jobDetails.textContent = "Sending file to server...";

    try {
      const res = await fetch("/api/documents/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }

      const job = await res.json();
      pollJobStatus(job.job_id, file.name);
    } catch (err) {
      jobBadge.textContent = "FAILED";
      jobBadge.className = "badge failed";
      jobDetails.textContent = err.message;
    }
  }

  function pollJobStatus(jobId, filename) {
    jobBadge.textContent = "INDEXING";
    progressFill.style.width = "65%";
    jobDetails.textContent = "Parsing text, computing embeddings & indexing...";

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/documents/jobs/${jobId}`);
        if (!res.ok) return;
        const job = await res.json();

        if (job.status === "INDEXED") {
          clearInterval(interval);
          progressFill.style.width = "100%";
          jobBadge.textContent = "INDEXED";
          jobBadge.className = "badge";
          jobDetails.textContent = `Successfully indexed ${job.chunks_count} chunks!`;

          // Add to doc list
          const item = document.createElement("div");
          item.className = "doc-item";
          item.innerHTML = `<span>${filename}</span><span class="chip">${job.chunks_count} chunks</span>`;
          docList.appendChild(item);

          checkHealth();
          setTimeout(() => uploadCard.classList.add("hidden"), 4000);
        } else if (job.status === "FAILED") {
          clearInterval(interval);
          jobBadge.textContent = "FAILED";
          jobBadge.className = "badge failed";
          jobDetails.textContent = job.error_message || "Ingestion error";
        }
      } catch {
        clearInterval(interval);
      }
    }, 800);
  }

  // Chat Submission
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = chatInput.value.trim();
    if (!query) return;

    chatInput.value = "";
    chatInput.style.height = "auto";

    appendMessage("user", query);

    const topK = parseInt(topKInput.value);
    const threshold = parseFloat(thresholdInput.value);
    const isStream = streamToggle.checked;

    if (isStream) {
      await streamResponse(query, topK, threshold);
    } else {
      await standardResponse(query, topK, threshold);
    }
  });

  function appendMessage(role, text) {
    const msg = document.createElement("div");
    msg.className = `message ${role}-msg`;
    msg.innerHTML = `
      <div class="msg-avatar">${role === "user" ? "👤" : "🤖"}</div>
      <div class="msg-content">
        <p>${escapeHtml(text)}</p>
      </div>
    `;
    messagesContainer.appendChild(msg);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return msg;
  }

  // SSE Streaming Request
  async function streamResponse(query, topK, threshold) {
    const msg = document.createElement("div");
    msg.className = "message assistant-msg";
    msg.innerHTML = `
      <div class="msg-avatar">🤖</div>
      <div class="msg-content">
        <div class="typing-target"></div>
        <div class="citations-box hidden">
          <div class="citations-title">Verified Sources:</div>
          <div class="citations-list"></div>
        </div>
      </div>
    `;
    messagesContainer.appendChild(msg);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    const textTarget = msg.querySelector(".typing-target");
    const citationsBox = msg.querySelector(".citations-box");
    const citationsList = msg.querySelector(".citations-list");

    try {
      const response = await fetch("/api/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: topK, min_similarity: threshold }),
      });

      if (!response.ok) {
        throw new Error("Stream request failed");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let accumulatedText = "";

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
                citationsBox.classList.remove("hidden");
                citationsList.innerHTML = sources
                  .map(
                    (s) => `
                    <div class="citation-card">
                      <span>📄 <strong>${escapeHtml(s.source)}</strong> (Page ${s.page}, Chunk #${s.chunk_index})</span>
                      <span class="score">${Math.round(s.similarity * 100)}% match</span>
                    </div>`
                  )
                  .join("");
              }
            } catch {}
          } else if (eventName === "token") {
            try {
              const tokenData = JSON.parse(dataStr);
              accumulatedText += tokenData.token || "";
              textTarget.innerHTML = formatMarkdown(accumulatedText);
              messagesContainer.scrollTop = messagesContainer.scrollHeight;
            } catch {}
          } else if (eventName === "error") {
            try {
              const err = JSON.parse(dataStr);
              textTarget.innerHTML = `<p style="color: var(--red);">${escapeHtml(err.message)}</p>`;
            } catch {}
          }
        }
      }
    } catch (err) {
      textTarget.innerHTML = `<p style="color: var(--red);">Error retrieving response: ${err.message}</p>`;
    }
  }

  // Standard non-streaming fallback
  async function standardResponse(query, topK, threshold) {
    const msg = appendMessage("assistant", "Thinking...");
    const content = msg.querySelector(".msg-content");

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: topK, min_similarity: threshold }),
      });

      const data = await res.json();
      content.innerHTML = `
        <div>${formatMarkdown(data.answer)}</div>
        ${
          data.sources.length > 0
            ? `<div class="citations-box">
                <div class="citations-title">Verified Sources (${data.latency_ms} ms):</div>
                ${data.sources
                  .map(
                    (s) => `
                  <div class="citation-card">
                    <span>📄 <strong>${escapeHtml(s.source)}</strong> (Page ${s.page}, Chunk #${s.chunk_index})</span>
                    <span class="score">${Math.round(s.similarity * 100)}% match</span>
                  </div>`
                  )
                  .join("")}
              </div>`
            : ""
        }
      `;
    } catch (err) {
      content.innerHTML = `<p style="color: var(--red);">${err.message}</p>`;
    }
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
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

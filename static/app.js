const state = {
  pages: [],
  selected: null,
  dragging: null,
};

let controlsBound = false;

const statusEl = document.getElementById("status");
const fontSizeEl = document.getElementById("fontSize");
const fontColorEl = document.getElementById("fontColor");
const deleteBtn = document.getElementById("deleteBox");
const saveBtn = document.getElementById("saveBtn");
const pagesEl = document.getElementById("pages");
const editedLink = document.getElementById("editedPdfLink");

function setStatus(message) {
  if (statusEl) {
    statusEl.textContent = message;
  }
}

function updateEditedLink(url) {
  if (!editedLink) return;
  if (url) {
    editedLink.href = url;
    editedLink.style.display = "inline-flex";
  }
}

function polyToBbox(poly) {
  const xs = poly.map((p) => p[0]);
  const ys = poly.map((p) => p[1]);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const maxX = Math.max(...xs);
  const maxY = Math.max(...ys);
  return {
    x: minX,
    y: minY,
    w: Math.max(1, maxX - minX),
    h: Math.max(1, maxY - minY),
  };
}

function buildState(data) {
  state.pages = data.pages.map((page) => {
    const boxes = page.rec_polys.map((poly, index) => {
      const bbox = polyToBbox(poly);
      const text = page.edit_texts[index] ?? page.rec_texts[index] ?? "";
      const baseSize = Math.max(10, Math.min(28, bbox.h * 0.6));
      return {
        id: index,
        bbox,
        text,
        fontSize: baseSize,
        color: "#1c3c5a",
        deleted: false,
        element: null,
      };
    });
    return {
      pageIndex: page.page_index_0based,
      imageUrl: page.image_url,
      imageSize: page.image_size_px,
      boxes,
      element: null,
      overlay: null,
      image: null,
      scale: 1,
    };
  });
}

function updateBoxElement(page, box) {
  if (!box.element) return;
  const scale = page.scale || 1;
  const left = box.bbox.x * scale;
  const top = box.bbox.y * scale;
  const width = box.bbox.w * scale;
  const height = box.bbox.h * scale;

  box.element.style.left = `${left}px`;
  box.element.style.top = `${top}px`;
  box.element.style.width = `${width}px`;
  box.element.style.height = `${height}px`;
  box.element.style.color = box.color;
  box.element.querySelector(".text").style.fontSize = `${box.fontSize * scale}px`;
  box.element.classList.toggle("is-deleted", box.deleted);
}

function updatePageLayout(page) {
  if (!page.image || !page.overlay) return;
  const img = page.image;
  const scale = img.clientWidth / img.naturalWidth;
  page.scale = scale || 1;
  page.overlay.style.width = `${img.clientWidth}px`;
  page.overlay.style.height = `${img.clientHeight}px`;
  page.boxes.forEach((box) => updateBoxElement(page, box));
}

function selectBox(pageIdx, boxIndex) {
  if (state.selected) {
    const prev = state.selected;
    const prevPage = state.pages[prev.pageIdx];
    const prevBox = prevPage?.boxes[prev.boxIndex];
    if (prevBox?.element) {
      prevBox.element.classList.remove("selected");
    }
  }

  const page = state.pages[pageIdx];
  const box = page?.boxes[boxIndex];
  if (!page || !box) {
    state.selected = null;
    return;
  }
  state.selected = { pageIdx, boxIndex };
  if (box.element) {
    box.element.classList.add("selected");
  }
  if (fontSizeEl) fontSizeEl.value = Math.round(box.fontSize).toString();
  if (fontColorEl) fontColorEl.value = box.color;
}

function onDragStart(event, pageIdx, boxIndex) {
  if (event.button !== 0) return;
  const page = state.pages[pageIdx];
  const box = page?.boxes[boxIndex];
  if (!page || !box || box.deleted) return;
  selectBox(pageIdx, boxIndex);
  state.dragging = {
    pageIdx,
    boxIndex,
    startX: event.clientX,
    startY: event.clientY,
    originX: box.bbox.x,
    originY: box.bbox.y,
  };
  box.element.setPointerCapture(event.pointerId);
}

function onDragMove(event) {
  if (!state.dragging) return;
  const { pageIdx, boxIndex, startX, startY, originX, originY } = state.dragging;
  const page = state.pages[pageIdx];
  const box = page?.boxes[boxIndex];
  if (!page || !box) return;
  const scale = page.scale || 1;
  const dx = (event.clientX - startX) / scale;
  const dy = (event.clientY - startY) / scale;
  box.bbox.x = Math.max(0, originX + dx);
  box.bbox.y = Math.max(0, originY + dy);
  updateBoxElement(page, box);
}

function onDragEnd(event) {
  if (!state.dragging) return;
  const { pageIdx, boxIndex } = state.dragging;
  const page = state.pages[pageIdx];
  const box = page?.boxes[boxIndex];
  if (box?.element) {
    box.element.releasePointerCapture(event.pointerId);
  }
  state.dragging = null;
}

function renderPages() {
  pagesEl.innerHTML = "";
  state.pages.forEach((page, pageIdx) => {
    const pageEl = document.createElement("article");
    pageEl.className = "page";

    const header = document.createElement("div");
    header.className = "page-header";
    header.innerHTML = `<span class="page-number">Page ${page.pageIndex + 1}</span>`;

    const wrap = document.createElement("div");
    wrap.className = "page-wrap";

    const img = document.createElement("img");
    img.src = page.imageUrl;
    img.alt = `Page ${page.pageIndex + 1}`;

    const overlay = document.createElement("div");
    overlay.className = "overlay";

    img.addEventListener("load", () => {
      updatePageLayout(page);
    });

    window.addEventListener("resize", () => updatePageLayout(page));

    wrap.appendChild(img);
    wrap.appendChild(overlay);

    pageEl.appendChild(header);
    pageEl.appendChild(wrap);
    pagesEl.appendChild(pageEl);

    page.element = pageEl;
    page.overlay = overlay;
    page.image = img;

    page.boxes.forEach((box, index) => {
      const boxEl = document.createElement("div");
      boxEl.className = "text-box";
      boxEl.style.left = "0px";
      boxEl.style.top = "0px";

      const textEl = document.createElement("div");
      textEl.className = "text";
      textEl.contentEditable = "true";
      textEl.spellcheck = false;
      textEl.textContent = box.text;

      boxEl.appendChild(textEl);
      overlay.appendChild(boxEl);

      boxEl.addEventListener("pointerdown", (event) => {
        if (event.target.closest(".text")) {
          selectBox(pageIdx, index);
          return;
        }
        onDragStart(event, pageIdx, index);
      });
      boxEl.addEventListener("pointermove", onDragMove);
      boxEl.addEventListener("pointerup", onDragEnd);
      boxEl.addEventListener("pointercancel", onDragEnd);

      textEl.addEventListener("focus", () => selectBox(pageIdx, index));
      textEl.addEventListener("input", () => {
        const sanitized = textEl.textContent.replace(/\n+/g, " ").trim();
        box.text = sanitized;
        if (textEl.textContent !== sanitized) {
          textEl.textContent = sanitized;
        }
      });

      box.element = boxEl;
      updateBoxElement(page, box);
    });
  });
}

function buildSavePayload() {
  return {
    pages: state.pages.map((page) => ({
      page_index_0based: page.pageIndex,
      boxes: page.boxes.map((box) => ({
        id: box.id,
        deleted: box.deleted,
        bbox: box.bbox,
        text: box.text,
        font_size: box.fontSize,
        color: box.color,
      })),
    })),
  };
}

async function saveEdits() {
  const jobId = document.body.dataset.jobId;
  if (!jobId) return;
  const originalText = saveBtn ? saveBtn.textContent : null;
  if (saveBtn) {
    saveBtn.disabled = true;
    saveBtn.textContent = "Saving...";
  }
  setStatus("Saving edits...");
  try {
    const payload = buildSavePayload();
    const res = await fetch(`/api/job/${jobId}/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await res.json().catch(() => ({}));
    if (res.ok) {
      if (body.edited_pdf_url) {
        updateEditedLink(body.edited_pdf_url);
      }
      setStatus("Edits saved.");
    } else {
      setStatus(body.error ? `Save failed: ${body.error}` : "Save failed. Check server logs.");
    }
  } catch (error) {
    setStatus("Save failed. Check console/logs.");
  } finally {
    if (saveBtn) {
      saveBtn.disabled = false;
      saveBtn.textContent = originalText || "Save edits";
    }
  }
}

function bindControls() {
  if (controlsBound) return;
  controlsBound = true;
  if (fontSizeEl) {
    fontSizeEl.addEventListener("input", () => {
      if (!state.selected) return;
      const page = state.pages[state.selected.pageIdx];
      const box = page?.boxes[state.selected.boxIndex];
      if (!box) return;
      box.fontSize = Number(fontSizeEl.value);
      updateBoxElement(page, box);
    });
  }

  if (fontColorEl) {
    fontColorEl.addEventListener("input", () => {
      if (!state.selected) return;
      const page = state.pages[state.selected.pageIdx];
      const box = page?.boxes[state.selected.boxIndex];
      if (!box) return;
      box.color = fontColorEl.value;
      updateBoxElement(page, box);
    });
  }

  if (deleteBtn) {
    deleteBtn.addEventListener("click", () => {
      if (!state.selected) return;
      const page = state.pages[state.selected.pageIdx];
      const box = page?.boxes[state.selected.boxIndex];
      if (!box) return;
      box.deleted = true;
      updateBoxElement(page, box);
      setStatus("Box deleted.");
    });
  }

  if (saveBtn) {
    saveBtn.addEventListener("click", (event) => {
      event.preventDefault();
      saveEdits();
    });
  }
}

async function init() {
  const jobId = document.body.dataset.jobId;
  if (!jobId) return;
  bindControls();
  setStatus("Loading OCR data...");
  const res = await fetch(`/api/job/${jobId}`);
  if (!res.ok) {
    setStatus("Failed to load job data.");
    return;
  }
  const data = await res.json();
  buildState(data);
  renderPages();
  updateEditedLink(data.edited_pdf_url);
  setStatus("Ready.");
}

if (document.body.classList.contains("editor")) {
  init();
}

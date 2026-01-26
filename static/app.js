const state = {
  pages: [],
  selected: null,
  dragging: null,
  resizing: null,
  previewMode: "debug",
  selectedBoxes: new Set(),
  selecting: null,
  lastShiftKey: false,
};

let controlsBound = false;

const statusEl = document.getElementById("status");
const fontSizeEl = document.getElementById("fontSize");
const fontSizeNumberEl = document.getElementById("fontSizeNumber");
const fontColorEl = document.getElementById("fontColor");
const deleteBtn = document.getElementById("deleteBox");
const addBoxBtn = document.getElementById("addBox");
const saveBtn = document.getElementById("saveBtn");
const downloadBtn = document.getElementById("downloadBtn");
const batchTranslateBtn = document.getElementById("batchTranslateBtn");
const batchRestoreBtn = document.getElementById("batchRestoreBtn");
const pagesEl = document.getElementById("pages");
const editedLink = document.getElementById("editedPdfLink");
const previewEl = document.getElementById("pdfPreview");
const previewDebugBtn = document.getElementById("previewDebug");
const previewEditedBtn = document.getElementById("previewEdited");
const debugLinkEl = document.querySelector(".topbar-actions a[href*='overlay_debug.pdf']");

function setStatus(message) {
  if (statusEl) {
    statusEl.textContent = message;
  }
}

function setBatchButtonState(status) {
  if (!batchTranslateBtn) return;
  if (status === "running" || status === "queued") {
    batchTranslateBtn.disabled = true;
    batchTranslateBtn.textContent = "Batch 翻譯中...";
  } else {
    batchTranslateBtn.disabled = false;
    batchTranslateBtn.textContent = "Batch 翻譯";
  }
}

function renderBatchStatus(status) {
  if (!status || !statusEl) return;
  const label = status.status || "unknown";
  if (label === "completed") {
    setStatus("Batch 翻譯完成，已更新編輯內容。");
  } else if (label === "running" || label === "queued") {
    setStatus("Batch 翻譯進行中...");
  } else if (label === "failed") {
    setStatus(`Batch 翻譯失敗：${status.error || "unknown error"}`);
  } else {
    setStatus("Ready.");
  }
  setBatchButtonState(label);
}

function boxKey(pageIdx, boxIdx) {
  return `${pageIdx}:${boxIdx}`;
}

function getSelectedBoxes() {
  const items = [];
  state.selectedBoxes.forEach((key) => {
    const [pageIdx, boxIdx] = key.split(":").map((value) => Number.parseInt(value, 10));
    const page = state.pages[pageIdx];
    const box = page?.boxes[boxIdx];
    if (!page || !box || box.deleted) return;
    items.push({ pageIdx, boxIdx, page, box });
  });
  return items;
}

function applySelectionClasses() {
  state.pages.forEach((page, pageIdx) => {
    page.boxes.forEach((box, boxIdx) => {
      if (!box.element) return;
      box.element.classList.toggle("selected", state.selectedBoxes.has(boxKey(pageIdx, boxIdx)));
    });
  });
}

function syncInspectorFromBox(box) {
  if (!box) return;
  const sizeValue = Math.round(box.fontSize).toString();
  if (fontSizeEl) fontSizeEl.value = sizeValue;
  if (fontSizeNumberEl) fontSizeNumberEl.value = sizeValue;
  if (fontColorEl) fontColorEl.value = box.color;
}

function clearSelection() {
  state.selected = null;
  state.selectedBoxes.clear();
  applySelectionClasses();
}

function setSelection(pageIdx, boxIdx, additive = false) {
  const page = state.pages[pageIdx];
  const box = page?.boxes[boxIdx];
  if (!page || !box || box.deleted) {
    if (!additive) {
      clearSelection();
    }
    return;
  }
  const key = boxKey(pageIdx, boxIdx);
  if (!additive) {
    state.selectedBoxes.clear();
  }
  if (additive && state.selectedBoxes.has(key)) {
    state.selectedBoxes.delete(key);
  } else {
    state.selectedBoxes.add(key);
  }

  const selectedList = getSelectedBoxes();
  if (!selectedList.length) {
    state.selected = null;
    applySelectionClasses();
    return;
  }

  const primary = selectedList.find((item) => item.pageIdx === pageIdx && item.boxIdx === boxIdx) || selectedList[0];
  state.selected = { pageIdx: primary.pageIdx, boxIdx: primary.boxIdx };
  applySelectionClasses();
  syncInspectorFromBox(primary.box);
}

function boxesIntersect(a, b) {
  return a.x <= b.x + b.w && a.x + a.w >= b.x && a.y <= b.y + b.h && a.y + a.h >= b.y;
}

function deleteSelectedBoxes() {
  const selected = getSelectedBoxes();
  if (!selected.length) return;
  selected.forEach(({ page, box }) => {
    box.deleted = true;
    updateBoxElement(page, box);
  });
  clearSelection();
  setStatus("Box deleted.");
}

function updateEditedLink(url) {
  if (!editedLink) return;
  if (url) {
    editedLink.href = url;
    editedLink.style.display = "inline-flex";
    if (previewEditedBtn) {
      previewEditedBtn.disabled = false;
    }
    if (state.previewMode === "edited") {
      setPreviewMode("edited", url);
    }
  }
}

function setPreviewMode(mode, editedUrl) {
  if (!previewEl) return;
  const debugUrl = debugLinkEl?.href || previewEl.dataset.debugUrl || previewEl.getAttribute("data") || "";
  if (mode === "edited" && editedUrl) {
    previewEl.setAttribute("data", editedUrl);
    previewEditedBtn?.classList.add("is-active");
    previewDebugBtn?.classList.remove("is-active");
    state.previewMode = "edited";
  } else {
    previewEl.setAttribute("data", debugUrl);
    previewDebugBtn?.classList.add("is-active");
    previewEditedBtn?.classList.remove("is-active");
    state.previewMode = "debug";
  }
}

function triggerDownload(url) {
  if (!url) return;
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "edited.pdf";
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
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
      const fontSize = Number(page.font_sizes?.[index]);
      const color = page.colors?.[index] ?? "#1c3c5a";
      const id = page.box_ids?.[index] ?? index;
      return {
        id,
        bbox,
        text,
        fontSize: fontSize > 0 ? fontSize : baseSize,
        color,
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

function selectBox(pageIdx, boxIdx, additive = false) {
  setSelection(pageIdx, boxIdx, additive);
}

function buildDragGroup(pageIdx, boxIdx) {
  const selected = getSelectedBoxes().filter((item) => item.pageIdx === pageIdx);
  const key = boxKey(pageIdx, boxIdx);
  if (!state.selectedBoxes.has(key)) {
    state.selectedBoxes.add(key);
  }
  if (!selected.length) {
    const page = state.pages[pageIdx];
    const box = page?.boxes[boxIdx];
    if (!page || !box) return [];
    return [{ pageIdx, boxIdx, originX: box.bbox.x, originY: box.bbox.y }];
  }
  return selected.map((item) => ({
    pageIdx: item.pageIdx,
    boxIdx: item.boxIdx,
    originX: item.box.bbox.x,
    originY: item.box.bbox.y,
  }));
}

function onDragStart(event, pageIdx, boxIndex, groupMode = false) {
  if (event.button !== 0) return;
  const page = state.pages[pageIdx];
  const box = page?.boxes[boxIndex];
  if (!page || !box || box.deleted) return;
  const key = boxKey(pageIdx, boxIndex);
  const selectedOnPage = getSelectedBoxes().filter((item) => item.pageIdx === pageIdx);
  const shouldGroup = groupMode || (selectedOnPage.length > 1 && state.selectedBoxes.has(key));

  if (shouldGroup) {
    const key = boxKey(pageIdx, boxIndex);
    if (!state.selectedBoxes.has(key)) {
      state.selectedBoxes.add(key);
      applySelectionClasses();
    }
    state.selected = { pageIdx, boxIdx: boxIndex };
    syncInspectorFromBox(box);
  } else {
    selectBox(pageIdx, boxIndex);
  }
  const group = shouldGroup ? buildDragGroup(pageIdx, boxIndex) : [{ pageIdx, boxIdx: boxIndex, originX: box.bbox.x, originY: box.bbox.y }];
  state.dragging = {
    pageIdx,
    boxIndex,
    startX: event.clientX,
    startY: event.clientY,
    groupMode: shouldGroup,
    group,
  };
  box.element.setPointerCapture(event.pointerId);
}

function onDragMove(event) {
  if (state.resizing) return;
  if (!state.dragging) return;
  const { pageIdx, startX, startY, group } = state.dragging;
  const page = state.pages[pageIdx];
  if (!page) return;
  const scale = page.scale || 1;
  const dx = (event.clientX - startX) / scale;
  const dy = (event.clientY - startY) / scale;

  if (group && group.length) {
    group.forEach((item) => {
      const targetPage = state.pages[item.pageIdx];
      const targetBox = targetPage?.boxes[item.boxIdx];
      if (!targetPage || !targetBox) return;
      targetBox.bbox.x = Math.max(0, item.originX + dx);
      targetBox.bbox.y = Math.max(0, item.originY + dy);
      updateBoxElement(targetPage, targetBox);
    });
    return;
  }
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

function onResizeStart(event, pageIdx, boxIdx) {
  if (event.button !== 0) return;
  event.preventDefault();
  event.stopPropagation();
  const page = state.pages[pageIdx];
  const box = page?.boxes[boxIdx];
  if (!page || !box || box.deleted) return;
  setSelection(pageIdx, boxIdx);
  state.resizing = {
    pageIdx,
    boxIdx,
    startX: event.clientX,
    startY: event.clientY,
    originW: box.bbox.w,
    originH: box.bbox.h,
    originX: box.bbox.x,
    originY: box.bbox.y,
  };
  event.currentTarget.setPointerCapture(event.pointerId);
}

function onResizeMove(event) {
  if (!state.resizing) return;
  const { pageIdx, boxIdx, startX, startY, originW, originH, originX, originY } = state.resizing;
  const page = state.pages[pageIdx];
  const box = page?.boxes[boxIdx];
  if (!page || !box) return;
  const scale = page.scale || 1;
  const dx = (event.clientX - startX) / scale;
  const dy = (event.clientY - startY) / scale;
  const minSize = 12;
  let newW = Math.max(minSize, originW + dx);
  let newH = Math.max(minSize, originH + dy);

  if (page.imageSize && page.imageSize.length === 2) {
    const maxW = page.imageSize[0] - originX;
    const maxH = page.imageSize[1] - originY;
    if (Number.isFinite(maxW)) newW = Math.min(newW, maxW);
    if (Number.isFinite(maxH)) newH = Math.min(newH, maxH);
  }

  box.bbox.w = newW;
  box.bbox.h = newH;
  updateBoxElement(page, box);
}

function onResizeEnd(event) {
  if (!state.resizing) return;
  const { pageIdx, boxIdx } = state.resizing;
  const page = state.pages[pageIdx];
  const box = page?.boxes[boxIdx];
  if (box?.element) {
    box.element.releasePointerCapture(event.pointerId);
  }
  state.resizing = null;
}

function startRangeSelection(event, pageIdx) {
  if (event.button !== 0) return;
  if (event.target.closest(".text-box")) return;
  const page = state.pages[pageIdx];
  if (!page || !page.overlay) return;
  const bounds = page.overlay.getBoundingClientRect();
  const startX = event.clientX - bounds.left;
  const startY = event.clientY - bounds.top;
  const rectEl = page.selectionRect;
  if (!rectEl) return;
  const additive = event.shiftKey;

  rectEl.style.display = "block";
  rectEl.style.left = `${startX}px`;
  rectEl.style.top = `${startY}px`;
  rectEl.style.width = "0px";
  rectEl.style.height = "0px";

  state.selecting = {
    pageIdx,
    startX,
    startY,
    bounds,
    rectEl,
    captureEl: event.currentTarget,
    pointerId: event.pointerId,
    additive,
  };
  event.currentTarget.setPointerCapture(event.pointerId);
  if (!additive) {
    clearSelection();
  }
}

function updateRangeSelection(event) {
  if (!state.selecting) return;
  const { startX, startY, bounds, rectEl } = state.selecting;
  const currentX = event.clientX - bounds.left;
  const currentY = event.clientY - bounds.top;
  const left = Math.min(startX, currentX);
  const top = Math.min(startY, currentY);
  const width = Math.abs(currentX - startX);
  const height = Math.abs(currentY - startY);
  rectEl.style.left = `${left}px`;
  rectEl.style.top = `${top}px`;
  rectEl.style.width = `${width}px`;
  rectEl.style.height = `${height}px`;
  state.selecting.rect = { left, top, width, height };
}

function endRangeSelection(event) {
  if (!state.selecting) return;
  const { pageIdx, rect, rectEl, captureEl, pointerId, additive } = state.selecting;
  rectEl.style.display = "none";
  if (captureEl) {
    captureEl.releasePointerCapture(pointerId);
  }
  state.selecting = null;

  if (!rect || rect.width < 4 || rect.height < 4) {
    if (!additive) {
      clearSelection();
    }
    return;
  }

  const page = state.pages[pageIdx];
  if (!page) return;
  const scale = page.scale || 1;
  const selectionBox = {
    x: rect.left / scale,
    y: rect.top / scale,
    w: rect.width / scale,
    h: rect.height / scale,
  };

  page.boxes.forEach((box, boxIdx) => {
    if (box.deleted) return;
    if (boxesIntersect(box.bbox, selectionBox)) {
      state.selectedBoxes.add(boxKey(pageIdx, boxIdx));
    }
  });

  const selectedList = getSelectedBoxes();
  if (selectedList.length) {
    const primary = selectedList[0];
    state.selected = { pageIdx: primary.pageIdx, boxIdx: primary.boxIdx };
    syncInspectorFromBox(primary.box);
  } else {
    state.selected = null;
  }
  applySelectionClasses();
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
    img.draggable = false;
    img.addEventListener("dragstart", (event) => event.preventDefault());

    const overlay = document.createElement("div");
    overlay.className = "overlay";

    const selectionRect = document.createElement("div");
    selectionRect.className = "selection-rect";
    selectionRect.style.display = "none";
    overlay.appendChild(selectionRect);

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
    page.selectionRect = selectionRect;

    wrap.addEventListener("pointerdown", (event) => startRangeSelection(event, pageIdx));
    wrap.addEventListener("pointermove", updateRangeSelection);
    wrap.addEventListener("pointerup", endRangeSelection);
    wrap.addEventListener("pointercancel", endRangeSelection);

    page.boxes.forEach((box, index) => {
      createBoxElement(pageIdx, index);
    });
  });
}

function createBoxElement(pageIdx, boxIdx) {
  const page = state.pages[pageIdx];
  const box = page?.boxes[boxIdx];
  if (!page || !box || !page.overlay) return null;

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
  page.overlay.appendChild(boxEl);

  boxEl.addEventListener("pointerdown", (event) => {
    state.lastShiftKey = event.shiftKey;
    if (event.target.closest(".resize-handle")) {
      return;
    }
    if (event.target.closest(".text")) {
      selectBox(pageIdx, boxIdx, event.shiftKey);
      return;
    }
    if (event.shiftKey) {
      const key = boxKey(pageIdx, boxIdx);
      if (!state.selectedBoxes.has(key)) {
        state.selectedBoxes.add(key);
        applySelectionClasses();
      }
      onDragStart(event, pageIdx, boxIdx, true);
      return;
    }
    onDragStart(event, pageIdx, boxIdx);
  });
  boxEl.addEventListener("pointermove", onDragMove);
  boxEl.addEventListener("pointerup", onDragEnd);
  boxEl.addEventListener("pointercancel", onDragEnd);

  textEl.addEventListener("focus", () => {
    selectBox(pageIdx, boxIdx, state.lastShiftKey);
    state.lastShiftKey = false;
  });
  textEl.addEventListener("input", () => {
    const sanitized = textEl.textContent.replace(/\n+/g, " ").trim();
    box.text = sanitized;
    if (textEl.textContent !== sanitized) {
      textEl.textContent = sanitized;
    }
  });

  const handleEl = document.createElement("div");
  handleEl.className = "resize-handle";
  boxEl.appendChild(handleEl);

  handleEl.addEventListener("pointerdown", (event) => onResizeStart(event, pageIdx, boxIdx));
  handleEl.addEventListener("pointermove", onResizeMove);
  handleEl.addEventListener("pointerup", onResizeEnd);
  handleEl.addEventListener("pointercancel", onResizeEnd);

  box.element = boxEl;
  updateBoxElement(page, box);
  return textEl;
}

function addNewBox() {
  if (!state.pages.length) return;
  const targetPageIdx = state.selected?.pageIdx ?? 0;
  const page = state.pages[targetPageIdx];
  if (!page) return;

  const pageWidth = page.imageSize?.[0] ?? 800;
  const pageHeight = page.imageSize?.[1] ?? 1000;
  const defaultW = Math.min(220, pageWidth * 0.4);
  const defaultH = Math.min(48, pageHeight * 0.1);

  let x = 24;
  let y = 24;
  if (state.selected && state.selected.pageIdx === targetPageIdx) {
    const selBox = page.boxes[state.selected.boxIdx];
    if (selBox) {
      x = selBox.bbox.x;
      y = selBox.bbox.y + selBox.bbox.h + 12;
    }
  }
  if (x + defaultW > pageWidth) x = Math.max(0, pageWidth - defaultW - 12);
  if (y + defaultH > pageHeight) y = Math.max(0, pageHeight - defaultH - 12);

  const nextId = page.boxes.length ? Math.max(...page.boxes.map((box) => Number(box.id) || 0)) + 1 : 0;
  const box = {
    id: nextId,
    bbox: { x, y, w: defaultW, h: defaultH },
    text: "",
    fontSize: Math.max(10, Math.min(28, defaultH * 0.6)),
    color: "#1c3c5a",
    deleted: false,
    element: null,
  };
  page.boxes.push(box);
  const textEl = createBoxElement(targetPageIdx, page.boxes.length - 1);
  selectBox(targetPageIdx, page.boxes.length - 1);
  if (textEl) {
    textEl.focus();
  }
  setStatus("Text box added.");
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

async function loadJobData(jobId) {
  setStatus("Loading OCR data...");
  const res = await fetch(`/api/job/${jobId}`);
  if (!res.ok) {
    setStatus("Failed to load job data.");
    return null;
  }
  const data = await res.json();
  buildState(data);
  renderPages();
  updateEditedLink(data.edited_pdf_url);
  if (previewEditedBtn) {
    previewEditedBtn.disabled = !data.edited_pdf_url;
  }
  renderBatchStatus(data.batch_status);
  return data;
}

async function pollBatchStatus(jobId) {
  const res = await fetch(`/api/job/${jobId}/batch-status`);
  if (!res.ok) return null;
  const payload = await res.json();
  const status = payload.status || { status: "unknown" };
  renderBatchStatus(status);
  if (status.status === "running" || status.status === "queued") {
    setTimeout(() => pollBatchStatus(jobId), 5000);
  } else if (status.status === "completed") {
    await loadJobData(jobId);
  }
  return status;
}

async function saveEdits(shouldDownload = false) {
  const jobId = document.body.dataset.jobId;
  if (!jobId) return;
  const originalText = saveBtn ? saveBtn.textContent : null;
  if (saveBtn) {
    saveBtn.disabled = true;
    saveBtn.textContent = "Saving...";
  }
  if (downloadBtn) {
    downloadBtn.disabled = true;
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
        if (shouldDownload) {
          triggerDownload(body.edited_pdf_url);
        }
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
    if (downloadBtn) {
      downloadBtn.disabled = false;
    }
  }
}

function bindControls() {
  if (controlsBound) return;
  controlsBound = true;
  const applyFontSize = (value) => {
    if (!Number.isFinite(value)) return;
    const minValue = Math.max(
      Number(fontSizeEl?.min ?? value),
      Number(fontSizeNumberEl?.min ?? value),
    );
    const maxValue = Math.min(
      Number(fontSizeEl?.max ?? value),
      Number(fontSizeNumberEl?.max ?? value),
    );
    const clamped = Math.max(minValue, Math.min(maxValue, value));
    const displayValue = Math.round(clamped).toString();
    if (fontSizeEl) fontSizeEl.value = displayValue;
    if (fontSizeNumberEl) fontSizeNumberEl.value = displayValue;

    const selected = getSelectedBoxes();
    if (!selected.length) return;
    selected.forEach(({ page, box }) => {
      box.fontSize = clamped;
      updateBoxElement(page, box);
    });
  };

  if (fontSizeEl) {
    fontSizeEl.addEventListener("input", () => {
      applyFontSize(Number(fontSizeEl.value));
    });
  }

  if (fontSizeNumberEl) {
    fontSizeNumberEl.addEventListener("input", () => {
      applyFontSize(Number(fontSizeNumberEl.value));
    });
  }

  if (fontColorEl) {
    fontColorEl.addEventListener("input", () => {
      const selected = getSelectedBoxes();
      if (!selected.length) return;
      const value = fontColorEl.value;
      selected.forEach(({ page, box }) => {
        box.color = value;
        updateBoxElement(page, box);
      });
    });
  }

  if (deleteBtn) {
    deleteBtn.addEventListener("click", () => {
      deleteSelectedBoxes();
    });
  }

  if (addBoxBtn) {
    addBoxBtn.addEventListener("click", () => {
      addNewBox();
    });
  }

  if (saveBtn) {
    saveBtn.addEventListener("click", (event) => {
      event.preventDefault();
      saveEdits();
    });
  }

  if (downloadBtn) {
    downloadBtn.addEventListener("click", (event) => {
      event.preventDefault();
      saveEdits(true);
    });
  }

  if (previewDebugBtn) {
    previewDebugBtn.addEventListener("click", () => {
      setPreviewMode("debug");
    });
  }

  if (previewEditedBtn) {
    previewEditedBtn.addEventListener("click", () => {
      if (!previewEditedBtn.disabled && editedLink?.href) {
        setPreviewMode("edited", editedLink.href);
      }
    });
  }

  if (batchTranslateBtn) {
    batchTranslateBtn.addEventListener("click", async (event) => {
      event.preventDefault();
      const jobId = document.body.dataset.jobId;
      if (!jobId) return;
      setBatchButtonState("running");
      setStatus("啟動 Batch 翻譯...");
      try {
        const res = await fetch(`/api/job/${jobId}/batch-translate`, { method: "POST" });
        if (!res.ok) {
          setStatus("Batch 翻譯啟動失敗。");
          setBatchButtonState("failed");
          return;
        }
        setTimeout(() => pollBatchStatus(jobId), 3000);
      } catch (error) {
        setStatus("Batch 翻譯啟動失敗。");
        setBatchButtonState("failed");
      }
    });
  }

  if (batchRestoreBtn) {
    batchRestoreBtn.addEventListener("click", async (event) => {
      event.preventDefault();
      const jobId = document.body.dataset.jobId;
      if (!jobId) return;
      setStatus("回復翻譯結果中...");
      try {
        const res = await fetch(`/api/job/${jobId}/batch-restore`, { method: "POST" });
        const body = await res.json().catch(() => ({}));
        if (!res.ok) {
          setStatus(body.error ? `回復失敗：${body.error}` : "回復失敗。");
          return;
        }
        await loadJobData(jobId);
        setStatus("已回復翻譯結果。");
      } catch (error) {
        setStatus("回復失敗。");
      }
    });
  }

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Delete") return;
    const target = event.target;
    if (target && (target.isContentEditable || ["INPUT", "TEXTAREA"].includes(target.tagName))) {
      return;
    }
    deleteSelectedBoxes();
  });
}

async function init() {
  const jobId = document.body.dataset.jobId;
  if (!jobId) return;
  bindControls();
  const data = await loadJobData(jobId);
  if (!data) return;
  const status = data.batch_status?.status;
  if (status === "running" || status === "queued") {
    setTimeout(() => pollBatchStatus(jobId), 5000);
  }
}

if (document.body.classList.contains("editor")) {
  init();
}

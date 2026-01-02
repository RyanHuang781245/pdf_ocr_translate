async function getPdfjsLib() {
  if (window.pdfjsLib) {
    return window.pdfjsLib;
  }
  try {
    return await import(
      "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.min.mjs"
    );
  } catch (err) {
    throw new Error("PDF.js 載入失敗，請確認網路連線或 CDN 可用。");
  }
}

let pdfDoc = null;
let currentPage = 1;
let pdfScale = 1.5;

const pdfCanvas = document.getElementById("pdfCanvas");
const pdfCtx = pdfCanvas.getContext("2d");
const layerCanvasEl = document.getElementById("layerCanvas");
let fabricCanvas = null;

async function apiJson(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

async function loadMeta() {
  const meta = await apiJson(`/api/doc/${DOC_ID}/meta`);
  const sel = document.getElementById("pageSelect");
  sel.innerHTML = "";
  for (let i = 1; i <= meta.page_count; i++) {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = i;
    sel.appendChild(opt);
  }
  sel.value = "1";
}

async function loadPdf(pdfjsLib) {
  pdfDoc = await pdfjsLib.getDocument(`/api/doc/${DOC_ID}/pdf`).promise;
}

async function renderPage(pageNo) {
  currentPage = pageNo;
  const page = await pdfDoc.getPage(pageNo);
  const viewport = page.getViewport({ scale: pdfScale });

  pdfCanvas.width = viewport.width;
  pdfCanvas.height = viewport.height;

  layerCanvasEl.width = viewport.width;
  layerCanvasEl.height = viewport.height;

  await page.render({ canvasContext: pdfCtx, viewport }).promise;

  if (!fabricCanvas) {
    fabricCanvas = new fabric.Canvas("layerCanvas", {
      selection: true,
      preserveObjectStacking: true,
    });
  } else {
    fabricCanvas.setWidth(viewport.width);
    fabricCanvas.setHeight(viewport.height);
    fabricCanvas.clear();
  }

  await loadOrBuildInitialAnnotations(viewport);
}

function pdfPointsToCanvasPx(bboxPoints, pageViewport) {
  // bboxPoints: [x0,y0,x1,y1] in PDF points with origin bottom-left (common)
  // pdf.js viewport has transform; we can map using convertToViewportPoint
  const [x0, y0, x1, y1] = bboxPoints;
  const p0 = pageViewport.convertToViewportPoint(x0, y1);
  const p1 = pageViewport.convertToViewportPoint(x1, y0);
  // p0 = top-left, p1 = bottom-right in canvas space
  return { left: p0[0], top: p0[1], width: p1[0] - p0[0], height: p1[1] - p0[1] };
}

async function loadOrBuildInitialAnnotations(viewport) {
  const saved = await apiJson(`/api/doc/${DOC_ID}/page/${currentPage}/annotations`);
  if (saved.canvas) {
    fabricCanvas.loadFromJSON(saved.canvas, () => {
      fabricCanvas.renderAll();
    });
    return;
  }

  // 沒存過：用 blocks + translations 建初始旁註框
  const data = await apiJson(`/api/doc/${DOC_ID}/page/${currentPage}/blocks`);
  const blocks = data.blocks || [];

  blocks.forEach((b) => {
    const rect = pdfPointsToCanvasPx(b.bbox, viewport);

    // 預設：譯文放在原文右側（稍微偏移）
    const x = rect.left + rect.width + 10;
    const y = rect.top;

    const tb = new fabric.Textbox(b.text_translated || "", {
      left: Math.min(x, viewport.width - 220),
      top: y,
      width: 220,
      fontSize: 14,
      lineHeight: 1.2,
      editable: true,
      backgroundColor: "rgba(255,255,255,0.85)",
      borderColor: "#666",
      cornerColor: "#666",
    });

    // 附註：保存 block_id 以便之後你要做「回寫 translations」也方便
    tb.block_id = b.block_id;

    fabricCanvas.add(tb);
  });

  fabricCanvas.renderAll();
}

function attachUI() {
  document.getElementById("pageSelect").addEventListener("change", async (e) => {
    await renderPage(parseInt(e.target.value, 10));
  });

  document.getElementById("btnParse").addEventListener("click", async () => {
    document.getElementById("btnParse").disabled = true;
    try {
      await apiJson(`/api/doc/${DOC_ID}/parse_and_translate`, { method: "POST" });
      await renderPage(currentPage);
      alert("完成：已解析並翻譯");
    } catch (err) {
      alert(err.message);
    } finally {
      document.getElementById("btnParse").disabled = false;
    }
  });

  document.getElementById("btnSave").addEventListener("click", async () => {
    const canvasJson = fabricCanvas.toJSON(["block_id", "deleted"]);
    const payload = {
      canvas: {
        canvasWidth: fabricCanvas.getWidth(),
        canvasHeight: fabricCanvas.getHeight(),
        objects: canvasJson.objects,
      },
    };
    await apiJson(`/api/doc/${DOC_ID}/page/${currentPage}/annotations`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    alert("已儲存本頁");
  });

  document.getElementById("fontSize").addEventListener("input", (e) => {
    const obj = fabricCanvas.getActiveObject();
    if (!obj) return;
    obj.set("fontSize", parseInt(e.target.value, 10));
    fabricCanvas.requestRenderAll();
  });

  document.getElementById("btnDelete").addEventListener("click", () => {
    const obj = fabricCanvas.getActiveObject();
    if (!obj) return;
    // 軟刪除：標記 deleted，匯出時略過；也方便你之後做復原功能
    obj.deleted = true;
    obj.visible = false;
    fabricCanvas.discardActiveObject();
    fabricCanvas.requestRenderAll();
  });

  document.getElementById("btnExport").addEventListener("click", async () => {
    const res = await apiJson(`/api/doc/${DOC_ID}/export`, { method: "POST" });
    const link = document.getElementById("downloadLink");
    link.href = res.download_url;
    link.style.display = "inline";
    link.textContent = "下載翻譯 PDF";
    alert("匯出完成");
  });
}

(async function main() {
  const pdfjsLib = await getPdfjsLib();
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.worker.min.js";
  await loadMeta();
  await loadPdf(pdfjsLib);
  attachUI();
  await renderPage(1);
})();

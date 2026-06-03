(function () {
  function normalize(value) {
    return String(value || "").trim().toLowerCase();
  }

  function statusGroup(item) {
    const status = normalize(item?.job_status || item?.status_code || item?.status);
    if (["queued", "running", "cancel_requested", "uploaded", "ocr", "translate", "render", "structure", "html", "docx"].includes(status)) {
      return "active";
    }
    if (["completed", "done", "saved"].includes(status)) return "completed";
    if (status === "failed") return "failed";
    if (["cancelled", "canceled"].includes(status)) return "cancelled";
    if (status === "draft") return "draft";
    return status || "unknown";
  }

  function itemText(item, fields) {
    return fields.map((field) => normalize(item?.[field])).join(" ");
  }

  function apply(items, options) {
    const query = normalize(options.query);
    const status = normalize(options.status || "all");
    const mode = normalize(options.mode || "all");
    const fields = options.fields || ["job_name", "job_id", "creator_name", "owner_work_id", "status_label"];
    return (items || []).filter((item) => {
      if (query && !itemText(item, fields).includes(query)) return false;
      if (status !== "all" && statusGroup(item) !== status) return false;
      if (mode !== "all" && normalize(item?.document_mode || item?.job_type) !== mode) return false;
      return true;
    });
  }

  function bind(config) {
    const search = document.getElementById(config.searchId);
    const status = document.getElementById(config.statusId);
    const mode = config.modeId ? document.getElementById(config.modeId) : null;
    const reset = config.resetId ? document.getElementById(config.resetId) : null;

    const emit = () => config.onChange?.();
    search?.addEventListener("input", emit);
    status?.addEventListener("change", emit);
    mode?.addEventListener("change", emit);
    reset?.addEventListener("click", () => {
      if (search) search.value = "";
      if (status) status.value = "all";
      if (mode) mode.value = "all";
      emit();
    });
  }

  function paginate(items, state) {
    const pageSize = Number(state?.pageSize) > 0 ? Number(state.pageSize) : 10;
    const total = (items || []).length;
    const totalPages = Math.max(1, Math.ceil(total / pageSize));
    const currentPage = Math.min(Math.max(1, Number(state?.page) || 1), totalPages);
    if (state) state.page = currentPage;
    const start = (currentPage - 1) * pageSize;
    return {
      items: (items || []).slice(start, start + pageSize),
      page: currentPage,
      pageSize,
      start,
      end: Math.min(total, start + pageSize),
      total,
      totalPages,
    };
  }

  function renderPagination(container, pageInfo, state, onChange) {
    if (!container) return;
    container.innerHTML = "";
    if (!pageInfo || pageInfo.total <= pageInfo.pageSize) {
      container.style.display = "none";
      return;
    }
    container.style.display = "flex";

    const summary = document.createElement("span");
    summary.className = "jobs-pagination__summary";
    summary.textContent = `${pageInfo.start + 1}-${pageInfo.end} / ${pageInfo.total}`;

    const prev = document.createElement("button");
    prev.type = "button";
    prev.className = "ghost jobs-pagination__button";
    prev.textContent = "上一頁";
    prev.disabled = pageInfo.page <= 1;
    prev.addEventListener("click", () => {
      state.page = Math.max(1, pageInfo.page - 1);
      onChange?.();
    });

    const pageSelect = document.createElement("select");
    pageSelect.className = "jobs-pagination__select";
    pageSelect.setAttribute("aria-label", "切換頁碼");
    for (let page = 1; page <= pageInfo.totalPages; page += 1) {
      const option = document.createElement("option");
      option.value = String(page);
      option.textContent = `第 ${page} / ${pageInfo.totalPages} 頁`;
      option.selected = page === pageInfo.page;
      pageSelect.appendChild(option);
    }
    pageSelect.addEventListener("change", () => {
      state.page = Number(pageSelect.value) || 1;
      onChange?.();
    });

    const next = document.createElement("button");
    next.type = "button";
    next.className = "ghost jobs-pagination__button";
    next.textContent = "下一頁";
    next.disabled = pageInfo.page >= pageInfo.totalPages;
    next.addEventListener("click", () => {
      state.page = Math.min(pageInfo.totalPages, pageInfo.page + 1);
      onChange?.();
    });

    container.append(summary, prev, pageSelect, next);
  }

  window.JobFilters = { apply, bind, paginate, renderPagination };
})();

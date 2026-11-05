const app = document.querySelector("#app");
const searchForm = document.querySelector("#ticker-search");
const tickerInput = document.querySelector("#ticker-input");
const watchlistSort = { key: null, ascending: true };
const WATCHLIST_STORAGE_KEY = "stock-momentum.watchlist.v1";
let watchlistFilter = "";
let watchlistEditing = false;
let marketOverviewData = null;
let marketChartResizeTimeout;
let watchlistRefreshTimer;
const numberFormat = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 2 });
const currencyFormat = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 2,
});
const istDateFormat = new Intl.DateTimeFormat("en-IN", {
  dateStyle: "medium",
  timeStyle: "short",
  timeZone: "Asia/Kolkata",
});
const istDayFormat = new Intl.DateTimeFormat("en-IN", {
  dateStyle: "medium",
  timeZone: "Asia/Kolkata",
});

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character]);
}

function readSavedWatchlist() {
  const saved = localStorage.getItem(WATCHLIST_STORAGE_KEY);
  if (saved === null) return null;
  const tickers = JSON.parse(saved);
  if (!Array.isArray(tickers) || tickers.some((ticker) => typeof ticker !== "string")) {
    throw new Error("Saved watchlist data is invalid. Clear this browser's site data to reset it.");
  }
  return [...new Set(tickers.map(normalizeTicker).filter(Boolean))];
}

function saveWatchlist(tickers) {
  localStorage.setItem(WATCHLIST_STORAGE_KEY, JSON.stringify(tickers));
}

function displayNumber(value, suffix = "") {
  const number = Number(value);
  return Number.isFinite(number) ? `${numberFormat.format(number)}${suffix}` : "—";
}

function displayPercent(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  const number = Number(value);
  const prefix = number > 0 ? "+" : "";
  return `${prefix}${numberFormat.format(number)}%`;
}

function displayPrice(value, currency) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  if (!currency) return numberFormat.format(number);
  return new Intl.NumberFormat("en", {
    style: "currency",
    currency,
    maximumFractionDigits: 2,
  }).format(number);
}

function displayFinancialValue(value, currency, compact = false) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  if (!currency) return new Intl.NumberFormat("en-IN", {
    notation: compact ? "compact" : "standard",
    maximumFractionDigits: 2,
  }).format(number);
  return new Intl.NumberFormat("en", {
    style: "currency",
    currency,
    notation: compact ? "compact" : "standard",
    maximumFractionDigits: 2,
  }).format(number);
}

function displayPlainPercent(value) {
  return Number.isFinite(Number(value)) ? `${numberFormat.format(Number(value))}%` : "—";
}

function displayDate(value) {
  if (!value) return "Not available";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return escapeHtml(value);
  if (/^\d{4}-\d{2}-\d{2}$/.test(value)) return istDayFormat.format(date);
  return `${istDateFormat.format(date)} IST`;
}

function leanClass(lean) {
  if (lean === "Bullish") return "bullish";
  if (lean === "Bearish") return "bearish";
  return lean === "Neutral" ? "neutral" : "unavailable";
}

function signalClass(action) {
  return action === "BUY" ? "buy" : action === "SELL" ? "sell" : "hold";
}

function statusBadge(text, className) {
  return `<span class="badge ${className}">${escapeHtml(text)}</span>`;
}

async function getJson(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(`The server returned an unreadable response (${response.status}).`);
  }
  if (!response.ok) {
    const detail = typeof payload.detail === "string" ? payload.detail : "Request failed.";
    throw new Error(detail);
  }
  return payload;
}

function showError(message, title = "Could not load this page") {
  app.innerHTML = `
    <section class="panel error-state" role="alert">
      <div>
        <h2>${escapeHtml(title)}</h2>
        <p>${escapeHtml(message)}</p>
        <button class="button" type="button" onclick="location.reload()">Try again</button>
      </div>
    </section>`;
}

function setActiveNavigation() {
  const page = location.pathname.startsWith("/analyzer") || location.pathname.startsWith("/stock")
    ? "analyzer"
    : "home";
  document.querySelectorAll("[data-nav]").forEach((link) => {
    link.classList.toggle("active", link.dataset.nav === page);
    if (link.dataset.nav === page) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
}

function setupMobileNavigation() {
  const toggle = document.querySelector("#menu-toggle");
  const backdrop = document.querySelector("#sidebar-backdrop");
  const close = () => {
    document.body.classList.remove("sidebar-open");
    toggle.setAttribute("aria-expanded", "false");
  };
  toggle.addEventListener("click", () => {
    const open = document.body.classList.toggle("sidebar-open");
    toggle.setAttribute("aria-expanded", String(open));
  });
  backdrop.addEventListener("click", close);
  document.querySelectorAll(".side-nav a").forEach((link) => link.addEventListener("click", close));
}

function normalizeTicker(value) {
  const ticker = value.trim().toUpperCase();
  if (!ticker) return "";
  return /[.]/.test(ticker) || ticker.startsWith("^") ? ticker : `${ticker}.NS`;
}

function displayTicker(ticker) {
  return String(ticker ?? "").replace(/\.NS$/i, "");
}

function formatInfoValue(key, value) {
  if (value === null || value === undefined || value === "") return "Not available";
  if (key === "marketCap") {
    const number = Number(value);
    if (!Number.isFinite(number)) return escapeHtml(value);
    if (number >= 1e12) return `₹${numberFormat.format(number / 1e12)}T`;
    if (number >= 1e7) return `₹${numberFormat.format(number / 1e7)}Cr`;
    return currencyFormat.format(number);
  }
  if (["dividendYield", "revenueGrowth", "returnOnEquity"].includes(key)) {
    return displayPercent(Number(value) * 100);
  }
  return typeof value === "number" ? numberFormat.format(value) : escapeHtml(value);
}

function marketPrice(instrument) {
  if (instrument.status !== "available" || instrument.price === null) return "Unavailable";
  return new Intl.NumberFormat("en", {
    style: "currency",
    currency: instrument.currency,
    maximumFractionDigits: 2,
  }).format(instrument.price);
}

function marketChartMarkup(instrument) {
  if (!instrument || instrument.status !== "available" || !instrument.history.length) {
    return `<div class="market-chart-empty" role="status">${escapeHtml(instrument?.message || "Market chart is unavailable.")}</div>`;
  }
  const points = instrument.history
    .map((item) => ({ date: item.t, value: Number(item.value) }))
    .filter((item) => Number.isFinite(item.value));
  if (!points.length) return `<div class="market-chart-empty" role="status">Market chart is unavailable.</div>`;
  const width = Math.max(360, Math.min(1200, Math.round(window.innerWidth * 0.82)));
  const height = 300;
  const plot = { left: 14, right: width - 125, top: 18, bottom: 244 };
  const min = Math.min(...points.map((item) => item.value));
  const max = Math.max(...points.map((item) => item.value));
  const spread = max - min || Math.abs(max) * 0.01 || 1;
  const x = (index) => plot.left + index * (plot.right - plot.left) / Math.max(points.length - 1, 1);
  const y = (value) => plot.bottom - (value - min) / spread * (plot.bottom - plot.top);
  const path = points.map((point, index) => {
    const pointX = x(index);
    const pointY = y(point.value);
    return `${index ? "L" : "M"}${pointX.toFixed(2)},${pointY.toFixed(2)}`;
  }).join(" ");
  const area = `${path} L${plot.right},${plot.bottom} L${plot.left},${plot.bottom} Z`;
  const priceFormat = new Intl.NumberFormat("en", {
    style: "currency",
    currency: instrument.currency,
    maximumFractionDigits: Math.abs(max) < 100 ? 2 : 0,
  });
  const horizontalGrid = Array.from({ length: 5 }, (_, index) => {
    const ratio = index / 4;
    const gridY = plot.top + ratio * (plot.bottom - plot.top);
    const value = max - ratio * spread;
    return `<g><line x1="${plot.left}" y1="${gridY}" x2="${plot.right}" y2="${gridY}" stroke="rgba(176, 153, 218, .13)" stroke-width="1"/><text x="${plot.right + 12}" y="${gridY + 4}" fill="#a8a0b9" font-size="11">${escapeHtml(priceFormat.format(value))}</text></g>`;
  }).join("");
  const dateTickCount = Math.min(6, points.length);
  const dateTicks = Array.from({ length: dateTickCount }, (_, index) => {
    const pointIndex = dateTickCount === 1
      ? 0
      : Math.round(index * (points.length - 1) / (dateTickCount - 1));
    const point = points[pointIndex];
    const date = new Date(`${point.date}T00:00:00Z`);
    const label = Number.isNaN(date.getTime())
      ? point.date
      : new Intl.DateTimeFormat("en", {
        day: "numeric",
        month: "short",
        timeZone: "UTC",
      }).format(date);
    const anchor = index === 0 ? "start" : index === dateTickCount - 1 ? "end" : "middle";
    return `<text x="${x(pointIndex)}" y="${height - 12}" text-anchor="${anchor}" fill="#9188a6" font-size="11">${escapeHtml(label)}</text>`;
  }).join("");
  return `
    <svg class="market-chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(instrument.label)} daily price history with dates and prices">
      <defs><linearGradient id="market-fill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="#9b70ec" stop-opacity=".27"/><stop offset="100%" stop-color="#9b70ec" stop-opacity=".015"/></linearGradient></defs>
      ${horizontalGrid}
      <path d="${area}" fill="url(#market-fill)"></path>
      <path d="${path}" fill="none" stroke="#f0a52b" stroke-width="2.5" vector-effect="non-scaling-stroke"></path>
      ${dateTicks}
    </svg>`;
}

function renderMarketOverview(market) {
  const instruments = Array.isArray(market?.instruments) ? market.instruments : [];
  const activeKey = instruments.some((item) => item.key === renderMarketOverview.activeKey)
    ? renderMarketOverview.activeKey
    : "nifty";
  const active = instruments.find((item) => item.key === activeKey) || instruments[0];
  renderMarketOverview.activeKey = active?.key ?? "nifty";
  const quoteCards = instruments.map((instrument) => {
    const change = instrument.change_pct;
    const changeClass = Number(change) >= 0 ? "positive" : "negative";
    return `<button class="market-quote ${instrument.key === active?.key ? "active" : ""}" type="button" data-market-key="${escapeHtml(instrument.key)}" aria-pressed="${instrument.key === active?.key}">
      <span class="market-quote-label">${escapeHtml(instrument.label)}</span>
      <strong>${escapeHtml(marketPrice(instrument))}</strong>
      <span class="market-quote-meta">${instrument.status === "available" ? `<span class="${changeClass}">${displayPercent(change)}</span><span>${escapeHtml(instrument.unit)}</span>` : `<span class="unavailable-text">${escapeHtml(instrument.message || "Price unavailable")}</span>`}</span>
    </button>`;
  }).join("");
  return `
    <section class="market-overview" aria-label="Global market overview">
      ${market?.error ? `<p class="market-provider-error" role="status">${escapeHtml(market.error)}</p>` : ""}
      <div class="market-quote-grid">${quoteCards || `<div class="panel market-chart-empty">Market quotes are unavailable.</div>`}</div>
      <section class="panel market-chart-panel">
        <div class="market-chart-heading">
          <div><p class="eyebrow">Market trend</p><h2>${escapeHtml(active?.label || "NIFTY 50")}</h2><p class="subtle">${escapeHtml(active?.unit || "Daily close")} · ${active?.as_of ? `latest close ${escapeHtml(active.as_of)}` : "Data unavailable"}</p></div>
          <div class="market-chart-value">${active ? escapeHtml(marketPrice(active)) : "—"}<small>${escapeHtml(active?.asset_class || "Market index")}</small></div>
        </div>
        <div class="market-chart-wrap">${marketChartMarkup(active)}</div>
        <div class="market-chart-footer"><span>Daily history · last 90 sessions</span><span>Market data</span></div>
      </section>
    </section>`;
}

renderMarketOverview.activeKey = "nifty";

function bindMarketQuoteButtons(marketData) {
  app.querySelectorAll("[data-market-key]").forEach((button) => {
    button.addEventListener("click", () => {
      renderMarketOverview.activeKey = button.dataset.marketKey;
      app.querySelector(".market-overview").outerHTML = renderMarketOverview(marketData);
      bindMarketQuoteButtons(marketData);
    });
  });
}

function renderWatchlist(data, marketData) {
  marketOverviewData = marketData;
  const rows = [...(Array.isArray(data.rows) ? data.rows : [])];
  const availableRows = rows.filter((row) => row.status === "ok");
  const flows = data.flows ?? {};
  const flowDay = flows.days?.[0];
  app.innerHTML = `
    <div class="page-heading">
      <div>
        <div class="breadcrumb"><span>Workspace</span><b>/</b><strong>Market overview</strong></div>
        <h1>Market overview</h1>
        <p class="muted">Nifty, global indices, energy prices, and your stock watchlist.</p>
      </div>
      <span class="toolbar-meta"><i class="live-dot"></i> Updated ${displayDate(data.updated_at)}</span>
    </div>
    ${renderMarketOverview(marketData)}
    <section class="panel panel-pad watchlist-flows" aria-label="FII and DII flows">
      <div class="panel-heading">
        <div><p class="eyebrow">Market-wide activity</p><h2>FII / DII flows</h2></div>
        <span class="subtle">${flowDay ? `As of ${displayDate(flowDay.date)}` : "Latest session unavailable"}</span>
      </div>
      ${flows.status === "available" && flowDay
        ? `<div class="flow-summary">
            <p><strong>FII/FPI net</strong><span>${displayNumber(flowDay.fii_net)} ₹ crore</span><small>Buy ${displayNumber(flowDay.fii_buy)} · Sell ${displayNumber(flowDay.fii_sell)}</small></p>
            <p><strong>DII net</strong><span>${displayNumber(flowDay.dii_net)} ₹ crore</span><small>Buy ${displayNumber(flowDay.dii_buy)} · Sell ${displayNumber(flowDay.dii_sell)}</small></p>
          </div>`
        : `<p class="muted">${escapeHtml(flows.message || "Market-wide flow data is temporarily unavailable.")}</p>`}
    </section>
    <section class="panel watchlist-panel ${watchlistEditing ? "is-editing" : ""}" aria-label="Watchlist">
        <div class="panel-heading">
          <div><p class="eyebrow">Your universe</p><h2>Watchlist</h2><p class="subtle">Live prices and daily change. Select a symbol for its stock-specific analysis.</p></div>
          <div class="watchlist-controls">
            <label class="watchlist-filter"><span aria-hidden="true">⌕</span><input id="watchlist-filter" type="search" value="${escapeHtml(watchlistFilter)}" placeholder="Search watchlist" aria-label="Search watchlist symbols"></label>
            <button id="watchlist-edit-toggle" class="icon-button" type="button" aria-label="${watchlistEditing ? "Done editing watchlist" : "Edit watchlist"}" aria-pressed="${watchlistEditing}" title="${watchlistEditing ? "Done editing" : "Edit watchlist"}">${watchlistEditing ? "✓" : "✎"}</button>
            <span class="table-count">${availableRows.length} / ${rows.length} AVAILABLE</span>
          </div>
        </div>
        <form id="add-watchlist-symbol" class="add-symbol-form watchlist-add-form">
          <label class="sr-only" for="watchlist-symbol">Add a symbol</label>
          <input id="watchlist-symbol" name="ticker" placeholder="Add ticker, e.g. INFY" maxlength="24" required>
          <button class="button" type="submit">Add symbol</button>
          <span class="editor-note">Up to 20 symbols. Search with a company’s common ticker.</span>
          <p id="watchlist-edit-message" class="watchlist-edit-message" role="status" aria-live="polite"></p>
        </form>
        <div class="table-wrap"><table>
        <thead><tr>
          <th scope="col"><button class="table-sort" data-sort="ticker">Stock</button></th>
          <th scope="col"><button class="table-sort" data-sort="current_price">Current price</button></th>
          <th scope="col"><button class="table-sort" data-sort="day_change_pct">Day return</button></th>
          <th scope="col"><span class="sr-only">Remove</span></th>
        </tr></thead>
        <tbody>
          ${rows.length ? rows.map((row) => {
            const available = row.status === "ok";
            return `            <tr tabindex="0" data-ticker="${escapeHtml(row.ticker)}" aria-label="Analyze ${escapeHtml(displayTicker(row.ticker))}">
            <td class="ticker-cell"><span class="ticker-symbol">${escapeHtml(displayTicker(row.ticker))}</span><small>${escapeHtml(row.company_name)}</small></td>
              <td>${available ? displayPrice(row.current_price, row.currency) : `<span class="unavailable-text">${escapeHtml(row.message || "Unavailable")}</span>`}<small class="table-subvalue">${row.as_of ? `Updated ${displayDate(row.as_of)}` : ""}</small></td>
              <td class="${row.day_change_pct === null || row.day_change_pct === undefined ? "" : Number(row.day_change_pct) >= 0 ? "positive-value" : "negative-value"}">${available ? displayPercent(row.day_change_pct) : "—"}</td>
              <td class="remove-cell"><button class="remove-symbol" type="button" data-remove="${escapeHtml(row.ticker)}" aria-label="Remove ${escapeHtml(displayTicker(row.ticker))} from watchlist">×</button></td>
            </tr>`;
          }).join("") : `<tr><td class="empty-table" colspan="4">Your watchlist is empty. Use the edit button to add a symbol.</td></tr>`}
        </tbody>
        </table></div>
        <p class="table-footnote">Prices refresh automatically about once a minute and may be delayed. Changes are saved in this browser only.</p>
    </section>
    <p class="disclaimer prominent-disclaimer">For experiments only. Not financial advice. Market data may be delayed, incomplete, or inaccurate.</p>`;

  const addForm = app.querySelector("#add-watchlist-symbol");
  const addInput = app.querySelector("#watchlist-symbol");
  const editMessage = app.querySelector("#watchlist-edit-message");
  const filterInput = app.querySelector("#watchlist-filter");
  filterInput.addEventListener("input", () => {
    watchlistFilter = filterInput.value.trim().toLowerCase();
    app.querySelectorAll("tbody tr[data-ticker]").forEach((row) => {
      row.hidden = !`${row.dataset.ticker} ${row.querySelector(".ticker-cell small")?.textContent || ""}`
        .toLowerCase().includes(watchlistFilter);
    });
  });
  app.querySelector("#watchlist-edit-toggle").addEventListener("click", () => {
    watchlistEditing = !watchlistEditing;
    renderWatchlist(data, marketData);
  });
  bindMarketQuoteButtons(marketData);
  addInput.addEventListener("input", () => addInput.setCustomValidity(""));
  addForm.addEventListener("submit", (event) => {
    event.preventDefault();
    addInput.setCustomValidity("");
    editMessage.textContent = "";
    const ticker = normalizeTicker(addInput.value);
    if (!ticker || !/^[A-Z0-9^.-]{1,20}$/.test(ticker)) {
      editMessage.textContent = "Enter a valid ticker symbol.";
      return;
    }
    if (rows.some((row) => row.ticker === ticker)) {
      editMessage.textContent = `${displayTicker(ticker)} is already in your watchlist.`;
      return;
    }
    if (rows.length >= 20) {
      editMessage.textContent = "A watchlist can contain at most 20 symbols.";
      return;
    }
    const updatedTickers = [...rows.map((row) => row.ticker), ticker];
    try {
      saveWatchlist(updatedTickers);
      loadWatchlist();
    } catch (error) {
      showError(error.message, "Could not save watchlist");
    }
  });

  app.querySelectorAll("[data-remove]").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      const updatedTickers = rows.filter((row) => row.ticker !== button.dataset.remove).map((row) => row.ticker);
      try {
        saveWatchlist(updatedTickers);
        loadWatchlist();
      } catch (error) {
        showError(error.message, "Could not update watchlist");
      }
    });
  });

  app.querySelectorAll(".table-sort").forEach((button) => {
    button.addEventListener("click", () => {
      const nextKey = button.dataset.sort;
      watchlistSort.ascending = watchlistSort.key === nextKey ? !watchlistSort.ascending : true;
      watchlistSort.key = nextKey;
      rows.sort((left, right) => {
        const a = left[nextKey] ?? "";
        const b = right[nextKey] ?? "";
        const result = typeof a === "number" && typeof b === "number"
          ? a - b
          : String(a).localeCompare(String(b));
        return result * (watchlistSort.ascending ? 1 : -1);
      });
      renderWatchlist({ ...data, rows }, marketData);
    });
  });

  app.querySelectorAll("tbody tr[data-ticker]").forEach((row) => {
    const openStock = () => { location.href = `/analyzer/${encodeURIComponent(displayTicker(row.dataset.ticker))}`; };
    row.addEventListener("click", openStock);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openStock();
      }
    });
  });
}

function makeCandlestickChart(candles) {
  const values = candles.filter((bar) => [bar.o, bar.h, bar.l, bar.c].every((value) => Number.isFinite(Number(value))));
  if (!values.length) return `<div class="empty-state"><p>Price chart is unavailable.</p></div>`;
  const width = 800;
  const height = 250;
  const padding = { top: 16, right: 16, bottom: 28, left: 66 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  const low = Math.min(...values.map((bar) => Number(bar.l)));
  const high = Math.max(...values.map((bar) => Number(bar.h)));
  const range = high - low || Math.abs(high) * 0.01 || 1;
  const y = (price) => padding.top + (high - Number(price)) / range * chartHeight;
  const step = chartWidth / values.length;
  const bodyWidth = Math.max(1, Math.min(7, step * 0.62));
  const grid = Array.from({ length: 5 }, (_, index) => {
    const value = high - range * index / 4;
    const lineY = padding.top + chartHeight * index / 4;
    return `<g><line x1="${padding.left}" y1="${lineY}" x2="${width - padding.right}" y2="${lineY}" stroke="#253146" stroke-dasharray="3 5"/><text x="${padding.left - 8}" y="${lineY + 4}" text-anchor="end" fill="#8190a8" font-size="10">${escapeHtml(numberFormat.format(value))}</text></g>`;
  }).join("");
  const bars = values.map((bar, index) => {
    const open = Number(bar.o);
    const close = Number(bar.c);
    const color = close >= open ? "#78b0ff" : "#ff7189";
    const x = padding.left + index * step + step / 2;
    const bodyY = Math.min(y(open), y(close));
    const bodyHeight = Math.max(1, Math.abs(y(open) - y(close)));
    return `<g><line x1="${x}" y1="${y(bar.h)}" x2="${x}" y2="${y(bar.l)}" stroke="${color}" stroke-width="1"/><rect x="${x - bodyWidth / 2}" y="${bodyY}" width="${bodyWidth}" height="${bodyHeight}" fill="${color}" rx="1"><title>${escapeHtml(new Date(bar.t).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" }))} O ${escapeHtml(numberFormat.format(open))} H ${escapeHtml(numberFormat.format(bar.h))} L ${escapeHtml(numberFormat.format(bar.l))} C ${escapeHtml(numberFormat.format(close))}</title></rect></g>`;
  }).join("");
  const firstDate = new Date(values[0].t);
  const lastDate = new Date(values[values.length - 1].t);
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="15-minute candlestick price chart">
    ${grid}${bars}
    <text x="${padding.left}" y="${height - 6}" fill="#8190a8" font-size="10">${escapeHtml(firstDate.toLocaleDateString("en-IN", { timeZone: "Asia/Kolkata" }))}</text>
    <text x="${width - padding.right}" y="${height - 6}" text-anchor="end" fill="#8190a8" font-size="10">${escapeHtml(lastDate.toLocaleDateString("en-IN", { timeZone: "Asia/Kolkata" }))}</text>
  </svg>`;
}

function makeTrendChart(series) {
  const values = (series ?? []).filter((item) => Number.isFinite(Number(item.adx)));
  if (!values.length) return `<p class="muted">ADX/DI history is unavailable.</p>`;
  const width = 700;
  const height = 190;
  const pad = { top: 14, right: 15, bottom: 24, left: 35 };
  const chartWidth = width - pad.left - pad.right;
  const chartHeight = height - pad.top - pad.bottom;
  const maximum = Math.max(50, ...values.map((item) => Math.max(Number(item.adx), Number(item.di_plus) || 0, Number(item.di_minus) || 0)));
  const x = (index) => pad.left + (values.length <= 1 ? chartWidth / 2 : index / (values.length - 1) * chartWidth);
  const y = (value) => pad.top + chartHeight - Number(value) / maximum * chartHeight;
  const line = (key, color) => {
    const points = values.map((item, index) => `${x(index)},${y(item[key] ?? 0)}`).join(" ");
    return `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
  };
  const thresholdY = y(25);
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="ADX and directional indicator history">
    <line x1="${pad.left}" y1="${thresholdY}" x2="${width - pad.right}" y2="${thresholdY}" stroke="#ffc46b" stroke-dasharray="4 4"/>
    <text x="${pad.left - 8}" y="${thresholdY + 4}" text-anchor="end" fill="#ffc46b" font-size="10">25</text>
    ${line("adx", "#9aabff")}${line("di_plus", "#55b8ff")}${line("di_minus", "#ff7189")}
  </svg>`;
}

function evidenceCard({ title, evidence, keyLabel, keyValue, description }) {
  const status = evidence?.status === "unavailable"
    ? "Unavailable"
    : evidence?.lean ?? "Available";
  const tone = evidence?.status === "unavailable"
    ? "unavailable"
    : evidence?.lean ? leanClass(evidence.lean) : "neutral";
  const flowDay = evidence?.source_url ? evidence.days?.[0] : null;
  return `<section class="panel evidence-card">
    <div class="evidence-top">
      <div><h3>${escapeHtml(title)}</h3><p class="evidence-number">${escapeHtml(keyValue)}</p></div>
      ${statusBadge(status, tone)}
    </div>
    <p class="evidence-caption">${escapeHtml(keyLabel)} · ${escapeHtml(description)}</p>
    ${flowDay ? `<p class="evidence-flow-details">FII/FPI buy ${displayNumber(flowDay.fii_buy)} · sell ${displayNumber(flowDay.fii_sell)}<br>DII buy ${displayNumber(flowDay.dii_buy)} · sell ${displayNumber(flowDay.dii_sell)} <span>(₹ crore)</span></p>` : ""}
    <p class="subtle">As of ${displayDate(evidence?.as_of)}</p>
    ${evidence?.source_url ? `<a class="text-link evidence-source" href="${escapeHtml(evidence.source_url)}" target="_blank" rel="noopener noreferrer">Source: NSE India <span aria-hidden="true">↗</span></a>` : ""}
  </section>`;
}

function renderHeadlines(headlines) {
  if (!headlines?.length) return `<p class="muted">No recent headlines were found.</p>`;
  return `<ul class="headline-list">${headlines.map((headline) => `
    <li>${escapeHtml(headline.title)}<span class="headline-meta">${escapeHtml(headline.source)}${headline.date ? ` · ${displayDate(headline.date)}` : ""}</span></li>
  `).join("")}</ul>`;
}

function renderFundamentalDetails(fundamentals) {
  const metrics = fundamentals?.metrics ?? [];
  const info = fundamentals?.info ?? {};
  const infoLabels = {
    sector: "Sector",
    industry: "Industry",
    marketCap: "Market cap",
    trailingPE: "Trailing P/E",
    forwardPE: "Forward P/E",
    dividendYield: "Dividend yield",
    beta: "Beta",
  };
  const analyst = fundamentals?.analyst ?? {};
  return `<div class="details-content">
    <section class="panel details-card">
      <h3>Fundamental metrics ${fundamentals?.score === null ? "" : `· ${escapeHtml(displayNumber(fundamentals.score, "/100"))}`}</h3>
      ${metrics.length ? `<div class="table-wrap"><table><thead><tr><th>Metric</th><th>Value</th><th>Score</th></tr></thead><tbody>
        ${metrics.map((metric) => {
          const score = Number(metric.score) || 0;
          const maximum = Number(metric.max_score) || 0;
          const width = maximum > 0 ? Math.max(0, Math.min(100, score / maximum * 100)) : 0;
          return `<tr><td>${escapeHtml(metric.name)}</td><td>${escapeHtml(metric.value ?? "—")}</td><td>${escapeHtml(score)}/${escapeHtml(maximum)}<div class="metric-bar"><span style="width:${width}%"></span></div></td></tr>`;
        }).join("")}
      </tbody></table></div>` : `<p class="muted">Fundamental metrics are unavailable.</p>`}
      ${fundamentals?.message ? `<p class="subtle">${escapeHtml(fundamentals.message)}</p>` : ""}
    </section>
    <section class="panel details-card">
      <h3>Company and analyst details</h3>
      <dl class="detail-list">${Object.entries(infoLabels).map(([key, label]) => `<div><dt>${escapeHtml(label)}</dt><dd>${formatInfoValue(key, info[key])}</dd></div>`).join("")}
      <div><dt>Analyst consensus</dt><dd>${escapeHtml(analyst.consensus ?? "Not available")}</dd></div>
      <div><dt>Analyst target</dt><dd>${analyst.target === null ? "Not available" : currencyFormat.format(analyst.target)}</dd></div>
      <div><dt>Target upside</dt><dd>${displayPercent(analyst.upside_pct)}</dd></div>
      <div><dt>Analyst count</dt><dd>${escapeHtml(analyst.count ?? "Not available")}</dd></div></dl>
    </section>
    <section class="panel details-card">
      <h3>ADX and directional indicators</h3>
      <div class="chart-legend">
        <span class="legend-item"><i class="legend-dot"></i>ADX</span>
        <span class="legend-item"><i class="legend-dot green"></i>DI+</span>
        <span class="legend-item"><i class="legend-dot red"></i>DI−</span>
        <span class="legend-item">25 trend threshold</span>
      </div>
      ${makeTrendChart(snapshotTrendSeries)}
    </section>
  </div>`;
}

let snapshotTrendSeries = [];

function renderStock(snapshot) {
  const prediction = snapshot.prediction ?? {};
  const evidence = snapshot.evidence ?? {};
  const trend = evidence.trend ?? {};
  const flows = evidence.flows ?? {};
  const fundamentals = evidence.fundamentals ?? {};
  const ai = snapshot.ai_read ?? {};
  snapshotTrendSeries = trend.series ?? [];
  const action = prediction.action ?? "HOLD";
  const latestFlow = flows.days?.[0] ?? {};
  const flowValue = flows.status === "available"
    ? `${displayNumber(latestFlow.fii_net)} / ${displayNumber(latestFlow.dii_net)}`
    : "Unavailable";
  const alignment = snapshot.alignment
    ? `${snapshot.alignment.agree} agree · ${snapshot.alignment.against} against · ${snapshot.alignment.neutral} neutral`
    : "No directional alignment for HOLD";
  app.innerHTML = `
    <div class="stock-title">
      <div><p class="eyebrow">Stock analysis · stock-specific model</p><h1>${escapeHtml(snapshot.company_name)} <span class="muted">${escapeHtml(displayTicker(snapshot.ticker))}</span></h1><p class="muted">Last updated ${displayDate(snapshot.updated_at)}</p></div>
      <div class="price-block"><span class="metric-label">Latest price</span><strong class="price">${currencyFormat.format(snapshot.price)}</strong></div>
    </div>
    <section class="overview-grid">
      <article class="panel prediction-card">
        <div class="prediction-top"><p class="eyebrow">Model prediction</p>${statusBadge(action, signalClass(action))}</div>
        <p class="signal-word ${signalClass(action)}">${escapeHtml(action)}</p>
        <p class="action-label">${action === "BUY" ? "Model signal: bullish direction" : action === "SELL" ? "Model signal: bearish direction" : "No directional action"}</p>
        <div class="metric-grid">
          <div class="metric"><span class="metric-label">Confidence</span><span class="metric-value">${displayPlainPercent(Number(prediction.confidence) * 100)}</span></div>
          <div class="metric"><span class="metric-label">Strength</span><span class="metric-value">${escapeHtml(prediction.strength ?? "—")}</span></div>
          <div class="metric"><span class="metric-label">Expected return*</span><span class="metric-value">${displayPercent(prediction.expected_return_pct)}</span></div>
          <div class="metric"><span class="metric-label">Alignment</span><span class="metric-value">${escapeHtml(alignment)}</span></div>
        </div>
        <p class="subtle">Prediction model: ${escapeHtml(prediction.model?.name ?? "Not available")} · trained on ${displayNumber(prediction.model?.training_samples)} labeled bars from this stock’s history</p>
      </article>
      <section class="panel chart-card" aria-label="Price chart">
        <div class="chart-head"><div><h2>15-minute price action</h2><span class="subtle">Stock history · ${escapeHtml(snapshot.period ?? "60d")}</span></div><label class="subtle" for="period-select">Window
          <select id="period-select" aria-label="Chart time window">
            ${["30d", "45d", "60d"].map((period) => `<option value="${period}" ${snapshot.period === period ? "selected" : ""}>${period}</option>`).join("")}
          </select>
        </label></div>
        <div class="chart-legend"><span class="legend-item"><i class="legend-dot green"></i>Up bar</span><span class="legend-item"><i class="legend-dot red"></i>Down bar</span></div>
        <div class="chart-wrap">${makeCandlestickChart(snapshot.candles_15m ?? [])}</div>
      </section>
    </section>
    <section class="evidence-grid" aria-label="Supporting evidence">
      ${evidenceCard({ title: "Trend", evidence: trend, keyLabel: "ADX", keyValue: trend.adx === null || trend.adx === undefined ? "Unavailable" : displayNumber(trend.adx), description: trend.strength ?? (trend.status === "unavailable" ? "Price trend unavailable" : "Trend strength") })}
      ${evidenceCard({ title: "FII / DII flows", evidence: flows, keyLabel: "Latest session · FII / DII net (₹ crore)", keyValue: flowValue, description: flows.message ?? "Daily cash-market flows" })}
      ${evidenceCard({ title: "Fundamentals", evidence: fundamentals, keyLabel: "Score", keyValue: fundamentals.score === null || fundamentals.score === undefined ? "Unavailable" : displayNumber(fundamentals.score, "/100"), description: fundamentals.status === "unavailable" ? fundamentals.message ?? "Fundamentals unavailable" : "Market data snapshot" })}
    </section>
    <section class="peer-quarter-grid" aria-label="Company comparisons and quarterly results">
      <article class="panel details-card">
        <h2>Industry peer comparison</h2>
        <p class="subtle">Market capitalization and reported fundamentals; search matches may not be direct competitors.</p>
        ${snapshot.peer_comparisons?.length ? `<div class="table-wrap"><table>
          <thead><tr><th>Company</th><th>Market cap</th><th>P/E</th><th>Revenue growth</th><th>ROE</th></tr></thead>
          <tbody>${snapshot.peer_comparisons.map((peer) => `<tr>
            <td class="ticker-cell"><span class="ticker-symbol">${escapeHtml(displayTicker(peer.ticker))}</span><small>${escapeHtml(peer.company_name)}</small></td>
            <td>${displayFinancialValue(peer.market_cap, peer.currency, true)}</td>
            <td>${displayNumber(peer.trailing_pe)}</td>
            <td>${peer.revenue_growth === null || peer.revenue_growth === undefined ? "—" : displayPercent(Number(peer.revenue_growth) * 100)}</td>
            <td>${peer.return_on_equity === null || peer.return_on_equity === undefined ? "—" : displayPercent(Number(peer.return_on_equity) * 100)}</td>
          </tr>`).join("")}</tbody>
        </table></div>` : `<p class="muted">${escapeHtml(snapshot.peer_comparisons_message || "Comparable companies are unavailable.")}</p>`}
      </article>
      <article class="panel details-card">
        <h2>Quarterly results</h2>
        <p class="subtle">Latest reported quarters available for this company.</p>
        ${snapshot.quarterly_results?.length ? `<div class="table-wrap"><table>
          <thead><tr><th>Quarter</th><th>Revenue</th><th>Operating income</th><th>Net income</th><th>Diluted EPS</th></tr></thead>
          <tbody>${snapshot.quarterly_results.map((quarter) => `<tr>
            <td>${escapeHtml(displayDate(quarter.quarter))}</td>
            <td>${displayFinancialValue(quarter.revenue, quarter.currency, true)}</td>
            <td>${displayFinancialValue(quarter.operating_income, quarter.currency, true)}</td>
            <td>${displayFinancialValue(quarter.net_income, quarter.currency, true)}</td>
            <td>${displayFinancialValue(quarter.diluted_eps, quarter.currency)}</td>
          </tr>`).join("")}</tbody>
        </table></div>` : `<p class="muted">${escapeHtml(snapshot.quarterly_results_message || "Quarterly results are unavailable.")}</p>`}
      </article>
    </section>
    <section class="panel read-card">
      <div class="read-header"><div><p class="eyebrow">Supporting context · not a signal</p><h2>AI market read</h2></div>${statusBadge(ai.status === "ok" ? `AI mood · ${ai.mood}` : "AI read unavailable", ai.status === "ok" ? leanClass(ai.mood) : "unavailable")}</div>
      <p class="read-summary">${escapeHtml(ai.status === "ok" ? ai.summary : ai.message ?? ai.summary ?? "AI read unavailable. The prediction and evidence remain available.")}</p>
      ${ai.risks?.length ? `<h3>Risks noted</h3><ul class="risks">${ai.risks.map((risk) => `<li>${escapeHtml(risk)}</li>`).join("")}</ul>` : ""}
      <h3>Recent headlines</h3>${renderHeadlines(ai.headlines ?? [])}
    </section>
    <details class="details">
      <summary>More details</summary>
      ${renderFundamentalDetails(fundamentals)}
    </details>
    <p class="disclaimer prominent-disclaimer">For experiments only. Not financial advice. The stock-specific model is fitted from available historical bars; expected return is an estimate, not a tested or guaranteed outcome. Market and company data may be delayed or incomplete.</p>`;

  app.querySelector("#period-select").addEventListener("change", (event) => {
    loadStock(snapshot.ticker, event.target.value);
  });
}

async function loadWatchlist(refreshOnly = false) {
  window.clearTimeout(watchlistRefreshTimer);
  if (!refreshOnly) {
    app.innerHTML = `<section class="loading-state panel"><span class="spinner" aria-hidden="true"></span><p>Loading watchlist…</p></section>`;
  }
  try {
    const savedTickers = readSavedWatchlist();
    const query = savedTickers === null ? "" : `?${new URLSearchParams({ tickers: savedTickers.join(",") })}`;
    let data;
    let marketData = marketOverviewData;
    if (refreshOnly) {
      data = await getJson(`/api/index${query}`);
    } else {
      [data, marketData] = await Promise.all([
        getJson(`/api/index${query}`),
        getJson("/api/market").catch((error) => ({
          instruments: [],
          error: `Market overview could not be loaded: ${error.message}`,
        })),
      ]);
    }
    if (savedTickers === null) saveWatchlist(data.rows.map((row) => row.ticker));
    renderWatchlist(data, marketData);
  } catch (error) {
    showError(error.message, "Watchlist unavailable");
  } finally {
    if (location.pathname === "/") {
      watchlistRefreshTimer = window.setTimeout(() => loadWatchlist(true), 60000);
    }
  }
}

function renderAnalyzerHome() {
  app.innerHTML = `
    <div class="page-heading">
      <div><div class="breadcrumb"><span>Workspace</span><b>/</b><strong>Stock analyzer</strong></div><h1>Stock analyzer</h1><p class="muted">Search a symbol to open its prediction and supporting market evidence.</p></div>
    </div>
    <section class="panel analyzer-welcome">
      <span class="analyzer-welcome-icon" aria-hidden="true">⌕</span>
      <p class="eyebrow">Research workspace</p>
      <h2>Analyze a stock</h2>
      <p>Search for a symbol above. NSE tickers such as HDFCBANK are normalized automatically.</p>
      <a class="text-link" href="/">Return to watchlist <span aria-hidden="true">→</span></a>
    </section>`;
  tickerInput.focus();
}

async function loadStock(ticker, period = "60d") {
  app.innerHTML = `<section class="loading-state panel"><span class="spinner" aria-hidden="true"></span><p>Loading ${escapeHtml(displayTicker(ticker))} analysis…</p></section>`;
  try {
    const snapshot = await getJson(`/api/stock/${encodeURIComponent(ticker)}?period=${encodeURIComponent(period)}`);
    renderStock(snapshot);
    document.title = `${displayTicker(snapshot.ticker)} · Stock Momentum`;
  } catch (error) {
    showError(error.message, `Analysis unavailable for ${displayTicker(ticker)}`);
  }
}

searchForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const ticker = normalizeTicker(tickerInput.value);
  if (!ticker || !/^[A-Z0-9^.-]{1,20}$/.test(ticker)) {
    tickerInput.setCustomValidity("Enter a valid stock symbol.");
    tickerInput.reportValidity();
    return;
  }
  tickerInput.setCustomValidity("");
  location.href = `/analyzer/${encodeURIComponent(displayTicker(ticker))}`;
});

setActiveNavigation();
setupMobileNavigation();
window.addEventListener("resize", () => {
  window.clearTimeout(marketChartResizeTimeout);
  marketChartResizeTimeout = window.setTimeout(() => {
    const marketSection = app.querySelector(".market-overview");
    if (!marketSection || !marketOverviewData) return;
    marketSection.outerHTML = renderMarketOverview(marketOverviewData);
    bindMarketQuoteButtons(marketOverviewData);
  }, 150);
});
const stockMatch = location.pathname.match(/^\/(?:analyzer|stock)\/([^/]+)\/?$/);
if (location.pathname === "/analyzer" || location.pathname === "/analyzer/") {
  renderAnalyzerHome();
} else if (stockMatch) {
  let routeTicker;
  try {
    routeTicker = decodeURIComponent(stockMatch[1]);
  } catch {
    routeTicker = stockMatch[1];
  }
  const ticker = normalizeTicker(routeTicker);
  const visibleTicker = displayTicker(ticker);
  tickerInput.value = visibleTicker;
  if (routeTicker !== visibleTicker) {
    const route = location.pathname.startsWith("/stock/") ? "stock" : "analyzer";
    history.replaceState(null, "", `/${route}/${encodeURIComponent(visibleTicker)}`);
  }
  loadStock(ticker);
} else {
  loadWatchlist();
}

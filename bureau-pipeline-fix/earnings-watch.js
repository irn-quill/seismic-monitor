var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });

// src/earnings-watch.js
var FIN_HUB_KEYS = ["fin_hub_key_1", "fin_hub_key_2", "fin_hub_key_3"];
var LOOKAHEAD_DAYS = 7;
var LOOKBACK_DAYS = 2;
async function getSecret(env, keyName) {
  try {
    return await env[keyName].get();
  } catch {
    return null;
  }
}
__name(getSecret, "getSecret");
function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(36);
}
__name(hashString, "hashString");
function dateStr(date) {
  return date.toISOString().split("T")[0];
}
__name(dateStr, "dateStr");
var finnhubKeyIndex = 0;
async function getFinnhubKey(env) {
  const key = FIN_HUB_KEYS[finnhubKeyIndex % FIN_HUB_KEYS.length];
  finnhubKeyIndex++;
  return await getSecret(env, key);
}
__name(getFinnhubKey, "getFinnhubKey");
async function getEarningsCalendar(env, fromDate, toDate) {
  const apiKey = await getFinnhubKey(env);
  if (!apiKey) throw new Error("No Finnhub key available");
  const url = `https://finnhub.io/api/v1/calendar/earnings?from=${fromDate}&to=${toDate}&token=${apiKey}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Finnhub calendar error: ${res.status}`);
  const data = await res.json();
  return data.earningsCalendar || [];
}
__name(getEarningsCalendar, "getEarningsCalendar");
function buildEarningsItem(company, entry) {
  const epsEstimate = entry.epsEstimate ?? null;
  const revenueEstimate = entry.revenueEstimate ?? null;
  const earningsDate = entry.date || null;
  const quarter = entry.quarter || null;
  const year = entry.year || (/* @__PURE__ */ new Date()).getFullYear();
  const title = `${company.name} (${company.ticker}) Earnings — Q${quarter} ${year}`;
  const description = [
    earningsDate ? `Earnings date: ${earningsDate}` : null,
    epsEstimate !== null ? `EPS estimate: $${epsEstimate}` : null,
    revenueEstimate !== null ? `Revenue estimate: $${(revenueEstimate / 1e9).toFixed(2)}B` : null
  ].filter(Boolean).join(" | ");
  return {
    title,
    url: `https://finnhub.io/stock-earnings?symbol=${company.ticker}`,
    description,
    source: "finnhub",
    source_type: "earnings",
    topic: company.beat,
    earnings_data: {
      ticker: company.ticker,
      company: company.name,
      earnings_date: earningsDate,
      quarter,
      year,
      eps_estimate: epsEstimate,
      revenue_estimate: revenueEstimate
    },
    ingested_at: (/* @__PURE__ */ new Date()).toISOString(),
    scraped_at: (/* @__PURE__ */ new Date()).toISOString(),
    status: "ready"
  };
}
__name(buildEarningsItem, "buildEarningsItem");
async function runEarningsWatch(env, logs) {
  const log = /* @__PURE__ */ __name((msg) => {
    logs.push(`${(/* @__PURE__ */ new Date()).toISOString()} - [earnings-watch] ${msg}`);
    console.log(`[earnings-watch] ${msg}`);
  }, "log");
  log("Run started");
  const registryRaw = await env.SOURCE_REGISTRY.get("registry:config");
  const registry = registryRaw ? JSON.parse(registryRaw) : [];
  const companies = registry.filter((s) => s.active && s.owning_worker === "earnings-watch").map((s) => ({
    ticker: s.finnhub_config?.ticker,
    name: s.finnhub_config?.company_name,
    beat: s.finnhub_config?.beat
  })).filter((c) => c.ticker);
  log(`Loaded ${companies.length} target companies from registry`);
  const now = /* @__PURE__ */ new Date();
  const fromDate = dateStr(new Date(now - LOOKBACK_DAYS * 864e5));
  const toDate = dateStr(new Date(now.getTime() + LOOKAHEAD_DAYS * 864e5));
  log(`Checking earnings calendar: ${fromDate} to ${toDate}`);
  let calendar;
  try {
    calendar = await getEarningsCalendar(env, fromDate, toDate);
    log(`Calendar entries returned: ${calendar.length}`);
  } catch (e) {
    log(`ERROR: Calendar fetch failed — ${e.message}`);
    return;
  }
  const tickers = new Set(companies.map((c) => c.ticker));
  const relevant = calendar.filter((e) => tickers.has(e.symbol));
  log(`Relevant target companies in window: ${relevant.length}`);
  if (relevant.length === 0) {
    log("No target company earnings in window — done");
    return;
  }
  let queued = 0;
  let skipped = 0;
  for (const entry of relevant) {
    const company = companies.find((c) => c.ticker === entry.symbol);
    if (!company) continue;
    const dedupeKey = `earnings:seen:${company.ticker}:${entry.date || entry.quarter || "unknown"}`;
    const already = await env.RSS_SCORE_STORE.get(dedupeKey);
    if (already) {
      log(`Skipping ${company.ticker} — already queued for this period`);
      skipped++;
      continue;
    }
    const item = buildEarningsItem(company, entry);
    const itemKey = `article:earnings_${company.ticker}_${hashString(entry.date || Date.now().toString())}`;
    await env.SCRAPE_S_S.put(itemKey, JSON.stringify(item), { expirationTtl: 7 * 86400 });
    await env.RSS_SCORE_STORE.put(dedupeKey, "1", { expirationTtl: 30 * 86400 });
    log(`Queued: ${item.title}`);
    queued++;
  }
  log(`Done: queued=${queued} skipped=${skipped}`);
  await env.RSS_SCORE_STORE.put("earnings:last_run", JSON.stringify({
    timestamp: (/* @__PURE__ */ new Date()).toISOString(),
    window: { from: fromDate, to: toDate },
    calendar_entries: calendar.length,
    relevant: relevant.length,
    queued,
    skipped
  }), { expirationTtl: 86400 });
}
__name(runEarningsWatch, "runEarningsWatch");
var earnings_watch_default = {
  async scheduled(event, env, ctx) {
    const logs = [];
    ctx.waitUntil(runEarningsWatch(env, logs));
  },
  async fetch(request, env, ctx) {
    const path = new URL(request.url).pathname;
    if (path === "/run") {
      // FIX: respond immediately, do work in background so pipeline conductor doesn't hang
      ctx.waitUntil(runEarningsWatch(env, []).catch(e => console.error(`[earnings-watch] run failed: ${e.message}`)));
      return new Response(JSON.stringify({ status: "started" }), { headers: { "Content-Type": "application/json" } });
    }
    if (path === "/status") {
      const last = await env.RSS_SCORE_STORE.get("earnings:last_run");
      return new Response(last || "{}", { headers: { "Content-Type": "application/json" } });
    }
    return new Response("earnings-watch — /run | /status", { status: 200 });
  }
};
export {
  earnings_watch_default as default
};

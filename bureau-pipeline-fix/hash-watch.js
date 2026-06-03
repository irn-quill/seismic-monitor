var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });

// src/hash-watch.js
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
function hashBody(text) {
  const normalised = text.replace(/\s+/g, " ").trim();
  return hashString(normalised);
}
__name(hashBody, "hashBody");
async function checkTarget(target, env, logs) {
  const log = /* @__PURE__ */ __name((msg) => {
    logs.push(`${(/* @__PURE__ */ new Date()).toISOString()} - [hash-watch] ${msg}`);
    console.log(`[hash-watch] ${msg}`);
  }, "log");
  const hashKey = `hw:hash:${target.id}`;
  const timeKey = `hw:checked:${target.id}`;
  const lastChecked = await env.RSS_SCORE_STORE.get(timeKey);
  if (lastChecked) {
    const elapsed = Date.now() - parseInt(lastChecked, 10);
    const intervalMs = target.interval_hours * 3600 * 1e3;
    if (elapsed < intervalMs) {
      log(`Skipping ${target.id} — checked ${Math.round(elapsed / 36e5)}h ago (interval: ${target.interval_hours}h)`);
      return { id: target.id, result: "skipped" };
    }
  }
  let body;
  try {
    const res = await fetch(target.url, {
      headers: {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml"
      },
      redirect: "follow"
    });
    if (!res.ok) {
      log(`Fetch failed for ${target.id}: HTTP ${res.status}`);
      return { id: target.id, result: "fetch_error", status: res.status };
    }
    body = await res.text();
  } catch (e) {
    log(`Fetch error for ${target.id}: ${e.message}`);
    return { id: target.id, result: "fetch_error", error: e.message };
  }
  await env.RSS_SCORE_STORE.put(timeKey, Date.now().toString(), { expirationTtl: 7 * 86400 });
  const newHash = hashBody(body);
  const oldHash = await env.RSS_SCORE_STORE.get(hashKey);
  if (oldHash === newHash) {
    log(`No change: ${target.id}`);
    return { id: target.id, result: "unchanged" };
  }
  log(`Change detected: ${target.id}`);
  await env.RSS_SCORE_STORE.put(hashKey, newHash, { expirationTtl: 30 * 86400 });
  if (!oldHash) {
    log(`First run baseline set for ${target.id} — no item queued`);
    return { id: target.id, result: "baseline_set" };
  }
  const itemKey = `item:hw_${target.id}_${Date.now()}`;
  const ttlJitter = 259200 + Math.floor(Math.random() * 21600);
  await env.RSS_SCORE_STORE.put(itemKey, JSON.stringify({
    title: `${target.title} — updated ${(/* @__PURE__ */ new Date()).toUTCString()}`,
    url: target.url,
    description: `Content change detected on ${target.title}. Full page will be scraped and scored.`,
    source: target.url,
    source_type: target.source_type,
    topic: target.topic,
    ingested_at: (/* @__PURE__ */ new Date()).toISOString(),
    status: "pending",
    hw_target_id: target.id
  }), { expirationTtl: ttlJitter });
  log(`Queued pending item: ${itemKey}`);
  return { id: target.id, result: "queued", itemKey };
}
__name(checkTarget, "checkTarget");
async function runHashWatch(env) {
  const logs = [];
  const results = [];
  logs.push(`${(/* @__PURE__ */ new Date()).toISOString()} - [hash-watch] Run started`);
  console.log("[hash-watch] Run started");
  const registryRaw = await env.SOURCE_REGISTRY.get("registry:config");
  const registry = registryRaw ? JSON.parse(registryRaw) : [];
  const targets = registry.filter((s) => s.active && s.owning_worker === "hash-watch").map((s) => ({
    id: s.id,
    url: s.url,
    title: s.name,
    source_type: s.source_type,
    topic: s.topic,
    interval_hours: s.hash_watch_config?.interval_hours || 24
  }));
  console.log(`[hash-watch] Loaded ${targets.length} targets from registry`);
  for (const target of targets) {
    const result = await checkTarget(target, env, logs);
    results.push(result);
  }
  const queued = results.filter((r) => r.result === "queued").length;
  const unchanged = results.filter((r) => r.result === "unchanged").length;
  const skipped = results.filter((r) => r.result === "skipped").length;
  const errors = results.filter((r) => r.result === "fetch_error").length;
  const baselines = results.filter((r) => r.result === "baseline_set").length;
  const summary = `queued=${queued} unchanged=${unchanged} skipped=${skipped} errors=${errors} baselines=${baselines}`;
  logs.push(`${(/* @__PURE__ */ new Date()).toISOString()} - [hash-watch] Done: ${summary}`);
  console.log(`[hash-watch] Done: ${summary}`);
  await env.RSS_SCORE_STORE.put("hw:last_run", JSON.stringify({
    timestamp: (/* @__PURE__ */ new Date()).toISOString(),
    results,
    summary: { queued, unchanged, skipped, errors, baselines }
  }), { expirationTtl: 86400 });
  return logs;
}
__name(runHashWatch, "runHashWatch");
var hash_watch_default = {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(runHashWatch(env));
  },
  async fetch(request, env, ctx) {
    const path = new URL(request.url).pathname;
    if (path === "/run") {
      // FIX: respond immediately, do work in background so pipeline conductor doesn't hang
      ctx.waitUntil(runHashWatch(env).catch(e => console.error(`[hash-watch] run failed: ${e.message}`)));
      return new Response(JSON.stringify({ status: "started" }), { headers: { "Content-Type": "application/json" } });
    }
    if (path === "/status") {
      const lastRun = await env.RSS_SCORE_STORE.get("hw:last_run");
      return new Response(lastRun || "{}", { headers: { "Content-Type": "application/json" } });
    }
    return new Response("hash-watch worker — /run | /status", { status: 200 });
  }
};
export {
  hash_watch_default as default
};

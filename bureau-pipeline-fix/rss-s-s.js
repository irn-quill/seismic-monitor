var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });

// src/rss-s-s.js
var THRESHOLDS = {
  primary: 10,
  editorial: 11,
  wire: 12,
  vendor: 13,
  googlenews: 14
};
var MAX_AGE_DAYS = 3;
var STORAGE_CAP_FULL = 50;
var STORAGE_CAP_META = 50;
var KEYWORDS = [
  "ai",
  "artificial intelligence",
  "funding",
  "acquisition",
  "merger",
  "regulation",
  "chip",
  "compute",
  "semiconductor",
  "llm",
  "model",
  "openai",
  "anthropic",
  "google deepmind",
  "nvidia",
  "venture capital",
  "vc",
  "earnings",
  "valuation",
  "startup",
  "enterprise",
  "deployment",
  "inference",
  "capex",
  "infrastructure",
  "policy",
  "enforcement"
];
var GEMINI_KEY_NAMES = ["gemini_key_1", "gemini_key_2", "gemini_key_3"];
var CB_TTL = 1800;
async function getKeyValue(env, keyName) {
  try {
    return await env[keyName].get();
  } catch {
    return null;
  }
}
__name(getKeyValue, "getKeyValue");
async function tripKey(env, keyName) {
  await env.RSS_SCORE_STORE.put(`cb:${keyName}`, "1", { expirationTtl: CB_TTL });
}
__name(tripKey, "tripKey");
async function isKeyTripped(env, keyName) {
  return !!await env.RSS_SCORE_STORE.get(`cb:${keyName}`);
}
__name(isKeyTripped, "isKeyTripped");
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
function normalizeTitle(title) {
  return (title || "").toLowerCase().replace(/[^a-z0-9\s]/g, "").replace(/\s+/g, " ").trim();
}
__name(normalizeTitle, "normalizeTitle");
function passesDateFilter(pubDate) {
  if (!pubDate) return false;
  const t = Date.parse(pubDate);
  if (isNaN(t)) return false;
  const ageMs = Date.now() - t;
  return ageMs >= 0 && ageMs <= MAX_AGE_DAYS * 86400 * 1e3;
}
__name(passesDateFilter, "passesDateFilter");
var NON_ENGLISH_MARKERS = [
  "de", "da", "do", "das", "dos", "em", "na", "no", "para", "por", "uma", "com",
  "der", "die", "und", "ein", "f\xFCr", "mit", "des", "dem", "als",
  "les", "des", "une", "sur", "pour", "dans", "qui", "que", "est",
  "los", "las", "del", "con", "por", "una", "este", "esta", "son"
];
function isLikelyEnglish(title, description) {
  const text = `${title} ${description}`;
  const letters = text.match(/\p{L}/gu) || [];
  const asciiLetters = text.match(/[a-zA-Z]/g) || [];
  if (letters.length === 0) return true;
  if (asciiLetters.length / letters.length <= 0.97) return false;
  const words = text.toLowerCase().split(/\s+/);
  const nonEnglishCount = words.filter((w) => NON_ENGLISH_MARKERS.includes(w)).length;
  return nonEnglishCount / words.length < 0.15;
}
__name(isLikelyEnglish, "isLikelyEnglish");
function passesPreFilter(title, description) {
  const text = `${title} ${description}`.toLowerCase();
  return KEYWORDS.some((kw) => text.includes(kw));
}
__name(passesPreFilter, "passesPreFilter");
function stripTags(html) {
  return (html || "").replace(/<!\[CDATA\[([\s\S]*?)\]\]>/gi, "$1").replace(/<[^>]+>/g, "").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'").trim();
}
__name(stripTags, "stripTags");
function parseFeed(xml) {
  const items = [];
  const atomMatches = xml.match(/<entry[\s\S]*?<\/entry>/gi) || [];
  if (atomMatches.length > 0) {
    for (const entry of atomMatches) {
      const title = (entry.match(/<title[^>]*>([\s\S]*?)<\/title>/i) || [])[1] || "";
      const link = (entry.match(/<link[^>]*href="([^"]*)"/) || [])[1] || (entry.match(/<link[^>]*>([\s\S]*?)<\/link>/i) || [])[1] || "";
      const summary = (entry.match(/<summary[^>]*>([\s\S]*?)<\/summary>/i) || [])[1] || (entry.match(/<content[^>]*>([\s\S]*?)<\/content>/i) || [])[1] || "";
      const pubDate = (entry.match(/<updated>([\s\S]*?)<\/updated>/i) || [])[1] || (entry.match(/<published>([\s\S]*?)<\/published>/i) || [])[1] || "";
      if (title || link) items.push({ title: stripTags(title).trim(), url: link.trim(), description: stripTags(summary).substring(0, 300).trim(), pubDate: pubDate.trim() });
    }
  } else {
    for (const item of xml.match(/<item[\s\S]*?<\/item>/gi) || []) {
      const title = (item.match(/<title[^>]*>([\s\S]*?)<\/title>/i) || [])[1] || "";
      const link = (item.match(/<link[^>]*>([\s\S]*?)<\/link>/i) || [])[1] || (item.match(/<guid[^>]*>([\s\S]*?)<\/guid>/i) || [])[1] || "";
      const description = (item.match(/<description[^>]*>([\s\S]*?)<\/description>/i) || [])[1] || "";
      const pubDate = (item.match(/<pubDate[^>]*>([\s\S]*?)<\/pubDate>/i) || [])[1] || "";
      if (title || link) items.push({ title: stripTags(title).trim(), url: link.trim(), description: stripTags(description).substring(0, 300).trim(), pubDate: pubDate.trim() });
    }
  }
  return items;
}
__name(parseFeed, "parseFeed");
async function scoreItemWithKey(apiKey, keyName, env, title, description, sourceType) {
  const vendorNote = sourceType === "vendor" ? " This is from a vendor blog — only score materiality high for concrete product launches, pricing changes, or enterprise announcements. Marketing copy scores 1-2 on materiality." : "";
  const prompt = `You are scoring articles for an AI business intelligence feed.

Return ONLY this JSON, no markdown:
{"ai_score": <1-10>, "materiality_score": <1-10>, "category": "<funding|markets|policy|infrastructure|other>"}

ai_score: How central is AI to this article? 10 = entirely about AI. 1 = AI barely mentioned.
materiality_score: How much does this matter to AI business?
  - Score 9-10: Major funding rounds ($100M+), acquisitions, earnings beats, significant regulation
  - Score 7-8: Meaningful funding, product launches with clear commercial impact, policy developments
  - Score 5-6: Moderate business relevance
  - Score 1-4: Stock price targets, analyst price targets, investment outlooks, speculative forecasts — cap at 4 regardless of AI angle
  - Score 1-2: Marketing, listicles, opinion without new information${vendorNote}

Infrastructure examples (score 7+): '40 inference optimization engineers hired', '10,000 H100 GPU procurement', 'new datacenter with $500M capex'
Policy examples (score 7+): 'EU AI Act amendment affecting inference compute', 'FTC investigation into AI market concentration'

Title: ${title}
Summary: ${description}`;
  try {
    const response = await fetch(
      "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
      {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-goog-api-key": apiKey },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          generationConfig: { maxOutputTokens: 200, temperature: 0.1, thinkingConfig: { thinkingBudget: 0 } }
        })
      }
    );
    if (response.status === 429 || response.status === 503) {
      await tripKey(env, keyName);
      console.warn(`[rss-s-s] Tripped ${keyName} (${response.status})`);
      return null;
    }
    if (!response.ok) {
      console.error(`[rss-s-s] Gemini error ${response.status} on ${keyName}`);
      return null;
    }
    const data = await response.json();
    const text = data?.candidates?.[0]?.content?.parts?.[0]?.text || "";
    const jsonMatch = text.match(/\{[\s\S]*?\}/);
    if (!jsonMatch) return null;
    const parsed = JSON.parse(jsonMatch[0]);
    const aiScore = Math.min(10, Math.max(1, parseInt(parsed.ai_score) || 0));
    const matScore = Math.min(10, Math.max(1, parseInt(parsed.materiality_score) || 0));
    return { ai_score: aiScore, materiality_score: matScore, score: aiScore + matScore, category: parsed.category || "other" };
  } catch (e) {
    console.error(`[rss-s-s] Scoring failed "${title}": ${e.message}`);
    return null;
  }
}
__name(scoreItemWithKey, "scoreItemWithKey");
async function scoreBatch(env, candidates, keyName) {
  const tripped = await isKeyTripped(env, keyName);
  if (tripped) {
    console.warn(`[rss-s-s] ${keyName} tripped, skipping batch`);
    return candidates.map(() => null);
  }
  const apiKey = await getKeyValue(env, keyName);
  if (!apiKey) {
    console.warn(`[rss-s-s] ${keyName} not available`);
    return candidates.map(() => null);
  }
  const results = [];
  for (const c of candidates) {
    results.push(await scoreItemWithKey(apiKey, keyName, env, c.item.title, c.item.description, c.feed.type));
  }
  return results;
}
__name(scoreBatch, "scoreBatch");
async function runIngestion(env) {
  const runStart = (/* @__PURE__ */ new Date()).toISOString();
  console.log(`[rss-s-s] Starting run at ${runStart}`);
  await env.RSS_SCORE_STORE.put("debug:last_run", JSON.stringify({
    timestamp: runStart,
    status: "running",
    fetched: 0,
    filtered_by_date: 0,
    filtered_by_language: 0,
    new_after_dedupe: 0,
    filtered_by_keyword: 0,
    sent_to_ai: 0,
    below_threshold: 0,
    stored: 0,
    dropped_at_cap: 0,
    feeds: []
  }, null, 2), { expirationTtl: 86400 });
  let totalFetched = 0, totalFilteredDate = 0, totalFilteredLang = 0, totalNew = 0;
  let totalScored = 0, totalStored = 0, totalBelowThreshold = 0, totalDroppedAtCap = 0;
  let currentFullCount = 0;
  let currentMetaCount = 0;
  const categoryDistribution = {};
  const scoreDistribution = { "1-5": 0, "6-10": 0, "11-15": 0, "16-20": 0 };
  const feedStats = [];
  const registryRaw = await env.SOURCE_REGISTRY.get("registry:config");
  const registry = registryRaw ? JSON.parse(registryRaw) : [];
  const feeds = registry.filter((s) => s.active && s.owning_worker === "rss-s-s").map((s) => ({
    url: s.url,
    type: s.scrape_config?.feed_type || "editorial",
    topic: s.topic
  }));
  console.log(`[rss-s-s] Loaded ${feeds.length} feeds from registry`);
  try {
    const seenUrls = /* @__PURE__ */ new Map();
    const seenTitles = /* @__PURE__ */ new Map();
    const feedFetches = await Promise.all(feeds.map(async (feed) => {
      const feedResult = { url: feed.url, type: feed.type, topic: feed.topic, status: null, items: 0, error: null };
      try {
        const response = await fetch(feed.url, {
          headers: {
            "User-Agent": "Mozilla/5.0 (compatible; Mad-Cat Bureau RSS Reader)",
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml"
          },
          cf: { cacheTtl: 300 }
        });
        feedResult.status = response.status;
        if (!response.ok) {
          feedResult.error = `http_${response.status}`;
          return { feedResult, items: [] };
        }
        const xml = await response.text();
        const allItems = parseFeed(xml);
        const freshItems = allItems.filter((item) => passesDateFilter(item.pubDate));
        const englishItems = freshItems.filter((item) => isLikelyEnglish(item.title, item.description));
        const items = englishItems.slice(0, 20);
        feedResult.items = items.length;
        return { feedResult, items, feed, filteredDate: allItems.length - freshItems.length, filteredLang: freshItems.length - englishItems.length };
      } catch (e) {
        feedResult.error = e.message;
        return { feedResult, items: [] };
      }
    }));
    for (const { feedResult, items, feed, filteredDate = 0, filteredLang = 0 } of feedFetches) {
      feedStats.push(feedResult);
      totalFilteredDate += filteredDate;
      totalFilteredLang += filteredLang;
      totalFetched += items?.length || 0;
      for (const item of items || []) {
        if (!item.url && !item.title) continue;
        const urlKey = item.url || item.title;
        const titleKey = normalizeTitle(item.title);
        if (!seenUrls.has(urlKey) && !seenTitles.has(titleKey)) {
          seenUrls.set(urlKey, { item, feed });
          if (item.title) seenTitles.set(titleKey, urlKey);
        }
      }
    }
    const candidates = [];
    const allEntries = [...seenUrls.values()];
    const dedupeKeys = allEntries.map(({ item }) => ({
      urlKey: item.url ? `seen:${hashString(item.url)}` : `seen:${hashString(item.title)}`,
      titleKey: `seentitle:${hashString(normalizeTitle(item.title))}`
    }));
    const chunkSize = 25;
    const seenResults = [];
    for (let i = 0; i < dedupeKeys.length; i += chunkSize) {
      const chunk = dedupeKeys.slice(i, i + chunkSize);
      const chunkResults = await Promise.all(
        chunk.flatMap(({ urlKey, titleKey }) => [
          env.RSS_SCORE_STORE.get(urlKey),
          env.RSS_SCORE_STORE.get(titleKey)
        ])
      );
      seenResults.push(...chunkResults);
    }
    for (let i = 0; i < allEntries.length; i++) {
      const { item, feed } = allEntries[i];
      const alreadySeenUrl = seenResults[i * 2];
      const alreadySeenTitle = seenResults[i * 2 + 1];
      const { urlKey, titleKey } = dedupeKeys[i];
      if (alreadySeenUrl || alreadySeenTitle) continue;
      totalNew++;
      if (!passesPreFilter(item.title, item.description)) {
        await Promise.all([
          env.RSS_SCORE_STORE.put(urlKey, "filtered", { expirationTtl: 604800 }),
          env.RSS_SCORE_STORE.put(titleKey, "filtered", { expirationTtl: 604800 })
        ]);
        continue;
      }
      await Promise.all([
        env.RSS_SCORE_STORE.put(urlKey, "seen", { expirationTtl: 604800 }),
        env.RSS_SCORE_STORE.put(titleKey, "seen", { expirationTtl: 604800 })
      ]);
      candidates.push({ item, feed });
    }
    totalScored = candidates.length;
    console.log(`[rss-s-s] ${candidates.length} candidates to score`);
    const batches = [[], [], []];
    candidates.forEach((c, i) => batches[i % 3].push(c));
    const [results0, results1, results2] = await Promise.all([
      scoreBatch(env, batches[0], "gemini_key_1"),
      scoreBatch(env, batches[1], "gemini_key_2"),
      scoreBatch(env, batches[2], "gemini_key_3")
    ]);
    const allResults = [];
    for (let i = 0; i < candidates.length; i++) {
      const batchIndex = i % 3;
      const posInBatch = Math.floor(i / 3);
      if (batchIndex === 0) allResults.push(results0[posInBatch]);
      else if (batchIndex === 1) allResults.push(results1[posInBatch]);
      else allResults.push(results2[posInBatch]);
    }
    for (let i = 0; i < candidates.length; i++) {
      const { item, feed } = candidates[i];
      const result = allResults[i];
      if (!result) continue;
      const combined = result.score;
      if (combined <= 5) scoreDistribution["1-5"]++;
      else if (combined <= 10) scoreDistribution["6-10"]++;
      else if (combined <= 15) scoreDistribution["11-15"]++;
      else scoreDistribution["16-20"]++;
      const threshold = THRESHOLDS[feed.type] || 14;
      if (combined < threshold) {
        totalBelowThreshold++;
        continue;
      }
      const cat = result.category || "other";
      const ttlJitter = 259200 + Math.floor(Math.random() * 21600);
      if (currentFullCount >= STORAGE_CAP_FULL) {
        if (currentMetaCount >= STORAGE_CAP_META) {
          totalDroppedAtCap++;
          continue;
        }
        await env.RSS_SCORE_STORE.put(`meta:${hashString(item.url || item.title)}`, JSON.stringify({
          title: item.title,
          url: item.url,
          score: combined,
          ai_score: result.ai_score,
          materiality_score: result.materiality_score,
          category: cat,
          source_type: feed.type,
          topic: feed.topic,
          ingested_at: (/* @__PURE__ */ new Date()).toISOString()
        }), { expirationTtl: ttlJitter });
        currentMetaCount++;
        continue;
      }
      await env.RSS_SCORE_STORE.put(`item:${hashString(item.url || item.title)}`, JSON.stringify({
        title: item.title,
        url: item.url,
        description: item.description,
        pubDate: item.pubDate,
        source: feed.url,
        source_type: feed.type,
        topic: feed.topic,
        ai_score: result.ai_score,
        materiality_score: result.materiality_score,
        score: combined,
        category: cat,
        ingested_at: (/* @__PURE__ */ new Date()).toISOString(),
        status: "pending"
      }), { expirationTtl: ttlJitter });
      totalStored++;
      currentFullCount++;
      categoryDistribution[cat] = (categoryDistribution[cat] || 0) + 1;
      console.log(`[rss-s-s] Stored (ai:${result.ai_score} mat:${result.materiality_score} total:${combined} ${cat} ${feed.topic}): ${item.title}`);
    }
  } finally {
    await env.RSS_SCORE_STORE.put("debug:last_run", JSON.stringify({
      timestamp: (/* @__PURE__ */ new Date()).toISOString(),
      status: "complete",
      fetched: totalFetched,
      filtered_by_date: totalFilteredDate,
      filtered_by_language: totalFilteredLang,
      new_after_dedupe: totalNew,
      filtered_by_keyword: totalNew - totalScored,
      sent_to_ai: totalScored,
      below_threshold: totalBelowThreshold,
      stored: totalStored,
      dropped_at_cap: totalDroppedAtCap,
      category_distribution: categoryDistribution,
      score_distribution: scoreDistribution,
      feeds: feedStats
    }, null, 2), { expirationTtl: 86400 });
    console.log(`[rss-s-s] Run complete: stored=${totalStored} scored=${totalScored} new=${totalNew}`);
  }
}
__name(runIngestion, "runIngestion");
var rss_s_s_default = {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(runIngestion(env));
  },
  async fetch(request, env, ctx) {
    const path = new URL(request.url).pathname;
    if (path === "/run") {
      // FIX: respond immediately, do work in background so pipeline conductor doesn't hang
      ctx.waitUntil(runIngestion(env).catch(e => console.error(`[rss-s-s] run failed: ${e.message}`)));
      return new Response(JSON.stringify({ status: "started" }), { headers: { "Content-Type": "application/json" } });
    }
    if (path === "/stats") {
      const stats = await env.RSS_SCORE_STORE.get("debug:last_run");
      return new Response(stats || "No stats yet — run /run first", {
        headers: { "Content-Type": "application/json" }
      });
    }
    if (path === "/debug") {
      const stats = await env.RSS_SCORE_STORE.get("debug:last_run");
      const { keys } = await env.RSS_SCORE_STORE.list({ prefix: "item:" });
      const items = [];
      for (const key of keys.slice(0, 20)) {
        const val = await env.RSS_SCORE_STORE.get(key.name);
        if (val) items.push(JSON.parse(val));
      }
      items.sort((a, b) => b.score - a.score);
      const html = `
        <h2>Last Run Stats</h2>
        <pre>${stats || "No stats yet"}</pre>
        <h2>Top Stored Items (sorted by score)</h2>
        ${items.length === 0 ? "<p>No items stored yet</p>" : items.map((i) => `
          <div style="border:1px solid #ccc;padding:8px;margin:8px 0">
            <strong>[${i.score}] ${i.category?.toUpperCase()}</strong> — ${i.title}<br>
            <small>ai:${i.ai_score} mat:${i.materiality_score} | ${i.source_type} | topic:${i.topic} | ${i.ingested_at}</small><br>
            <a href="${i.url}" target="_blank">${i.url}</a>
          </div>
        `).join("")}
        <h2>KV Counts</h2>
        <p>Full items: <strong>${keys.length}</strong></p>
      `;
      return new Response(html, { headers: { "Content-Type": "text/html" } });
    }
    if (path === "/circuit-breaker") {
      const results = [];
      for (const keyName of GEMINI_KEY_NAMES) {
        const tripped = await env.RSS_SCORE_STORE.get(`cb:${keyName}`);
        results.push(`${keyName}: ${tripped ? "TRIPPED" : "OK"}`);
      }
      return new Response(results.join("\n"), { headers: { "Content-Type": "text/plain" } });
    }
    if (path === "/reset-circuit-breaker") {
      for (const keyName of GEMINI_KEY_NAMES) {
        await env.RSS_SCORE_STORE.delete(`cb:${keyName}`);
      }
      return new Response("All circuit breakers reset", { status: 200 });
    }
    if (path === "/test") {
      const results = [];
      try {
        const response = await fetch("https://techcrunch.com/category/venture/feed/", {
          headers: { "User-Agent": "Mozilla/5.0", "Accept": "application/rss+xml, application/xml" }
        });
        results.push(`Feed status: ${response.status}`);
        const xml = await response.text();
        const items = parseFeed(xml);
        results.push(`Items parsed: ${items.length}`);
        if (items.length > 0) {
          const item = items[0];
          results.push(`First item: ${item.title}`);
          results.push(`Date filter: ${passesDateFilter(item.pubDate)}`);
          results.push(`Language filter: ${isLikelyEnglish(item.title, item.description)}`);
          results.push(`Pre-filter: ${passesPreFilter(item.title, item.description)}`);
          for (const keyName of GEMINI_KEY_NAMES) {
            const tripped = await isKeyTripped(env, keyName);
            const apiKey2 = await getKeyValue(env, keyName);
            results.push(`${keyName}: ${tripped ? "TRIPPED" : apiKey2 ? "OK" : "MISSING"}`);
          }
          const apiKey = await getKeyValue(env, "gemini_key_1");
          if (apiKey) {
            const scored = await scoreItemWithKey(apiKey, "gemini_key_1", env, item.title, item.description, "editorial");
            results.push(`Score: ${JSON.stringify(scored)}`);
          }
        }
      } catch (e) {
        results.push(`ERROR: ${e.message}`);
      }
      return new Response(results.join("\n\n"), { headers: { "Content-Type": "text/plain" } });
    }
    if (path === "/keys") {
      const { keys } = await env.RSS_SCORE_STORE.list({ limit: 100 });
      const rows = keys.map((k) => `<tr><td>${k.name}</td><td>${k.expiration ? new Date(k.expiration * 1e3).toISOString() : "no expiry"}</td></tr>`).join("");
      return new Response(`<table border="1"><tr><th>Key</th><th>Expires</th></tr>${rows}</table>`, { headers: { "Content-Type": "text/html" } });
    }
    return new Response("Mad-Cat Bureau rss-s-s | /run | /debug | /stats | /test | /keys | /circuit-breaker | /reset-circuit-breaker", { status: 200 });
  }
};
export {
  rss_s_s_default as default
};

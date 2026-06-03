# Bureau Pipeline Fix — How to Deploy

## What this fixes
The bureau pipeline was getting stuck because the discover workers (opinion-rss, rss-s-s, etc.)
were designed to run in the background on a timer, but the pipeline conductor was calling them
and waiting for them to finish — with no time limit. This fix makes each worker respond
immediately ("I've started") and do the actual work in the background. The pipeline
conductor can then move on without waiting.

---

## Workers that need updating (5 total)

### 1. hash-watch
**Method:** Full file replacement
1. Go to Cloudflare dashboard → Workers & Pages → hash-watch
2. Click **Quick Edit** (or the edit/pencil icon)
3. Select all the existing code (Ctrl+A) and delete it
4. Copy the entire contents of `hash-watch.js` from this folder and paste it in
5. Click **Save and Deploy**

### 2. earnings-watch
**Method:** Full file replacement
1. Go to Cloudflare dashboard → Workers & Pages → earnings-watch
2. Click **Quick Edit**
3. Select all (Ctrl+A), delete, paste contents of `earnings-watch.js` from this folder
4. Click **Save and Deploy**

### 3. rss-s-s
**Method:** Full file replacement
1. Go to Cloudflare dashboard → Workers & Pages → rss-s-s
2. Click **Quick Edit**
3. Select all (Ctrl+A), delete, paste contents of `rss-s-s.js` from this folder
4. Click **Save and Deploy**

### 4. opinion-rss
**Method:** Find and replace (file is too large to replace wholesale via the dashboard)
1. Go to Cloudflare dashboard → Workers & Pages → opinion-rss
2. Click **Quick Edit**
3. Press Ctrl+F to open find in the editor
4. Search for: `result = await runRss(env);`
5. You'll find the 10-line block starting with `if (path === "/run") {` — the one that has `await runRss(env)` inside it
6. Replace the ENTIRE block (from the `if (path === "/run") {` line through its closing `}`) with:
```
    if (path === "/run") {
      ctx.waitUntil(runRss(env).catch(e => console.error(`[rss] run failed: ${e.message}`)));
      return new Response(JSON.stringify({ status: "started" }), {
        headers: { "Content-Type": "application/json" }
      });
    }
```
7. Click **Save and Deploy**

See `opinion-rss-patch.js` in this folder for the full before/after if needed.

### 5. opinion-edgar
**Method:** Find and replace (same reason as above)
1. Go to Cloudflare dashboard → Workers & Pages → opinion-edgar
2. Click **Quick Edit**
3. Press Ctrl+F, search for: `result = await runEdgar(env);`
4. Replace the ENTIRE surrounding `if (path === "/run") {` block with:
```
    if (path === "/run") {
      ctx.waitUntil(runEdgar(env).catch(e => console.error(`[edgar] run failed: ${e.message}`)));
      return new Response(JSON.stringify({ status: "started" }), {
        headers: { "Content-Type": "application/json" }
      });
    }
```
5. Click **Save and Deploy**

See `opinion-edgar-patch.js` in this folder for the full before/after if needed.

---

## After deploying all 5

Once all 5 workers are updated, trigger the bureau pipeline again via the forge MCP
(`run_bureau_pipeline`). It should now complete and send your Telegram notification.

The workers will still do all their fetching, scoring, and anti-fingerprinting exactly
as before — they just won't make the pipeline wait for them to finish.

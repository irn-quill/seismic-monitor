// ============================================================
// PATCH for opinion-rss worker — /run handler only
//
// In the Cloudflare dashboard:
//   Workers & Pages > opinion-rss > Quick Edit
//   Use Ctrl+F to find the OLD block below, replace with NEW block
// ============================================================

// ── OLD (find this exact text and DELETE it) ──────────────────
/*
    if (path === "/run") {
      let result;
      try {
        result = await runRss(env);
      } catch (e) {
        result = { status: "error", error: e.message };
      }
      return new Response(JSON.stringify(result, null, 2), {
        headers: { "Content-Type": "application/json" }
      });
    }
*/

// ── NEW (replace the block above with this) ──────────────────
/*
    if (path === "/run") {
      ctx.waitUntil(runRss(env).catch(e => console.error(`[rss] run failed: ${e.message}`)));
      return new Response(JSON.stringify({ status: "started" }), {
        headers: { "Content-Type": "application/json" }
      });
    }
*/

// ============================================================
// That is the ONLY change needed in this file.
// All other code stays exactly as-is.
// ============================================================

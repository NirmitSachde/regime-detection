/* Site config — edit these after you have hosting set up.
 *
 * - API_BASE: your deployed FastAPI base URL (e.g. https://regime-detection-api.onrender.com)
 *   Leave empty to hide the API links.
 * - GITHUB_URL: your GitHub repo URL.
 *   Leave empty to hide GitHub links.
 *
 * No build step. Edit, refresh, deploy.
 */
window.SITE_CONFIG = {
  // Deployed FastAPI base URL. When empty, the dashboard pages render
  // baked-in sample data with a banner. Render free tier sleeps after
  // 15 min idle — first request after sleep can take ~50s. The dashboard
  // gracefully falls back to sample data on timeout, then auto-recovers
  // once the API wakes up.
  API_BASE:   "https://regime-detection-api.onrender.com",

  GITHUB_URL: "https://github.com/NirmitSachde/regime-detection",
};

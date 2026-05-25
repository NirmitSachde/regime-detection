# Deploy guide

Three artifacts to host:

1. **Static landing site** &mdash; `web/index.html` + assets &rarr; GitHub Pages
2. **Code reference** (pdoc) &mdash; generated from Python docstrings &rarr; GitHub Pages `/reference/` subpath
3. **REST API** &mdash; FastAPI at `src/regime/api/main.py` &rarr; Render (free tier)

Everything is free. No credit card required.

---

## 0. One-time GitHub setup

Push the repo. Do **not** add Claude or any AI assistant as a repo collaborator.

```bash
cd "regime detection"
git remote add origin git@github.com:<your-handle>/regime-detection.git
git push -u origin main
```

In **Settings &rarr; Pages &rarr; Build and deployment**, set **Source** to
**GitHub Actions**. That's it &mdash; the workflow in
`.github/workflows/pages.yml` handles the rest.

---

## 1. Landing site + code reference &rarr; GitHub Pages

This is automatic. On every push to `main`, the workflow:

1. Installs uv and Python 3.12.
2. Runs `pdoc` against `src/regime/` to build the code reference into `docs-site/`.
3. Copies `web/` into `_site/`.
4. Copies `docs-site/` into `_site/reference/`.
5. Uploads `_site/` as the Pages artifact and deploys.

After the first run the site is live at:

```
https://<your-handle>.github.io/regime-detection/
https://<your-handle>.github.io/regime-detection/reference/
```

**Build it locally first to preview:**

```bash
make docs           # builds ./docs-site/
make docs-serve     # builds + serves at http://localhost:8088
```

---

## 2. FastAPI &rarr; Render

Render's free tier hosts the API. The repo includes `render.yaml` and
`Dockerfile.api` &mdash; one-click deploy.

1. Go to <https://dashboard.render.com/blueprints>.
2. Click **New Blueprint Instance**.
3. Connect your GitHub account and pick this repo.
4. Render reads `render.yaml`, provisions a web service, builds with
   `Dockerfile.api`, and exposes it at `https://<service-name>.onrender.com`.

Once it's up:

```
https://<service-name>.onrender.com/health
https://<service-name>.onrender.com/regime/latest
https://<service-name>.onrender.com/docs       (interactive Swagger UI)
https://<service-name>.onrender.com/redoc       (ReDoc layout)
```

**Free-tier caveats:**

- Service sleeps after 15 min idle &mdash; first request after sleep costs ~50s
  cold start.
- 750 free hours/month per account.
- For "always warm" behaviour, point a free uptime monitor at `/health`
  (e.g. <https://uptimerobot.com>) every 10 min.

**Locking down CORS** &mdash; in `src/regime/api/main.py`, change
`allow_origins=["*"]` to your actual Pages origin:

```python
allow_origins=["https://<your-handle>.github.io"],
```

---

## 3. Wire the static site to your live URLs

Edit `web/config.js`:

```js
window.SITE_CONFIG = {
  API_BASE:   "https://<your-service>.onrender.com",
  GITHUB_URL: "https://github.com/<your-handle>/regime-detection",
};
```

Commit, push, and the GH Pages workflow re-deploys. The "API", "Reference",
and "GitHub" links in the nav + footer now point at your live services.

If `API_BASE` is empty, the API links hide automatically. Same for `GITHUB_URL`.

---

## Other hosts (if you don't want Render)

| Host | What you change |
|---|---|
| **Fly.io** | `fly launch --dockerfile Dockerfile.api`. Free tier is restricted now &mdash; check current limits. |
| **Railway** | New project &rarr; Deploy from repo &rarr; pick `Dockerfile.api`. ~$5/mo of free credit. |
| **Cloudflare Workers** | Doesn't support arbitrary Python &mdash; would require a rewrite to JS or Pyodide. Skip. |
| **AWS App Runner / Google Cloud Run** | Both work with `Dockerfile.api`. Pay-as-you-go, scales to zero. Free tier covers low traffic. |

Static landing + reference can also go to **Cloudflare Pages**, **Vercel**, or
**Netlify** &mdash; point them at the `_site/` artifact built by the same workflow
(or point them at `web/` and run `pdoc` in their build step).

---

## Local development

```bash
make api            # FastAPI with auto-reload, http://localhost:8000
make docs-serve     # pdoc reference at http://localhost:8088
cd web && python3 -m http.server 8000   # static site
```

The static site loads sample data baked into `web/sample_data.js`, so it works
without the API running. Once you set `API_BASE` in `config.js` and the API is
reachable, you can wire the demo charts to call live endpoints (see
`app.js` &mdash; left as a small follow-up).

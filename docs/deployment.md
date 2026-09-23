# Deployment on Vercel

The website is a static build of `web/checkers`. The policy network and the rules engine run in the visitor's browser, so there is no server process, database, or model host to operate.

## First deployment

1. Push the repository to GitHub.
2. In Vercel, choose **Add New → Project** and import the repository (or use the **Deploy with Vercel** button in the README).
3. Keep **Root Directory** at the repository root and leave the build settings on their defaults. [`vercel.json`](../vercel.json) supplies them:

   | Setting | Value |
   |---|---|
   | Framework | Vite |
   | Install command | `npm --prefix web/checkers ci` |
   | Build command | `npm --prefix web/checkers run build` |
   | Output directory | `web/checkers/dist` |

4. Deploy. Every push to the production branch redeploys, and pull requests get preview URLs.

No environment variables are required. Vercel exposes `VERCEL_PROJECT_PRODUCTION_URL` during the build, and `vite.config.ts` uses it to write absolute `og:image` URLs for link previews. Set `SITE_URL` (for example `https://checkers.example.com`) to override it.

[`.vercelignore`](../.vercelignore) keeps the Python training code, data, and reports out of the deployment upload; only `web/checkers` is needed to build.

## What gets served

```text
index.html                         entry page (revalidated on every request)
favicon.svg, apple-touch-icon.png  icons
og.png                             1200×630 social preview
assets/index-<hash>.js|css         React app
assets/policy.worker-<hash>.js     inference worker
assets/policy-<hash>.bin           1.88 MB float32 weights, fetched once by the worker
```

Vite content-hashes everything under `assets/`, and `vercel.json` serves that path with `Cache-Control: public, max-age=31536000, immutable`. A new model export therefore gets a new URL, and returning visitors download the weights only once per version.

## Security headers

Every response carries:

- `Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; worker-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'`
- `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, `Cross-Origin-Opener-Policy: same-origin`, and a restrictive `Permissions-Policy`.

The app loads no third-party scripts, fonts, or analytics. The worker also checks the weights' SHA-256 against the manifest before building the network, and it refuses weights whose layout, size, or values do not match.

`npm --prefix web/checkers run preview` serves the production build with the same headers (read from `vercel.json`), and the Playwright suite runs against that server. A CSP regression therefore fails the end-to-end tests before it reaches Vercel. Vercel's preview-deployment toolbar loads a third-party script, which the strict CSP may block on preview URLs; production is unaffected.

## Updating the model

1. Download the release bundle and run `scripts/export_browser_policy.py` (see the README).
2. Run `make check` and the web gate. Both test suites replay the regenerated parity fixture.
3. Commit `web/checkers/src/model/*` and `web/checkers/src/test/fixtures/parity.json`, then push. Vercel redeploys, and the new weights get a new hashed URL.

## Verify a deployment

```bash
curl -sI https://<your-domain>/ | grep -i content-security-policy
```

Then open the site and confirm that "Model ready" appears, that a game starts from both sides, and that the browser console is clean.

## Rollback

Use **Instant Rollback** on a previous production deployment in the Vercel dashboard, or revert the commit and push. Deployments are immutable, so a rollback restores the exact earlier build, weights included.

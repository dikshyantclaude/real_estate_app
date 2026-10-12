# Deploy District Atlas yourself

## Run locally

From the extracted district-atlas folder, run:

```bash
python run_local.py
```

Open http://localhost:8000 . The trained model is already included; you do not need to install Python ML dependencies just to use the app. Do not open index.html directly: data and model loading require HTTP.

## Host the application

Deploy the contents of `dist/` to any static web hosting service.

- Framework preset: Other / static site.
- Build command: none.
- Publish/output directory: dist.
- Entry point: index.html.
- Backend, API keys, environment variables: none.

Upload the entire dist directory, including data.json, model.json, report.json, homes.csv, vendor/, app.js, style.css and favicon.svg. Hash-based page navigation needs no custom routing rules. Use HTTPS. OpenStreetMap tiles require an internet connection; its attribution is included in the app.

## Retrain the model

```bash
python -m pip install -r requirements.txt
python scripts/train.py
```

Run from the project folder with Python 3.12 or compatible. This updates the data/model/report files in dist. Then redeploy dist. The training script can take a few minutes.

## Verify predictions and application logic

With Node.js installed:

```bash
node scripts/check-app.cjs
```

Full browser visual QA was unavailable during creation. Check all tabs, map tiles, a sample estimate, and a comparable search in your own browser before public release.

## Folder overview

- dist/: ready-to-host frontend, dataset, exported trained model, model metrics and Leaflet assets.
- scripts/: training and verification code.
- data/: original uploaded data, repaired data, land-area lookup, amenity provenance and inference fixtures.
- requirements.txt: Python training dependencies.
- README.md: methodology, limitations and project documentation.
- run_local.py: local launcher using only Python standard library.

This app is independent of ChatGPT hosting. No account integration or embedded hosting credentials are included. It estimates recorded sale prices and recommends historical comparable properties, not active listings.

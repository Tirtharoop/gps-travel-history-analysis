# Deploying to Streamlit Community Cloud

This project is ready to deploy as a public Streamlit app. The entry point is `app.py`, dependencies are in `requirements.txt`, and runtime settings are in `.streamlit/config.toml`.

## Deployment files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit entry point |
| `requirements.txt` | Runtime packages installed by Streamlit Cloud |
| `requirements-dev.txt` | Local testing and lint packages |
| `.streamlit/config.toml` | Theme, upload limit and privacy settings |
| `.gitignore` | Keeps caches, secrets and generated data out of Git |

No secrets or environment variables are required. Reverse geocoding is optional and uses OpenStreetMap Nominatim at runtime.

## 1. Push the project to GitHub

Create an empty GitHub repository, then run these commands from `C:\mini project v2`:

```powershell
git init
git add .
git commit -m "Prepare Streamlit deployment"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repository>.git
git push -u origin main
```

Do not commit `.streamlit/secrets.toml`, `.env`, uploaded Timeline files or `data/geocode_cache.json`.

## 2. Create the Streamlit Cloud app

1. Sign in at <https://share.streamlit.io> with GitHub.
2. Select **Create app**.
3. Choose the repository and the `main` branch.
4. Set **Main file path** to `app.py`.
5. In **Advanced settings**, select Python `3.12` or `3.13`.
6. Select **Deploy**.

Streamlit Cloud installs `requirements.txt` automatically. The first deployment usually takes a few minutes.

## 3. Verify the deployment

After the app starts:

1. Load the included demo dataset.
2. Open all five main tabs and confirm the maps and charts render.
3. Generate the PDF report and download it.
4. Upload a small Timeline JSON file and verify parsing.
5. Confirm the app URL opens in a private browser window.

## Updating the app

Push changes to the connected branch:

```powershell
git add .
git commit -m "Update dashboard"
git push
```

Streamlit Cloud redeploys automatically. Use the app menu in Streamlit Cloud to reboot manually if a dependency change requires a clean install.

## Configuration notes

- The upload limit is set to 200 MB in `.streamlit/config.toml`. Memory is still shared with the Streamlit Cloud container, so very large full-account exports may need to be split by month.
- The geocoding cache is written to `data/geocode_cache.json`. Streamlit Cloud storage is ephemeral, so the cache may reset after a reboot or redeploy.
- Reverse geocoding uses the public Nominatim service with rate limiting. Large batches can be slow or temporarily blocked.
- Usage telemetry is disabled with `browser.gatherUsageStats = false`.
- Detailed exception tracebacks are hidden in the browser with `client.showErrorDetails = "none"`.
- No database, background worker or persistent disk is required.

## Troubleshooting

### Dependency installation fails

Confirm `requirements.txt` is in the repository root and uses the runtime dependencies only. Development packages belong in `requirements-dev.txt`.

### App requires a newer Python version

Set Python `3.12` or `3.13` in Streamlit Cloud advanced settings. The code requires Python 3.11 or newer.

### Maps are blank

The app renders Folium maps through `st.iframe`. Check the browser console and confirm the browser allows embedded HTML content.

### Geocoding fails

This is expected when Nominatim is unavailable, rate limited or the app has no outbound network access. The rest of the dashboard continues to work without addresses.

### App exceeds memory limits

Upload a smaller export or split the Timeline file by month. The application samples points for heatmaps and base maps, but parsing still requires the uploaded JSON to fit in memory.

# Full Project Report — Plant Disease App

Generated: 2026-06-01T07:12:00Z

## Summary
This report summarizes the repository at `d:\ai data\Final` (workspace root). It documents the tools and dependencies used, how the frontend connects to the backend, API endpoints, model and heuristics locations, data and evaluation artifacts, and recommended next steps.

## Tools & Dependencies
- Backend Python (Flask) dependencies (see [plant-disease-backend/requirements.txt](plant-disease-backend/requirements.txt)):
  - Flask==3.1.3, Flask-CORS, numpy, opencv-python-headless, scikit-learn, Pillow, joblib, requests, waitress, python-dotenv
- Frontend (web) dependencies (see [web/package.json](web/package.json)):
  - React, react-dom, react-router-dom, Vite, @vitejs/plugin-react
- Other notable scripts/tools:
  - `scripts/generate_misprediction_report.py` — batch evaluation and CSV/JSON/MD output
  - `debug_predict.py` — single-image debug runner
  - `evaluate_sl_crops.py` (backend) — dataset eval helper

## Backend structure and key files
- Main app: [plant-disease-backend/app.py](plant-disease-backend/app.py)
  - Serves API endpoints and the built frontend (`web/dist`) when present.
  - Persists local users into `local_users.json` and history into `prediction_history.json`.
- Models and config: [plant-disease-backend/config.py](plant-disease-backend/config.py)
  - Model artifact paths: `plant-disease-backend/models/model.pkl`, `scaler.pkl`, `label_encoder.pkl` (resolved by `config.py`).
- Prediction pipeline: [plant-disease-backend/predict.py](plant-disease-backend/predict.py)
  - Loads model artifacts and performs feature extraction.
- Heuristics and analysis: [plant-disease-backend/leaf_analysis.py](plant-disease-backend/leaf_analysis.py)
  - Detection helpers, `format_display_name()`, and re-ranking heuristics (e.g., mango/apple/banana rules).
- Local history store: [plant-disease-backend/history_store.py](plant-disease-backend/history_store.py)
  - `prediction_history.json` holds saved scans per user.

## Frontend structure and connectivity
- Frontend source: `web/src/` (React + Vite). Key files:
  - API client: [web/src/api/client.js](web/src/api/client.js) — builds base URL and `apiRequest()` wrapper used across the app.
  - Predict API wrapper: [web/src/api/predict.js](web/src/api/predict.js) — functions `predictDisease()` and `uploadAndPredict()`.
  - Result and history pages: `web/src/pages/ResultPage.jsx`, `HistoryPage.jsx`, `DashboardPage.jsx`.
- How frontend connects to backend:
  - In production the Flask server serves the built UI and the frontend uses same-origin API paths like `/predict`.
  - In development Vite proxies `/api` to the backend; set `VITE_API_BASE_URL` to override (e.g., `http://<host>:5000`). See `getApiBase()` in [web/src/api/client.js](web/src/api/client.js).
  - Upload flow: `uploadAndPredict()` posts a `FormData` with `image` to `/upload_predict`, which returns a normalized object; `predictDisease()` posts to `/predict`.

## Public API endpoints (backend)
- `GET /health` — health/status and rules info.
- `POST /login` — (stub) returns token and creates in-memory session.
- `POST /register` — creates local user (writes `local_users.json`) and returns token.
- `POST /upload_predict` — multipart upload: saves image, optionally adds to dataset, runs prediction, returns `prediction`, `display_info`, `top_predictions`, `saved_copy`, etc.
- `POST /predict` — multipart upload: returns normalized `disease`, `display_name`, `confidence`, `all_predictions`, `treatment`, and `clarification_questions` when confidence low.
- `GET|POST /history` — list or append prediction history.
- `POST /clarify` — get clarification questions between two diseases.
- `POST /answer` — submit clarifying answers; returns resolved disease.
- `GET /treatment/<disease>` — fetch treatment info from `qa_engine`.
- `GET /diseases` — list known diseases.
- Static: serves `web/dist` (built frontend) for other paths.

## Models, outputs, and evaluation artifacts
- Model artifacts: `plant-disease-backend/models/` (pickled SKLearn artifacts referenced by `config.py`).
- Saved uploads: `plant-disease-backend/uploads/` (incoming images saved by `upload_predict`).
- Eval output and mispredictions: `plant-disease-backend/eval_output/`
  - Debug runner output: `debug_*.json` and `debug_*.jpg` created by `debug_predict.py`.
  - Misprediction report (sample run): [plant-disease-backend/eval_output/misprediction_report_20260601T070538Z.md](plant-disease-backend/eval_output/misprediction_report_20260601T070538Z.md)
  - Full report generated: `plant-disease-backend/eval_output/misprediction_report_*.csv|.jsonl|.md` (CSV/JSONL/MD available).

## Findings from the sample misprediction report
- Sample (100 images) shows `Mango_Anthracnose` predicted 60/100 times.
- Frequent mislabels: many Apple and Banana images are predicted as `Mango_Anthracnose` — this indicates a systemic bias where mango-like lesion features dominate the scoring.
- A targeted heuristic was added earlier for Apple vs Blueberry; more conservative heuristics or retraining may be needed for Apple→Mango cases.

## How labels/display_name are passed to the UI
- Backend returns `display_info` (dict) and endpoints `predict` & `upload_predict` include `display_info.display_name`.
- Frontend normalizes responses: `upload_predict` response is mapped to `disease`, `display_name`, `all_predictions` in [web/src/api/predict.js](web/src/api/predict.js).
- UI components display `display_name` (or format the disease id if missing) — see `PredictionCard.jsx` and `ResultPage.jsx`.

## How to run locally (quick commands)
- Create and activate Python venv, install backend deps:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r plant-disease-backend\requirements.txt
```

- Start backend (development):

```powershell
cd plant-disease-backend
python app.py   # or run via waitress in production
```

- Start frontend (dev mode):

```bash
cd web
npm install
npm run dev
# Vite dev server proxies /api to backend; set VITE_API_BASE_URL to backend if needed
```

- Build frontend for production and serve via Flask:

```bash
cd web
npm run build
# then start backend; Flask will serve web/dist
```

- Run debug prediction for a single image:

```powershell
cd plant-disease-backend
python debug_predict.py <path-to-image>
```

- Generate a full misprediction report (no limit):

```powershell
cd plant-disease-backend
python scripts\generate_misprediction_report.py --root eval_output\mispredictions --outdir eval_output --limit 0
```

## Recommendations / Next steps
- Run the full dataset misprediction report (limit=0) to collect all mislabels and prioritize top pairs (e.g., Apple→Mango). The generated CSV/JSONL will help target heuristics or curate training data.
- Expand logging in `predict.py` to store metrics with each saved prediction (already added for `debug_predict` and `/debug_predict`).
- Add unit tests for heuristics in `leaf_analysis.py` to prevent regressions when tuning rules.
- Consider further model retraining with balanced examples for Apple and Banana classes or training a small CNN if features are insufficient.
- Persist sessions/tokens to a simple DB (SQLite) if needed for multi-user testing.

## Artifacts produced in this run
- `plant-disease-backend/eval_output/misprediction_report_20260601T070538Z.csv`
- `plant-disease-backend/eval_output/misprediction_report_20260601T070538Z.jsonl`
- `plant-disease-backend/eval_output/misprediction_report_20260601T070538Z.md`
- `plant-disease-backend/eval_output/full_project_report_20260601T071200Z.md` (this file)

---
If you want, I can now:
- Run the full (no-limit) misprediction report and attach the CSV for download.
- Generate an HTML report with sample images inline for the top 20 mislabels.
- Start tuning heuristics for the top mislabel pairs (e.g., Apple→Mango) and run targeted tests.

Which option should I do next?
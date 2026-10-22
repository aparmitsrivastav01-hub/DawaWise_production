# DawaWise (local full-stack)

Local development layout for the existing React frontend and FastAPI backend. Medicine search is intended to use live retrieval rather than a local catalogue.

## Prerequisites

- Node.js 18+ (npm is used with `frontend/package-lock.json`)
- Python 3.10+ (the backend uses `str | None` type hints)
- Git (already used to clone the source repositories)

Optional:

- A Google Maps JavaScript API key, only for live Jan Aushadhi Kendra map search in the Nearby Stores section

## Folder structure

```text
dawa-wise/
├── frontend/       # https://github.com/Shreya04042006/Dawa-Wise.git
├── backend/        # https://github.com/aparmitsrivastav01-hub/backend.git
├── README.md
└── .gitignore
```

`frontend/` and `backend/` keep their own `.git` remotes. Do not run `git push` from the parent folder expecting it to update those remotes.

## Install

### Frontend

```powershell
cd frontend
npm ci
copy .env.example .env
```

Edit `frontend/.env` if the backend is not on port 8000.

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

## Environment variables

Use placeholders only. Never put private keys in frontend variables except keys that are designed to be public (Google Maps browser key, restricted by HTTP referrer).

### `frontend/.env`

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_GOOGLE_MAPS_API_KEY=YOUR_GOOGLE_MAPS_API_KEY_HERE
```

### `backend/.env`

```env
OPENROUTER_API_KEY=
TINYFISH_API_KEY=
# CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Both API keys stay on the server. The frontend never receives them. Live search currently fails closed until OpenRouter/NVIDIA Nemotron query planning is implemented and both keys are configured.

## Run locally (two terminals)

**Terminal 1 — backend** (http://127.0.0.1:8000)

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — frontend** (http://localhost:5173)

```powershell
cd frontend
npm run dev
```

Open http://localhost:5173

API docs: http://127.0.0.1:8000/docs

## API routes

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Process health |
| `POST` | `/api/search` | JSON body `{ "query": "..." }`. Live search endpoint; requires OpenRouter/NVIDIA Nemotron planning and TinyFish Search/Fetch |

`POST /api/search` statuses:

- `matched` — a live result with retrieved product details
- `not_found` — no trustworthy live source found
- `error` — configuration, integration, or retrieval failure

There is **no** prescription upload, OCR, or Gemma endpoint in this backend.

## Frontend-to-backend mapping

| UI action | Client | Backend |
| --- | --- | --- |
| Medicine search (hero form, popular chips) | `POST` via `frontend/src/api/client.js` → `searchMedicines()` | `POST /api/search` |
| Health (manual / tests) | `getHealth()` | `GET /api/health` |
| Upload prescription | Local file picker only | Not implemented |
| Nearby stores | Google Maps in the browser, or the built-in reference directory | Not a backend route |

## Known limitations

- Prescription scanning is incomplete. Selecting a file does not upload it and does not run OCR.
- OpenRouter/NVIDIA Nemotron query planning is not implemented in the active backend yet. Search returns an integration error until it is implemented and both backend keys are configured.
- Gemma prescription understanding is described in the frontend README but is not present in the backend code.
- Nearby Stores live maps need `VITE_GOOGLE_MAPS_API_KEY`. Without it, the page uses a small reference directory and Google Maps links.
- The help/contact form is still a client-side demonstration and does not send email.
- The language switcher changes the label only; page copy stays English.

## Troubleshooting

- **Search fails with a network error:** start the backend first, then confirm `VITE_API_BASE_URL` matches the uvicorn host/port. Restart `npm run dev` after changing `.env`.
- **CORS error:** the backend allows `http://localhost:5173` and `http://127.0.0.1:5173` by default. Add other origins with `CORS_ORIGINS`.
- **Live search is not ready:** configure `OPENROUTER_API_KEY` and `TINYFISH_API_KEY` in `backend/.env`. Search remains disabled until the Nemotron query-planning step is implemented.
- **Execution policy on Windows:** if `Activate.ps1` is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` or use `backend\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000`.

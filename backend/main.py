import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from event_log import log_event
from nemotron_planner import NemotronUnavailableError, plan_search
from tinyfish_search import TinyFishAuthenticationError, TinyFishUnavailableError, search_medicine

load_dotenv(Path(__file__).with_name(".env"), override=False)

app = FastAPI(title="DawaWise API")

DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
)


def get_cors_origins(value: str | None = None) -> list[str]:
    """Use exact configured browser origins; never enable wildcard CORS."""
    raw_value = os.getenv("CORS_ORIGINS", "") if value is None else value
    if not raw_value.strip():
        return list(DEFAULT_CORS_ORIGINS)

    origins = []
    for item in raw_value.split(","):
        origin = item.strip().rstrip("/")
        if not origin:
            continue
        parsed = urlsplit(origin)
        if origin == "*" or parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path or parsed.query or parsed.fragment:
            raise ValueError(f"Invalid CORS origin {item!r}; use exact http(s) origins separated by commas.")
        if origin not in origins:
            origins.append(origin)
    if not origins:
        raise ValueError("CORS_ORIGINS must contain at least one exact origin when configured.")
    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=240)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/search")
async def search(payload: SearchRequest):
    query = payload.query.strip()
    if not query:
        return {"status": "error", "message": "Enter a medicine name to search."}
    log_event("search_received", query_length=len(query))
    missing_keys = [
        name for name in ("OPENROUTER_API_KEY", "TINYFISH_API_KEY")
        if not os.getenv(name, "").strip()
    ]
    if missing_keys:
        return {
            "status": "error",
            "message": (
                "Live search needs OpenRouter-hosted NVIDIA Nemotron query planning and TinyFish retrieval. "
                f"Configure these backend/.env variables: {', '.join(missing_keys)}."
            ),
        }
    try:
        plan = await plan_search(query, os.environ["OPENROUTER_API_KEY"].strip())
    except NemotronUnavailableError as exc:
        return {"status": "error", "message": str(exc)}
    if plan.needs_clarification:
        log_event("search_completed", status="clarification_required")
        return {
            "status": "clarification_required",
            "question": plan.clarification_question or "Please clarify which medicine you mean.",
            "candidates": [],
        }
    try:
        result = await search_medicine(
            query,
            os.environ["TINYFISH_API_KEY"].strip(),
            search_queries=plan.search_queries,
            generic_name=plan.generic_name,
            strength=plan.strength,
            dosage_form=plan.dosage_form,
            brand_name=plan.brand_name,
        )
    except TinyFishAuthenticationError:
        return {"status": "error", "message": "TinyFish authentication failed. Check TINYFISH_API_KEY in backend/.env."}
    except TinyFishUnavailableError:
        return {"status": "error", "message": "TinyFish live web retrieval is temporarily unavailable. Please retry."}
    log_event("search_completed", status="success", result_status=result.get("status", "unknown"))
    return result

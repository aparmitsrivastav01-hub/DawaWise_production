"""OpenRouter-hosted Nemotron query planning. Never treats model output as price evidence."""

from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from event_log import log_event

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "nvidia/nemotron-3-ultra-550b-a55b:free"
TIMEOUT = httpx.Timeout(25.0, connect=8.0)


class SearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    original_query: str = Field(min_length=1, max_length=240)
    brand_name: str | None = Field(default=None, max_length=160)
    generic_name: str | None = Field(default=None, max_length=160)
    strength: str | None = Field(default=None, max_length=100)
    dosage_form: str | None = Field(default=None, max_length=100)
    search_queries: list[str] = Field(default_factory=list, max_length=2)
    needs_clarification: bool
    clarification_question: str | None = Field(default=None, max_length=400)

    @model_validator(mode="before")
    @classmethod
    def normalize_blank_optional_fields(cls, value: Any) -> Any:
        # Nemotron sometimes encodes an unknown optional value as "" instead of null.
        # Treat both representations identically so an otherwise valid clarification
        # plan can reach the user instead of failing the whole search.
        if isinstance(value, dict):
            value = value.copy()
            for field in ("brand_name", "generic_name", "strength", "dosage_form", "clarification_question"):
                if value.get(field) == "":
                    value[field] = None
            queries = value.get("search_queries")
            if isinstance(queries, list):
                value["search_queries"] = [
                    query for query in queries
                    if not (isinstance(query, str) and not query.strip())
                ]
        return value

    @model_validator(mode="after")
    def require_search_target(self) -> "SearchPlan":
        if self.needs_clarification:
            if not self.clarification_question:
                raise ValueError("clarification_question is required when needs_clarification is true")
            return self
        if not (self.brand_name or self.generic_name):
            raise ValueError("brand_name or generic_name is required when needs_clarification is false")
        if not self.search_queries:
            raise ValueError("at least one non-empty search query is required")
        if any(len(query) > 300 for query in self.search_queries):
            raise ValueError("search queries must be concise (300 characters or fewer)")
        return self


class NemotronUnavailableError(Exception):
    """Safe-to-report failure from OpenRouter or its response."""


PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "original_query": {"type": "string"},
        "brand_name": {"type": ["string", "null"]},
        "generic_name": {"type": ["string", "null"]},
        "strength": {"type": ["string", "null"]},
        "dosage_form": {"type": ["string", "null"]},
        "search_queries": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 2},
        "needs_clarification": {"type": "boolean"},
        "clarification_question": {"type": ["string", "null"]},
    },
    "required": ["original_query", "brand_name", "generic_name", "strength", "dosage_form", "needs_clarification", "clarification_question"],
}


def make_openrouter_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=TIMEOUT)


async def plan_search(query: str, api_key: str) -> SearchPlan:
    log_event("nemotron_request", model=MODEL_ID, status="started")
    messages = [
        {
            "role": "system",
            "content": (
                "Interpret a medicine search and produce a web-search plan only. Do not provide prices, "
                "medical advice, or unsupported product facts. Preserve the user's original query exactly. "
                "Use null for brand name, generic name, strength, or dosage form unless identifiable from "
                "the query itself. Do not ask which brand merely because the user supplied a clear generic "
                "name and strength: a generic name plus strength is a safe search target even when form is "
                "unspecified. Do not ask for a dosage form when live search can retrieve information across "
                "forms; validation will reject non-comparable products. Ask a concise clarification question "
                "only when the medicine identity cannot be searched safely (for example, the name could refer "
                "to different active ingredients or combinations). For an unbranded generic query, create at "
                "least one live web-search query using the known generic name and any known strength/form. "
                "For a brand query, return up to two queries: one for the branded product's published MRP and "
                "product details from manufacturer/trustworthy sources, and one for the same active ingredient, "
                "strength and form's generic equivalent price from Jan Aushadhi/PMBI. Do not claim results are verified."
            ),
        },
        {"role": "user", "content": query},
    ]
    request = {
        "model": MODEL_ID,
        "temperature": 0,
        "max_tokens": 500,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "medicine_search_plan", "strict": True, "schema": PLAN_SCHEMA},
        },
    }
    try:
        async with make_openrouter_client() as client:
            response = await client.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=request,
            )
        if response.status_code == 401:
            log_event("nemotron_result", status="authentication_error")
            raise NemotronUnavailableError("OpenRouter authentication failed. Check OPENROUTER_API_KEY in backend/.env.")
        if response.status_code == 429:
            log_event("nemotron_result", status="rate_limited")
            raise NemotronUnavailableError("Nemotron is rate-limited. Please try again shortly.")
        if response.is_error:
            log_event("nemotron_result", status="upstream_error", http_status=response.status_code)
            raise NemotronUnavailableError("Nemotron query planning is temporarily unavailable. Please retry.")
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Expected JSON text")
        plan = SearchPlan.model_validate_json(content)
        if plan.original_query != query:
            raise ValueError("Model changed the original query")
    except NemotronUnavailableError:
        raise
    except httpx.TimeoutException:
        log_event("nemotron_result", status="timeout")
        raise NemotronUnavailableError("Nemotron query planning timed out. Please retry.") from None
    except httpx.RequestError:
        log_event("nemotron_result", status="upstream_error")
        raise NemotronUnavailableError("Nemotron query planning is temporarily unavailable. Please retry.") from None
    except ValidationError as exc:
        validation_errors = [
            {
                "field": ".".join(str(part) for part in error["loc"]) or "plan",
                "code": error["type"],
                "message": error["msg"],
            }
            for error in exc.errors(include_input=False)[:5]
        ]
        log_event("nemotron_result", status="invalid_response", validation_errors=validation_errors)
        summary = "; ".join(f"{error['field']}: {error['message']}" for error in validation_errors)
        raise NemotronUnavailableError(
            f"Nemotron returned an invalid search plan ({summary}). Please retry."
        ) from None
    except (ValueError, TypeError, KeyError, IndexError):
        log_event("nemotron_result", status="invalid_response")
        raise NemotronUnavailableError("Nemotron returned an invalid search plan. Please retry.") from None

    log_event(
        "nemotron_result",
        status="success",
        needs_clarification=plan.needs_clarification,
        query_count=len(plan.search_queries),
    )
    return plan

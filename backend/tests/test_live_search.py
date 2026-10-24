import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import main
import event_log
import nemotron_planner
import tinyfish_search


def _mock_client(monkeypatch, handler):
    monkeypatch.setattr(
        tinyfish_search,
        "make_tinyfish_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(
        tinyfish_search,
        "make_pmbi_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(503))),
    )


def test_cors_origins_accept_exact_https_origins():
    assert main.get_cors_origins(
        "https://dawawise.vercel.app, https://dawawise-preview.vercel.app/"
    ) == ["https://dawawise.vercel.app", "https://dawawise-preview.vercel.app"]


def test_cors_origins_reject_wildcards_and_paths():
    with pytest.raises(ValueError):
        main.get_cors_origins("*")
    with pytest.raises(ValueError):
        main.get_cors_origins("https://dawawise.vercel.app/path")


def _result(url, title):
    return {"url": url, "title": title, "snippet": title, "site_name": url.split("/")[2]}


def _pmbi_table(rows):
    rendered = "".join(
        "<tr><th>{}</th><th>{}</th><th>{}</th><th>{}</th><th>{}</th></tr>".format(*row)
        for row in rows
    )
    return (
        '<table><tr><td>Sr. No.</td><td>Drug Code</td><td>Generic Name</td>'
        f'<td>Unit Size</td><td>MRP (in Rs.)</td></tr>{rendered}</table>'
    )


def _mock_pmbi_product_list(monkeypatch, table_html):
    def handler(request):
        assert request.url == tinyfish_search.PMBI_PRODUCT_LIST_URL
        if request.method == "GET":
            return httpx.Response(200, text='<form><input type="hidden" name="__VIEWSTATE" value="view" /></form>')
        body = request.read().decode()
        assert "ctl00%24Bppi_body%24txtSearch=Paracetamol" in body
        return httpx.Response(200, text=table_html)

    monkeypatch.setattr(
        tinyfish_search,
        "make_pmbi_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _mock_official_search_fetch(monkeypatch, brand_page=None, jan_page=None, capture=None):
    brand_url = brand_page["url"] if brand_page else None
    jan_url = jan_page["url"] if jan_page else "https://www.pmbi.co.in/Data/eBrochure/mock.pdf"

    def handler(request):
        if request.url.host == "api.search.tinyfish.ai":
            if request.url.params.get("include_domains"):
                if capture is not None:
                    capture["jan_query"] = request.url.params["query"]
                    capture["jan_domains"] = request.url.params["include_domains"]
                results = [_result(jan_url, "Official PMBI product catalogue PDF")]
            else:
                results = [_result(brand_url, brand_page["title"])] if brand_page else []
            return httpx.Response(200, json={"results": results})
        if request.url.host == "api.fetch.tinyfish.ai":
            urls = __import__("json").loads(request.read())["urls"]
            results = []
            if brand_page and brand_url in urls:
                results.append({"url": brand_url, **brand_page["content"]})
            if jan_url in urls:
                results.append({"url": jan_url, **(jan_page["content"] if jan_page else {"text": None})})
            return httpx.Response(200, json={"results": results})
        raise AssertionError(f"Unexpected TinyFish request host: {request.url.host}")

    _mock_client(monkeypatch, handler)
    return jan_url


def test_health_and_missing_key_are_safe(monkeypatch):
    monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = TestClient(main.app)
    assert client.get("/api/health").json() == {"status": "ok"}
    result = client.post("/api/search", json={"query": "example tablet"}).json()
    assert result["status"] == "error"
    assert "OPENROUTER_API_KEY" in result["message"]
    assert "TINYFISH_API_KEY" in result["message"]


def test_live_sources_match_all_comparison_fields(monkeypatch):
    monkeypatch.setenv("TINYFISH_API_KEY", "mock-key")
    brand_url = "https://www.cipla.com/medicines/sample-500"
    jan_url = "https://janaushadhi.gov.in/product/sample-500"

    def handler(request):
        if request.url.host == "api.search.tinyfish.ai":
            if "Jan Aushadhi" in request.url.params["query"]:
                return httpx.Response(200, json={"results": [_result(jan_url, "Sample 500 mg tablets 10 tablets")]})
            return httpx.Response(200, json={"results": [_result(brand_url, "Sample 500 mg tablets 10 tablets")]})
        if request.url.host == "api.fetch.tinyfish.ai":
            assert request.read()
            return httpx.Response(200, json={"results": [
                {"url": brand_url, "final_url": brand_url, "title": "Sample 500 mg tablets 10 tablets", "published_date": "2026-08-01", "text": "Active ingredient: paracetamol\nStrength: 500 mg\nDosage form: tablets\nPack size: 10 tablets\nMRP: ₹30.00"},
                {"url": jan_url, "final_url": jan_url, "title": "Sample 500 mg tablets 10 tablets", "published_date": "2026-07-15", "text": "Active ingredient: paracetamol\nStrength: 500 mg\nDosage form: tablets\nPack size: 10 tablets\nPrice: ₹12.00"},
            ], "errors": []})
        raise AssertionError(f"Unexpected TinyFish request host: {request.url.host}")

    _mock_client(monkeypatch, handler)
    payload = asyncio.run(tinyfish_search.search_medicine("Sample 500 mg tablets", "mock-key"))
    medicine = payload["medicine"]
    assert payload["status"] == "matched"
    assert medicine["net_savings"]["available"] is True
    assert medicine["net_savings"]["savings_per_pack"] == 18.0
    assert medicine["net_savings"]["savings_percentage"] == 60.0
    assert medicine["jan_price_date"] == "2026-07-15"
    assert medicine["branded_mrp_date"] == "2026-08-01"
    assert "not independently audited" in medicine["verification_status"]


@pytest.mark.parametrize(
    ("jan_ingredient", "jan_strength", "jan_form", "jan_pack"),
    [
        ("paracetamol", "500 mg", "tablets", "20 tablets"),
        ("ibuprofen", "500 mg", "tablets", "10 tablets"),
        ("paracetamol", "250 mg", "tablets", "10 tablets"),
        ("paracetamol", "500 mg", "capsules", "10 capsules"),
    ],
)
def test_non_equivalent_source_pair_does_not_calculate_savings(monkeypatch, jan_ingredient, jan_strength, jan_form, jan_pack):
    brand_url = "https://www.cipla.com/medicines/sample-500"
    jan_url = "https://janaushadhi.gov.in/product/sample-500"

    def handler(request):
        if request.url.host == "api.search.tinyfish.ai":
            if "Jan Aushadhi" in request.url.params["query"]:
                url, title = jan_url, f"Sample {jan_strength} {jan_form} {jan_pack}"
            else:
                url, title = brand_url, "Sample 500 mg tablets 10 tablets"
            return httpx.Response(200, json={"results": [_result(url, title)]})
        return httpx.Response(200, json={"results": [
            {"url": brand_url, "title": "Sample 500 mg tablets 10 tablets", "text": "Active ingredient: paracetamol\nStrength: 500 mg\nDosage form: tablets\nPack size: 10 tablets\nMRP: ₹30.00"},
            {"url": jan_url, "title": f"Sample {jan_strength} {jan_form} {jan_pack}", "text": f"Active ingredient: {jan_ingredient}\nStrength: {jan_strength}\nDosage form: {jan_form}\nPack size: {jan_pack}\nPrice: ₹20.00"},
        ], "errors": []})

    _mock_client(monkeypatch, handler)
    monkeypatch.setenv("TINYFISH_API_KEY", "mock-key")
    payload = asyncio.run(tinyfish_search.search_medicine("Sample 500 mg tablets", "mock-key"))
    assert payload["status"] == "matched"
    savings = payload["medicine"]["net_savings"]
    if jan_ingredient == "paracetamol" and jan_strength == "500 mg" and jan_form == "tablets":
        assert savings["available"] is True
        assert savings["savings_per_pack"] == 20.0
        assert payload["medicine"]["jan_aushadhi_price"] == 20.0
        assert payload["medicine"]["jan_comparable_pack_price"] == 10.0
    else:
        assert savings["available"] is False
        assert savings["savings_per_pack"] is None


def test_live_search_fails_closed_when_planner_is_not_implemented(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("TINYFISH_API_KEY", "mock-key")
    payload = TestClient(main.app).post("/api/search", json={"query": "Sample 500 mg"}).json()
    assert payload["status"] == "error"
    assert "OPENROUTER_API_KEY" in payload["message"]
    assert "mock-key" not in str(payload)


def test_tinyfish_auth_failure_is_classified_without_leaking_response(monkeypatch):
    def handler(request):
        return httpx.Response(401, json={"error": {"message": "secret details not for clients"}})

    _mock_client(monkeypatch, handler)
    with pytest.raises(tinyfish_search.TinyFishAuthenticationError):
        asyncio.run(tinyfish_search.search_medicine("Sample 500 mg", "mock-key"))


def _plan(query="Sample 500 mg tablets"):
    return {
        "original_query": query,
        "brand_name": "Sample",
        "generic_name": "paracetamol",
        "strength": "500 mg",
        "dosage_form": "tablets",
        "search_queries": [
            "Sample 500 mg tablets manufacturer MRP India",
            "Sample paracetamol 500 mg tablets Jan Aushadhi PMBI price",
        ],
        "needs_clarification": False,
        "clarification_question": None,
    }


def _mock_openrouter(monkeypatch, content=None, status=200):
    body = {"choices": [{"message": {"content": content if content is not None else __import__("json").dumps(_plan())}}]}

    def handler(request):
        assert request.url == nemotron_planner.OPENROUTER_URL
        assert request.headers["Authorization"] == "Bearer mocked-only"
        data = __import__("json").loads(request.read())
        assert data["model"] == nemotron_planner.MODEL_ID
        assert data["response_format"]["type"] == "json_object"
        assert data["reasoning"]["effort"] == "minimal"
        return httpx.Response(status, json=body)

    monkeypatch.setattr(
        nemotron_planner,
        "make_openrouter_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def test_openrouter_plan_is_validated_and_queries_flow_to_tinyfish(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "mocked-only")
    monkeypatch.setenv("TINYFISH_API_KEY", "mock-tinyfish")
    _mock_openrouter(monkeypatch)
    brand_url = "https://www.cipla.com/medicines/sample-500"
    jan_url = "https://janaushadhi.gov.in/product/sample-500"
    seen_queries = []

    def handler(request):
        if request.url.host == "api.search.tinyfish.ai":
            query = request.url.params["query"]
            seen_queries.append(query)
            url = jan_url if "Jan Aushadhi" in query else brand_url
            return httpx.Response(200, json={"results": [_result(url, "Sample 500 mg tablets 10 tablets")]})
        return httpx.Response(200, json={"results": [
            {"url": brand_url, "title": "Sample 500 mg tablets 10 tablets", "text": "Active ingredient: paracetamol\nStrength: 500 mg\nDosage form: tablets\nPack size: 10 tablets\nMRP: ₹30.00"},
            {"url": jan_url, "title": "Sample 500 mg tablets 10 tablets", "text": "Active ingredient: paracetamol\nStrength: 500 mg\nDosage form: tablets\nPack size: 10 tablets\nPrice: ₹12.00"},
        ], "errors": []})

    _mock_client(monkeypatch, handler)
    response = TestClient(main.app).post("/api/search", json={"query": "Sample 500 mg tablets"})
    assert response.json()["medicine"]["net_savings"]["savings_per_pack"] == 18.0
    assert len(seen_queries) == 4
    assert any("MRP" in query for query in seen_queries)
    assert any("Jan Aushadhi" in query for query in seen_queries)
    assert any("10 tablets" in query for query in seen_queries)


def test_generic_name_and_strength_without_brand_proceeds_to_tinyfish(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "mocked-only")
    monkeypatch.setenv("TINYFISH_API_KEY", "mock-tinyfish")
    plan = _plan("metformin 500")
    plan.update(
        brand_name=None,
        generic_name="metformin",
        strength="500 mg",
        dosage_form=None,
        search_queries=["metformin 500 mg medicine price and product details"],
    )
    _mock_openrouter(monkeypatch, content=json.dumps(plan))
    captured = {}

    async def fake_search_medicine(query, api_key, search_queries=None, **attributes):
        captured.update(query=query, api_key=api_key, search_queries=search_queries, **attributes)
        return {"status": "not_found", "medicine": None}

    monkeypatch.setattr(main, "search_medicine", fake_search_medicine)
    payload = TestClient(main.app).post("/api/search", json={"query": "metformin 500"}).json()
    assert payload == {"status": "not_found", "medicine": None}
    assert captured == {
        "query": "metformin 500",
        "api_key": "mock-tinyfish",
        "search_queries": ["metformin 500 mg medicine price and product details"],
        "generic_name": "metformin",
        "strength": "500 mg",
        "dosage_form": None,
        "brand_name": None,
    }


def test_brand_specific_plan_proceeds_to_live_retrieval(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "mocked-only")
    monkeypatch.setenv("TINYFISH_API_KEY", "mock-tinyfish")
    plan = _plan("Dolo 650")
    plan["brand_name"] = "Dolo"
    _mock_openrouter(monkeypatch, content=json.dumps(plan))
    captured = {}

    async def fake_search_medicine(query, api_key, search_queries=None, **attributes):
        captured.update(query=query, search_queries=search_queries, **attributes)
        return {"status": "matched", "match_source": "tinyfish", "medicine": {"name": "retrieved"}}

    monkeypatch.setattr(main, "search_medicine", fake_search_medicine)
    payload = TestClient(main.app).post("/api/search", json={"query": "Dolo 650"}).json()
    assert payload["status"] == "matched"
    assert captured["query"] == "Dolo 650"
    assert len(captured["search_queries"]) == 2
    assert captured["brand_name"] == "Dolo"


@pytest.mark.parametrize(
    ("requested", "record", "expected"),
    [
        ("Crocin 650", {"brand_name": "Dolo", "name": "Dolo 650 Tablet"}, False),
        ("Crocin", {"brand_name": "CROCIN", "name": "Crocin 650 Tablet"}, True),
        ("Glycomet", {"brand_name": "Glycomet", "name": "Glycomet 500 SR Tablet"}, True),
    ],
)
def test_requested_brand_must_match_product_evidence(requested, record, expected):
    assert tinyfish_search._brand_matches(requested, record) is expected


def test_ingredient_sentence_normalizes_to_the_active_ingredient():
    assert tinyfish_search._ingredient_key(
        "Contains exactly 650 mg of Paracetamol IP (Acetaminophen) per tablet."
    ) == tinyfish_search._ingredient_key("Paracetamol")
    assert tinyfish_search._ingredient_key("Amoxicillin 500 mg + clavulanic acid 125 mg") == (
        "amoxicillin", "clavulanicacid",
    )


def test_search_plan_accepts_generic_target_without_brand():
    plan = _plan("metformin 500")
    plan.update(brand_name=None, generic_name="metformin", strength="500 mg")
    validated = nemotron_planner.SearchPlan.model_validate(plan)
    assert validated.brand_name is None
    assert validated.generic_name == "metformin"


@pytest.mark.parametrize(
    "message",
    [
        {"content": json.dumps(_plan())},
        {"content": [{"type": "text", "text": json.dumps(_plan())}]},
        {"parsed": _plan()},
        {"content": f"```json\n{json.dumps(_plan())}\n```"},
    ],
)
def test_openrouter_structured_output_shapes_are_accepted(message):
    result = nemotron_planner._plan_from_response(
        {"choices": [{"message": message}]}, "Sample 500 mg tablets"
    )
    assert result.generic_name == "paracetamol"


def test_openrouter_missing_message_content_has_actionable_validation_error():
    with pytest.raises(nemotron_planner.InvalidPlannerResponse, match="content is missing"):
        nemotron_planner._plan_from_response(
            {"choices": [{"message": {}}]}, "Sample 500 mg tablets"
        )


def test_openrouter_unescaped_control_character_inside_json_string_is_repaired():
    malformed_content = json.dumps(_plan()).replace("manufacturer MRP", "manufacturer\nMRP")
    result = nemotron_planner._plan_from_response(
        {"choices": [{"message": {"content": malformed_content}}]},
        "Sample 500 mg tablets",
    )
    assert result.search_queries[0] == "Sample 500 mg tablets manufacturer MRP India"


def test_openrouter_truncated_completion_is_reported_as_such():
    with pytest.raises(nemotron_planner.InvalidPlannerResponse, match="truncated"):
        nemotron_planner._plan_from_response(
            {"choices": [{"finish_reason": "length", "message": {"content": "{"}}]},
            "Sample 500 mg tablets",
        )


def test_malformed_nemotron_json_returns_safe_error_and_skips_tinyfish(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "mocked-only")
    monkeypatch.setenv("TINYFISH_API_KEY", "mock-tinyfish")
    _mock_openrouter(monkeypatch, content="not JSON")
    monkeypatch.setattr(tinyfish_search, "make_tinyfish_client", lambda: pytest.fail("TinyFish must not run"))
    payload = TestClient(main.app).post("/api/search", json={"query": "Sample"}).json()
    assert payload["status"] == "error"
    assert "invalid search plan" in payload["message"]
    assert "mocked-only" not in str(payload)


def test_malformed_clarification_plan_returns_structured_validation_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "mocked-only")
    monkeypatch.setenv("TINYFISH_API_KEY", "mock-tinyfish")
    plan = _plan("Sample")
    plan["needs_clarification"] = True
    plan["clarification_question"] = ""
    _mock_openrouter(monkeypatch, content=json.dumps(plan))
    monkeypatch.setattr(tinyfish_search, "make_tinyfish_client", lambda: pytest.fail("TinyFish must not run"))
    payload = TestClient(main.app).post("/api/search", json={"query": "Sample"}).json()
    assert payload["status"] == "error"
    assert "clarification_question" in payload["message"]


def test_clarification_response_with_empty_query_is_returned_without_tinyfish(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "mocked-only")
    monkeypatch.setenv("TINYFISH_API_KEY", "mock-tinyfish")
    plan = {
        "original_query": "paracetamol",
        "brand_name": None,
        "generic_name": "paracetamol",
        "strength": None,
        "dosage_form": None,
        "search_queries": [
            "Which brand of paracetamol are you looking for? (e.g., Calpol, Crocin, Dolo, etc.)",
            "",
        ],
        "needs_clarification": True,
        "clarification_question": "Which brand of paracetamol are you looking for? (e.g., Calpol, Crocin, Dolo, etc.)",
    }
    _mock_openrouter(monkeypatch, content=json.dumps(plan))
    monkeypatch.setattr(tinyfish_search, "make_tinyfish_client", lambda: pytest.fail("TinyFish must not run"))
    payload = TestClient(main.app).post("/api/search", json={"query": "paracetamol"}).json()
    assert payload == {
        "status": "clarification_required",
        "question": plan["clarification_question"],
        "candidates": [],
    }


def test_empty_search_query_entries_are_filtered_before_tinyfish(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "mocked-only")
    monkeypatch.setenv("TINYFISH_API_KEY", "mock-tinyfish")
    plan = _plan()
    plan["search_queries"] = ["Sample 500 mg manufacturer MRP", "  "]
    _mock_openrouter(monkeypatch, content=json.dumps(plan))
    captured = {}

    async def fake_search_medicine(query, api_key, search_queries=None, **attributes):
        captured.update(query=query, api_key=api_key, search_queries=search_queries, **attributes)
        return {"status": "not_found", "medicine": None}

    monkeypatch.setattr(main, "search_medicine", fake_search_medicine)
    payload = TestClient(main.app).post("/api/search", json={"query": "Sample 500 mg tablets"}).json()
    assert payload["status"] == "not_found"
    assert captured == {
        "query": "Sample 500 mg tablets",
        "api_key": "mock-tinyfish",
        "search_queries": ["Sample 500 mg manufacturer MRP"],
        "generic_name": "paracetamol",
        "strength": "500 mg",
        "dosage_form": "tablets",
        "brand_name": "Sample",
    }


def test_openrouter_auth_failure_is_actionable_and_secret_safe(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "mocked-only")
    monkeypatch.setenv("TINYFISH_API_KEY", "mock-tinyfish")
    _mock_openrouter(monkeypatch, status=401)
    payload = TestClient(main.app).post("/api/search", json={"query": "Sample"}).json()
    assert payload["status"] == "error"
    assert "Check OPENROUTER_API_KEY" in payload["message"]
    assert "mocked-only" not in str(payload)


def test_tinyfish_failure_returns_safe_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "mocked-only")
    monkeypatch.setenv("TINYFISH_API_KEY", "mock-tinyfish")
    _mock_openrouter(monkeypatch)
    _mock_client(monkeypatch, lambda request: httpx.Response(503, text="secret upstream detail"))
    payload = TestClient(main.app).post("/api/search", json={"query": "Sample 500 mg tablets"}).json()
    assert payload["status"] == "error"
    assert "TinyFish live web retrieval" in payload["message"]
    assert "secret upstream detail" not in str(payload)


def test_missing_price_evidence_does_not_calculate_savings(monkeypatch):
    brand_url = "https://www.cipla.com/medicines/sample-500"

    def handler(request):
        if request.url.host == "api.search.tinyfish.ai":
            return httpx.Response(200, json={"results": [_result(brand_url, "Sample 500 mg tablets 10 tablets")]})
        return httpx.Response(200, json={"results": [
            {"url": brand_url, "title": "Sample 500 mg tablets 10 tablets", "text": "Active ingredient: paracetamol\nStrength: 500 mg\nDosage form: tablets\nPack size: 10 tablets"},
        ], "errors": []})

    _mock_client(monkeypatch, handler)
    payload = asyncio.run(tinyfish_search.search_medicine("Sample", "mock-key", _plan()["search_queries"]))
    assert payload["status"] == "matched"
    assert payload["medicine"]["net_savings"]["available"] is False
    assert payload["medicine"]["market_mrp"] is None
    assert payload["medicine"]["jan_aushadhi_price"] is None


def test_clarification_plan_stops_before_tinyfish(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "mocked-only")
    monkeypatch.setenv("TINYFISH_API_KEY", "mock-tinyfish")
    plan = _plan()
    plan["needs_clarification"] = True
    plan["clarification_question"] = "Which product?"
    _mock_openrouter(monkeypatch, content=__import__("json").dumps(plan))
    monkeypatch.setattr(tinyfish_search, "make_tinyfish_client", lambda: pytest.fail("TinyFish must not run"))
    payload = TestClient(main.app).post("/api/search", json={"query": "Sample 500 mg tablets"}).json()
    assert payload["status"] == "clarification_required"
    assert payload["question"] == "Which product?"


def test_terminal_events_are_single_line_json_without_secrets(capsys):
    event_log.log_event("search_received", query_length=12)
    line = capsys.readouterr().out.strip()
    assert __import__("json").loads(line) == {"stage": "search_received", "query_length": 12}
    assert "OPENROUTER_API_KEY" not in line


def test_official_pmbi_product_match_extracts_mrp_and_pack(monkeypatch):
    capture = {}
    _mock_official_search_fetch(monkeypatch, capture=capture)
    _mock_pmbi_product_list(monkeypatch, _pmbi_table([["1", "511", "Paracetamol Tablets IP 650 mg", "15's", "14.07"]]))
    payload = asyncio.run(tinyfish_search.search_medicine(
        "Dolo 650", "mock-key", ["Dolo 650 manufacturer MRP"],
        generic_name="Paracetamol", strength="650 mg", dosage_form="tablet",
    ))
    medicine = payload["medicine"]
    assert medicine["jan_aushadhi_price"] == 14.07
    assert medicine["jan_price_source_url"] == tinyfish_search.PMBI_PRODUCT_LIST_URL
    assert medicine["jan_product_description"] == "Paracetamol Tablets IP 650 mg"
    assert medicine["jan_unit_size"] == "15's"
    assert medicine["jan_strength"] == "650 mg"
    assert medicine["jan_dosage_form"] == "tablets"
    assert medicine["jan_price_retrieved_at"]
    assert "Paracetamol 650 mg tablet" in capture["jan_query"]
    assert "pmbi.co.in" in capture["jan_domains"]
    assert "janaushadhi.gov.in" in capture["jan_domains"]


def test_pmbi_product_without_unit_size_keeps_price_but_disables_savings(monkeypatch):
    _mock_official_search_fetch(monkeypatch)
    _mock_pmbi_product_list(monkeypatch, _pmbi_table([["1", "511", "Paracetamol Tablets IP 650 mg", "", "14.07"]]))
    payload = asyncio.run(tinyfish_search.search_medicine(
        "Dolo 650", "mock-key", ["Dolo 650 manufacturer MRP"],
        generic_name="Paracetamol", strength="650 mg", dosage_form="tablet",
    ))
    medicine = payload["medicine"]
    assert medicine["jan_aushadhi_price"] == 14.07
    assert medicine["jan_unit_size"] is None
    assert medicine["jan_price_status"] == "pack_size_missing"
    assert medicine["net_savings"]["available"] is False


def test_no_official_pmbi_match_returns_unavailable(monkeypatch):
    brand_url = "https://www.cipla.com/medicines/example-650"
    brand = {
        "url": brand_url,
        "title": "Example 650 mg tablets 15 tablets",
        "content": {
            "title": "Example 650 mg tablets 15 tablets",
            "text": "Active ingredient: Paracetamol\nStrength: 650 mg\nDosage form: tablets\nPack size: 15 tablets\nMRP: ₹30.00",
        },
    }
    _mock_official_search_fetch(monkeypatch, brand_page=brand)
    _mock_pmbi_product_list(monkeypatch, _pmbi_table([]))
    payload = asyncio.run(tinyfish_search.search_medicine(
        "Dolo 650", "mock-key", ["Dolo 650 manufacturer MRP"],
        generic_name="Paracetamol", strength="650 mg", dosage_form="tablet",
    ))
    assert payload["status"] == "matched"
    assert payload["medicine"]["jan_aushadhi_price"] is None
    assert payload["medicine"]["jan_price_status"] == "unavailable"
    assert payload["medicine"]["jan_price_reason"] == "no_verified_official_mrp_found"
    assert tinyfish_search.PMBI_PRODUCT_LIST_URL in payload["medicine"]["jan_price_sources_checked"]
    assert all(url in payload["medicine"]["jan_price_sources_checked"] for url in tinyfish_search.JAN_PORTFOLIO_URLS)
    assert payload["medicine"]["net_savings"]["available"] is False


def test_requested_brand_mismatch_does_not_return_another_brand(monkeypatch):
    brand_url = "https://www.1mg.com/drugs/dolo-650-tablet-123"
    brand = {
        "url": brand_url,
        "title": "Dolo 650 Tablet",
        "content": {
            "title": "Dolo 650 Tablet",
            "text": "Brand: Dolo\nActive ingredient: Paracetamol\nStrength: 650 mg\nDosage form: tablet\nPack size: 15 tablets\nMRP: ₹32.28",
        },
    }
    _mock_official_search_fetch(monkeypatch, brand_page=brand)
    _mock_pmbi_product_list(monkeypatch, _pmbi_table([]))
    payload = asyncio.run(tinyfish_search.search_medicine(
        "Crocin 650", "mock-key", ["Crocin 650 mg tablet price"],
        generic_name="Paracetamol", strength="650 mg", dosage_form="tablet",
        brand_name="Crocin",
    ))
    assert payload["status"] == "not_found"
    assert "requested brand Crocin" in payload["message"]
    assert "Dolo" not in payload["message"]


def test_retail_page_title_contamination_and_discount_percent_are_rejected():
    page = {
        "url": "https://www.1mg.com/drugs/cetirizine-10mg-tablet-543132",
        "title": "Buy Cetirizine 10mg Tablet Online: View Uses, Side Effects, Price | 1mg",
        "text": "Home\n# Cetirizine 10mg Tablet\n10 tablets\nComposition:\nCetirizine (10mg)\nMRP ₹155% off",
    }
    record = tinyfish_search._parse_page(page, "brand", {"url": page["url"], "site_name": "www.1mg.com"})

    assert record["name"] == "Cetirizine 10mg Tablet"
    assert record["brand_name"] is None
    assert record["ingredients"] == "Cetirizine"
    assert record["strength"] == "10mg"
    assert record["dosage_form"] == "tablet"
    assert record["pack_size"] == "10 tablets"
    assert record["price"] is None
    assert "1mg" not in (record["strength"] or "")
    assert "1mg" not in (record["brand_name"] or "")


def test_tinyfish_product_summary_extracts_brand_strength_pack_and_mrp_separately():
    page = {
        "url": "https://pharmeasy.in/online-medicine-order/cetirizine-10mg-strip-of-10-tablets-3608111",
        "final_url": "https://pharmeasy.in/online-medicine-order/cetirizine-10mg-strip-of-10-tablets-3608111",
        "title": "Cetirizine 10 MG Tablet (10): Uses, Side Effects, Price & Dosage | PharmEasy",
        "text": "# Cetirizine 10Mg Strip Of 10 Tablets\n10 Tablet(s) in Strip\nMRP ₹19.74\n₹13.82\n30% OFF\n₹1.38/tablet\nMade by\nGLENMARK GX\nDosage\nTABLET\nProduct Summary\n| Salt Content | Cetirizine(10.0 Mg) |\nManufacturer Details\nBrand:CETIRIZINE\nManufacturer Name:GLENMARK GX",
        "published_date": "21 Sept 2026",
    }
    record = tinyfish_search._parse_page(page, "brand", {"url": page["url"], "site_name": "pharmeasy.in"})

    assert record["name"] == "Cetirizine 10Mg Strip Of 10 Tablets"
    assert record["brand_name"] == "CETIRIZINE"
    assert record["ingredients"] == "Cetirizine"
    assert record["strength"] == "10.0 Mg"
    assert record["dosage_form"] == "tablet"
    assert record["pack_size"] == "10 tablets"
    assert record["price"] == __import__("decimal").Decimal("19.74")
    assert record["source_url"] == page["url"]
    assert record["source_date"] == "21 Sept 2026"


def test_price_without_pack_size_is_marked_unverified_and_not_comparable():
    brand = tinyfish_search._parse_page(
        {
            "url": "https://www.1mg.com/drugs/alerid-10-tablet",
            "title": "Alerid 10 Tablet",
            "text": "Ingredients: Cetirizine\nStrength: 10 mg\nDosage form: tablets\nMRP: ₹100",
        },
        "brand",
        {"url": "https://www.1mg.com/drugs/alerid-10-tablet"},
    )
    jan = {
        "price": __import__("decimal").Decimal("12"),
        "pack_key": ("10", "tablets"),
        "pack_size": "10 tablets",
        "ingredient_key": ("cetirizine",),
        "strength_key": ("10:mg",),
        "form_key": "tablet",
        "trusted": True,
    }
    public = tinyfish_search._public_medicine(brand, jan, "Alerid 10", False, [])

    assert brand["price"] == __import__("decimal").Decimal("100")
    assert brand["pack_key"] is None
    assert public["market_price_status"] == "price_basis_unverified"
    assert public["pack_size"] is None
    assert public["branded_unit_size"] is None
    assert public["jan_unit_size"] == "10 tablets"
    assert public["net_savings"]["available"] is False
    assert public["jan_comparable_pack_price"] is None


def test_unreadable_official_pdf_is_not_treated_as_a_price(monkeypatch, capsys):
    _mock_official_search_fetch(monkeypatch, jan_page={"url": "https://www.pmbi.co.in/Data/eBrochure/mock.pdf", "content": {"text": None}})
    _mock_pmbi_product_list(monkeypatch, _pmbi_table([]))
    payload = asyncio.run(tinyfish_search.search_medicine(
        "Dolo 650", "mock-key", ["Dolo 650 manufacturer MRP"],
        generic_name="Paracetamol", strength="650 mg", dosage_form="tablet",
    ))
    assert payload["status"] == "not_found"
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any(event.get("reason") == "unreadable_or_empty_content" for event in events)


@pytest.mark.parametrize(
    ("product", "unit"),
    [
        ("Paracetamol Tablets IP 500 mg", "15's"),
        ("Paracetamol Capsules IP 650 mg", "15's"),
    ],
)
def test_pmbi_mismatched_strength_or_form_is_rejected(monkeypatch, product, unit):
    _mock_official_search_fetch(monkeypatch)
    _mock_pmbi_product_list(monkeypatch, _pmbi_table([["1", "511", product, unit, "14.07"]]))
    payload = asyncio.run(tinyfish_search.search_medicine(
        "Dolo 650", "mock-key", ["Dolo 650 manufacturer MRP"],
        generic_name="Paracetamol", strength="650 mg", dosage_form="tablet",
    ))
    assert payload["status"] == "not_found"


def test_official_pmbi_price_supports_like_for_like_savings(monkeypatch):
    brand_url = "https://www.cipla.com/medicines/example-650"
    brand = {
        "url": brand_url,
        "title": "Example 650 mg tablets 15 tablets",
        "content": {
            "title": "Example 650 mg tablets 15 tablets",
            "text": "Active ingredient: Paracetamol\nStrength: 650 mg\nDosage form: tablets\nPack size: 15 tablets\nMRP: ₹30.00",
        },
    }
    capture = {}
    _mock_official_search_fetch(monkeypatch, brand_page=brand, capture=capture)
    _mock_pmbi_product_list(monkeypatch, _pmbi_table([["1", "511", "Paracetamol Tablets IP 650 mg", "15's", "14.07"]]))
    payload = asyncio.run(tinyfish_search.search_medicine(
        "Example 650 mg tablets", "mock-key", ["Example 650 mg tablets MRP"],
        generic_name="Paracetamol", strength="650 mg", dosage_form="tablet",
    ))
    medicine = payload["medicine"]
    assert medicine["jan_aushadhi_price"] == 14.07
    assert medicine["branded_mrp"] == 30.0
    assert medicine["net_savings"]["available"] is True
    assert medicine["net_savings"]["savings_per_pack"] == 15.93

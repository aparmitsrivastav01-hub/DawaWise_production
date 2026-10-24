"""Live medicine source retrieval with TinyFish Search and Fetch (no answer model)."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx
from event_log import log_event

SEARCH_URL = "https://api.search.tinyfish.ai"
FETCH_URL = "https://api.fetch.tinyfish.ai"
TIMEOUT = httpx.Timeout(45.0, connect=10.0)
FETCH_LIMIT = 5
PMBI_PRODUCT_LIST_URL = "https://www.pmbi.co.in/ProductList.aspx"
JAN_PORTFOLIO_URLS = (
    "https://janaushadhi.gov.in/product-portfolio/product-mrp-list",
    "https://janaushadhi.gov.in/product-portfolio/product-view-list",
)
JAN_OFFICIAL_DOMAINS = ("pmbi.co.in", "janaushadhi.gov.in")

OFFICIAL_DOMAINS = (
    "janaushadhi.gov.in", "pmbi.co.in", "pharmaceuticals.gov.in", "cdsco.gov.in",
    "sunpharma.com", "cipla.com", "drreddys.com", "lupin.com", "torrentpharma.com",
    "mankindpharma.com", "zyduslife.com", "abbott.co.in", "gsk.com", "pfizer.com",
    "sanofi.com", "alkemlabs.com", "intasparmaceuticals.com", "alembicpharmaceuticals.com",
)
TRUSTED_PHARMACY_DOMAINS = (
    "1mg.com", "apollopharmacy.in", "netmeds.com", "pharmeasy.in", "medplusmart.com",
)
FORM_ALIASES = {
    "tablet": ("tablet", "tablets", "tab", "tabs"),
    "capsule": ("capsule", "capsules", "cap", "caps"),
    "syrup": ("syrup",), "suspension": ("suspension",),
    "injection": ("injection", "injectable"), "cream": ("cream",),
    "ointment": ("ointment",), "drops": ("drop", "drops"),
    "gel": ("gel",), "solution": ("solution",), "powder": ("powder",),
}


class TinyFishAuthenticationError(Exception):
    pass


class TinyFishUnavailableError(Exception):
    pass


def make_tinyfish_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=TIMEOUT)


def make_pmbi_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True)


def _domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            return ""
        return parsed.hostname.lower().removeprefix("www.")
    except (TypeError, ValueError):
        return ""


def _is_on_domain(host: str, domains: tuple[str, ...]) -> bool:
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def _trusted_source(url: str, kind: str) -> bool:
    host = _domain(url)
    if not host:
        return False
    if kind == "jan":
        return host.endswith(".gov.in") or _is_on_domain(host, ("pmbi.co.in", "janaushadhi.gov.in"))
    return host.endswith(".gov.in") or _is_on_domain(host, OFFICIAL_DOMAINS + TRUSTED_PHARMACY_DOMAINS)


def _results(payload: object) -> list[dict]:
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return []
    out = []
    for item in payload["results"]:
        if isinstance(item, dict) and isinstance(item.get("url"), str) and _domain(item["url"]):
            out.append(item)
    return out


async def _search(client: httpx.AsyncClient, key: str, query: str, domains: tuple[str, ...] = ()) -> list[dict]:
    params = {"query": query, "location": "IN", "language": "en", "purpose": "Find current medicine product details and published price sources"}
    if domains:
        params["include_domains"] = ",".join(domains)
    response = await client.get(SEARCH_URL, headers={"X-API-Key": key}, params=params)
    if response.status_code == 401:
        raise TinyFishAuthenticationError
    response.raise_for_status()
    return _results(response.json())


async def _fetch(client: httpx.AsyncClient, key: str, urls: list[str]) -> list[dict]:
    if not urls:
        return []
    response = await client.post(
        FETCH_URL,
        headers={"X-API-Key": key, "Content-Type": "application/json"},
        json={"urls": urls[:10], "format": "markdown", "ttl": 0, "per_url_timeout_ms": 40000},
    )
    if response.status_code == 401:
        raise TinyFishAuthenticationError
    response.raise_for_status()
    data = response.json()
    return data.get("results", []) if isinstance(data, dict) and isinstance(data.get("results"), list) else []


def _source_date(page: dict) -> str | None:
    for key in ("published_date", "date", "last_modified", "lastModified"):
        value = page.get(key)
        if isinstance(value, str) and re.search(r"\b(?:19|20)\d{2}\b", value):
            return value[:40]
    text = str(page.get("text") or "")
    match = re.search(r"(?:updated|last updated|effective from|w.e.f.|published)(?:\s+on)?\s*[:\-]?\s*((?:\d{1,2}[/-]){1,2}\d{2,4}|\d{4}-\d{2}-\d{2})", text, re.I)
    return match.group(1) if match else None


def _money_value(text: str) -> Decimal | None:
    try:
        value = Decimal(text.replace(",", "").replace("₹", "").replace("Rs.", "").replace("Rs", "").strip())
    except (InvalidOperation, ValueError):
        return None
    return value if value.is_finite() and value > 0 else None


def _price(text: str, kind: str) -> Decimal | None:
    labels = r"(?:MRP|maximum\s+retail\s+price|market\s+price|selling\s+price|price)" if kind == "brand" else r"(?:MRP|price|unit\s+price|selling\s+price)"
    # A squeezed retailer fragment such as "MRP ₹155% off" means MRP ₹15
    # followed by 5% off; it must never be interpreted as ₹155.
    amount = r"([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)(?![0-9,.]|\s*%)"
    pattern = rf"(?i)\b{labels}\b\s*(?:\(\s*INR\s*\))?\s*(?:is|:|=|\-|–)?\s*(?:INR\s*|Rs\.?\s*|₹\s*)?{amount}"
    found = re.search(pattern, text)
    return _money_value(found.group(1)) if found else None


def _label(text: str, labels: str) -> str | None:
    pattern = rf"(?im)^\s*(?:{labels})\s*[:\-]\s*([^\n|;]+)"
    found = re.search(pattern, text)
    if not found:
        # TinyFish preserves key/value product summaries as Markdown tables.
        pattern = rf"(?im)^\s*\|?\s*(?:{labels})\s*\|\s*([^\n|]+)"
        found = re.search(pattern, text)
    if not found:
        pattern = rf"(?i)(?:{labels})\s*[:\-]\s*([^\n|;]+)"
        found = re.search(pattern, text)
    value = re.sub(r"\s+", " ", found.group(1)).strip(" *#\t") if found else ""
    return value[:160] or None


def _product_title(page: dict, title: str, text: str) -> str | None:
    heading = re.search(r"(?m)^\s*#{1,3}\s+([^\n]+)", text)
    value = heading.group(1).strip() if heading else title
    value = re.sub(r"(?i)^\s*(?:buy|order|shop)\s+", "", value)
    value = re.sub(r"(?i)\s*(?:\||[-–])\s*(?:1mg|apollopharmacy|netmeds|pharmeasy)(?:\.com|\.in)?\s*$", "", value)
    value = re.sub(r"(?i)\s+(?:online(?:\s+at\s+best\s+price)?|price,\s*substitutes.*)$", "", value)
    return re.sub(r"\s+", " ", value).strip(" -|:") or None


def _extract_pack_size(text: str, dosage_form: str | None) -> str | None:
    # Prefer explicit package descriptions, then a standalone package-count line.
    patterns = (
        r"(?i)\bstrip\s+of\s+(\d+(?:\.\d+)?)\s*(tablets?|tabs?|capsules?|caps?)\b",
        r"(?i)\b(\d+(?:\.\d+)?)\s*(tablets?|tabs?|capsules?|caps?)\s*\(s\)\s+in\s+(?:a\s+)?(?:strip|pack|box)\b",
        r"(?im)^\s*(\d+(?:\.\d+)?\s*(?:tablets?|tabs?|capsules?|caps?|units?|pieces?|pcs?|ml|sachets?|vials?|ampoules?))\s*$",
    )
    for pattern in patterns:
        found = re.search(pattern, text)
        if found:
            if len(found.groups()) > 1:
                return f"{found.group(1)} {found.group(2).lower()}"
            return re.sub(r"\s+", " ", found.group(1)).strip().lower()
    match = re.search(r"(?i)\b(\d+(?:\.\d+)?)\s*['’]s\b", text)
    if match and _form_key(dosage_form or "") in ("tablet", "capsule"):
        return f"{match.group(1)}'s"
    return None


def _infer_brand_name(product_name: str | None, ingredients: str | None) -> str | None:
    if not product_name or not ingredients:
        return None
    product_tokens = set(re.findall(r"[a-z0-9]+", product_name.lower()))
    ingredient_tokens = set(re.findall(r"[a-z0-9]+", ingredients.lower()))
    if product_tokens.intersection(ingredient_tokens):
        return None
    brand = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mcg|μg|µg|mg|g|ml|iu|%)\b", " ", product_name, flags=re.I)
    brand = re.sub(r"\b(?:tablets?|tabs?|capsules?|caps?|syrup|suspension|injection|cream|ointment|drops?|gel|solution|powder)\b", " ", brand, flags=re.I)
    brand = re.sub(r"\s+", " ", brand).strip(" -|:")
    return brand or None


_BRAND_MATCH_STOPWORDS = {
    "tablet", "tablets", "tab", "tabs", "capsule", "capsules", "cap", "caps",
    "syrup", "suspension", "injection", "cream", "ointment", "drops", "gel",
    "solution", "powder", "medicine", "price", "india", "mrp", "manufacturer",
    "strip", "pack", "box", "of", "mg", "mcg", "ug", "g", "ml", "iu",
}


def _brand_matches(requested_brand: str | None, record: dict) -> bool:
    """Require each requested brand word to appear in the extracted product evidence."""
    if not requested_brand:
        return True
    requested = {
        token for token in re.findall(r"[a-z]+", requested_brand.lower())
        if token not in _BRAND_MATCH_STOPWORDS
    }
    if not requested:
        return True
    evidence = " ".join(str(record.get(field) or "") for field in ("brand_name", "name"))
    observed = {
        token for token in re.findall(r"[a-z]+", evidence.lower())
        if token not in _BRAND_MATCH_STOPWORDS
    }
    return requested.issubset(observed)


def _ingredient_key(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    value = re.sub(r"\([^)]*\)", " ", value.lower())
    value = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mcg|μg|µg|mg|g|ml|iu|%)\b", " ", value)
    value = re.sub(
        r"\b(?:contains?|exactly|each|of|per|tablet|tablets|tab|tabs|capsule|capsules|cap|caps|ip|bp|usp)\b",
        " ", value,
    )
    parts = re.split(r"\s*(?:\+|,|/|\band\b)\s*", value)
    return tuple(sorted(re.sub(r"[^a-z0-9]", "", part) for part in parts if re.sub(r"[^a-z0-9]", "", part)))


def _strength_key(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    items = re.findall(r"(\d+(?:\.\d+)?)\s*(mcg|μg|µg|mg|g|ml|iu|%)(?=\W|$)", value.lower())
    normalized = []
    for amount, unit in items:
        unit = {"μg": "mcg", "µg": "mcg"}.get(unit, unit)
        normalized.append(f"{Decimal(amount).normalize()}:{unit}")
    return tuple(normalized)


def _form_key(value: str | None) -> str | None:
    if not value:
        return None
    tokens = set(re.findall(r"[a-z]+", value.lower()))
    form = next((form for form, aliases in FORM_ALIASES.items() if tokens.intersection(aliases)), None)
    if not form:
        return None
    release_terms = {
        "sr": "sustained_release", "sustained": "sustained_release", "sustainedrelease": "sustained_release",
        "er": "extended_release", "xr": "extended_release", "extended": "extended_release", "extendedrelease": "extended_release",
        "cr": "controlled_release", "controlled": "controlled_release", "controlledrelease": "controlled_release",
        "dr": "delayed_release", "delayed": "delayed_release", "enteric": "enteric_coated", "gastro": "gastro_resistant",
        "chewable": "chewable", "dispersible": "dispersible", "effervescent": "effervescent",
    }
    qualifiers = sorted({release_terms[token] for token in tokens if token in release_terms})
    return ":".join((form, *qualifiers))


def _pack_key(value: str | None, dosage_form: str | None = None) -> tuple[str, str] | None:
    if not value:
        return None
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*(tablets?|tabs?|capsules?|caps?|units?|pieces?|pcs?|ml|g|grams?|sachets?|vials?|ampoules?)\b", value, re.I)
    if not match:
        count = re.search(r"\b(\d+(?:\.\d+)?)\s*['’]s\b", value, re.I)
        form = _form_key(dosage_form or "")
        if count and form in ("tablet", "capsule"):
            return str(Decimal(count.group(1)).normalize()), "tablets" if form == "tablet" else "capsules"
    if not match:
        return None
    amount = Decimal(match.group(1)).normalize()
    unit = match.group(2).lower()
    if unit.startswith(("tablet", "tab")): unit = "tablets"
    elif unit.startswith(("capsule", "cap")): unit = "capsules"
    elif unit in ("piece", "pieces", "pcs"): unit = "units"
    elif unit == "gram": unit = "g"
    elif unit == "grams": unit = "g"
    return str(amount), unit


def _parse_page(page: dict, kind: str, search_item: dict) -> dict:
    text = str(page.get("text") or page.get("content") or "")
    title = str(page.get("title") or search_item.get("title") or "")
    product_name = _product_title(page, title, text)
    combined = f"{product_name or ''}\n{text}"
    ingredient_evidence = _label(text, r"active ingredients?|ingredients?|composition|generic name|salt composition|salt content")
    ingredients = ingredient_evidence
    if ingredients:
        ingredients = re.sub(r"\(\s*\d+(?:\.\d+)?\s*(?:mcg|μg|µg|mg|g|ml|iu|%)\s*\)", "", ingredients, flags=re.I).strip()
    strength = _label(text, r"strength|composition strength")
    if not strength and ingredients:
        strength_values = re.findall(r"\d+(?:\.\d+)?\s*(?:mcg|μg|µg|mg|g|ml|iu|%)", ingredient_evidence or "", re.I)
        strength = ", ".join(strength_values) or None
    dosage_form = _label(text, r"dosage form|product form") or _form_key(product_name or "")
    pack_size = _label(text, r"pack size|package size|quantity|contents") or _extract_pack_size(text, dosage_form)
    if not strength:
        strength_match = re.search(r"\b\d+(?:\.\d+)?\s*(?:mcg|μg|µg|mg|g|ml|iu|%)\b", product_name or "", re.I)
        strength = strength_match.group(0) if strength_match else None
    price = _price(combined, kind)
    source_url = page.get("final_url") or page.get("url") or search_item.get("url")
    host = _domain(source_url) if isinstance(source_url, str) else ""
    explicit_brand = _label(text, r"brand(?: name)?|trade name|product brand")
    brand_name = (explicit_brand or _infer_brand_name(product_name, ingredients)) if kind == "brand" else None
    return {
        "name": product_name,
        "ingredients": ingredients,
        "strength": strength,
        "dosage_form": dosage_form,
        "pack_size": pack_size,
        # Do not infer package quantity from a strength-bearing title such as
        # "Alerid 10 Tablet"; only an explicitly extracted package description is evidence.
        "pack_key": _pack_key(pack_size, dosage_form or product_name),
        "ingredient_key": _ingredient_key(ingredients),
        "strength_key": _strength_key(strength),
        "form_key": _form_key(f"{dosage_form or ''} {title}"),
        "price": price,
        "source_url": source_url if isinstance(source_url, str) else None,
        "source_date": _source_date(page),
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trusted": _trusted_source(source_url, kind) if isinstance(source_url, str) else False,
        "brand_name": brand_name,
        "site_name": search_item.get("site_name") or host,
    }


def _pmbi_ingredient_key(description: str) -> tuple[str, ...]:
    """Extract named active ingredients from the official PMBI product description."""
    text = description.lower()
    form_match = re.search(r"\b(?:tablets?|tabs?|capsules?|caps?|syrup|suspension|injection|cream|ointment|drops?|gel|solution|powder)\b", text)
    if form_match:
        text = text[:form_match.start()]
    text = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mcg|μg|µg|mg|g|ml|iu|%)\b", " ", text)
    text = re.sub(r"\b(?:ip|bp|usp|each|contains|uncoated|film|coated|hard|soft|gelatin|enteric|release|sustained|extended|controlled|delayed)\b", " ", text)
    return _ingredient_key(text)


class _PMBIFormAndTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden_fields: dict[str, str] = {}
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag.lower() == "input" and attributes.get("type", "").lower() == "hidden":
            name = attributes.get("name")
            if name:
                self.hidden_fields[name] = attributes.get("value") or ""
        elif tag.lower() == "tr":
            self._row = []
        elif tag.lower() in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(re.sub(r"\s+", " ", " ".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def _target_matches_pmbi(description: str, generic_name: str, strength: str, dosage_form: str) -> bool:
    expected_ingredients = _ingredient_key(generic_name)
    found_ingredients = _pmbi_ingredient_key(description)
    return bool(
        expected_ingredients and expected_ingredients == found_ingredients
        and _strength_key(strength) and _strength_key(strength) == _strength_key(description)
        and _form_key(dosage_form) and _form_key(dosage_form) == _form_key(description)
    )


def _extract_pmbi_table_rows(html: str, generic_name: str, strength: str, dosage_form: str) -> list[dict]:
    parser = _PMBIFormAndTableParser()
    parser.feed(html)
    header_index = next((i for i, row in enumerate(parser.rows) if len(row) >= 5 and "generic name" in " ".join(row).lower() and "unit size" in " ".join(row).lower() and "mrp" in " ".join(row).lower()), None)
    if header_index is None:
        return []
    records = []
    for row in parser.rows[header_index + 1:]:
        if len(row) < 5:
            continue
        product_description, unit_size, raw_price = row[2], row[3], row[4]
        if not _target_matches_pmbi(product_description, generic_name, strength, dosage_form):
            continue
        price = _money_value(raw_price)
        if not price:
            continue
        form = _form_key(dosage_form)
        normalized_form = "tablets" if form == "tablet" else "capsules" if form == "capsule" else dosage_form
        pack_key = _pack_key(unit_size, dosage_form)
        records.append({
            "name": product_description,
            "product_description": product_description,
            "ingredients": generic_name,
            "strength": strength,
            "dosage_form": normalized_form,
            "pack_size": unit_size or None,
            "pack_key": pack_key,
            "ingredient_key": _ingredient_key(generic_name),
            "strength_key": _strength_key(strength),
            "form_key": _form_key(dosage_form),
            "price": price,
            "source_url": PMBI_PRODUCT_LIST_URL,
            "source_date": None,
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "trusted": True,
            "brand_name": None,
            "site_name": "pmbi.co.in",
        })
    return records


async def _search_pmbi_product_list(client: httpx.AsyncClient, generic_name: str, strength: str, dosage_form: str) -> list[dict]:
    first_ingredient = re.split(r"\s*(?:\+|,|\band\b)\s*", generic_name, maxsplit=1, flags=re.I)[0]
    search_term = re.sub(r"[^A-Za-z0-9]", "", first_ingredient)
    if not search_term:
        log_event("jan_source", source="pmbi_product_list", status="unavailable", result_count=0, extraction_status="skipped", reason="missing_active_ingredient")
        return []
    try:
        page = await client.get(PMBI_PRODUCT_LIST_URL)
        page.raise_for_status()
        form = _PMBIFormAndTableParser()
        form.feed(page.text)
        if not form.hidden_fields:
            log_event("jan_source", source="pmbi_product_list", status="unavailable", result_count=0, extraction_status="failed", reason="form_state_not_found")
            return []
        response = await client.post(
            PMBI_PRODUCT_LIST_URL,
            data={**form.hidden_fields, "ctl00$Bppi_body$txtSearch": search_term, "ctl00$Bppi_body$btnSearch": "Search"},
            headers={"Referer": PMBI_PRODUCT_LIST_URL},
        )
        response.raise_for_status()
        records = _extract_pmbi_table_rows(response.text, generic_name, strength, dosage_form)
        log_event(
            "jan_source", source="pmbi_product_list", status="success" if records else "no_match",
            result_count=len(records), extraction_status="extracted" if records else "no_matching_product",
            reason=None if records else "no_exact_ingredient_strength_form_match",
        )
        return records
    except httpx.HTTPStatusError as exc:
        log_event("jan_source", source="pmbi_product_list", status="unavailable", result_count=0, extraction_status="failed", reason=f"http_{exc.response.status_code}")
        return []
    except (httpx.RequestError, ValueError, TypeError):
        log_event("jan_source", source="pmbi_product_list", status="unavailable", result_count=0, extraction_status="failed", reason="request_or_parse_error")
        return []


def _comparable(brand: dict, jan: dict) -> bool:
    return bool(
        brand.get("trusted") and jan.get("trusted")
        and brand.get("price") and jan.get("price")
        and brand.get("ingredient_key") and brand.get("ingredient_key") == jan.get("ingredient_key")
        and brand.get("strength_key") and brand.get("strength_key") == jan.get("strength_key")
        and brand.get("form_key") and brand.get("form_key") == jan.get("form_key")
        and brand.get("pack_key") and jan.get("pack_key")
        and brand["pack_key"][1] == jan["pack_key"][1]
    )


def _price_for_pack(record: dict, target_pack: tuple[str, str]) -> Decimal:
    price = record["price"]
    source_pack = record["pack_key"]
    if source_pack == target_pack:
        return price
    source_count = Decimal(source_pack[0])
    target_count = Decimal(target_pack[0])
    return (price * target_count / source_count).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def _candidate_urls(items: list[dict], kind: str) -> list[dict]:
    candidates = [item for item in items if _trusted_source(item.get("url", ""), kind)]
    candidates.sort(key=lambda item: (
        0 if kind == "jan" and item["url"] in JAN_PORTFOLIO_URLS else
        1 if _domain(item["url"]).endswith(".gov.in") or _is_on_domain(_domain(item["url"]), ("pmbi.co.in", "janaushadhi.gov.in")) else 2
    ))
    return candidates[:FETCH_LIMIT]


async def _retrieve_kind(client: httpx.AsyncClient, key: str, items: list[dict], kind: str) -> list[dict]:
    candidates = _candidate_urls(items, kind)
    pages = await _fetch(client, key, [item["url"] for item in candidates])
    by_url = {page.get("url"): page for page in pages if isinstance(page, dict)}
    parsed = []
    for item in candidates:
        page = by_url.get(item["url"])
        if page:
            result = _parse_page(page, kind, item)
            if result["source_url"]:
                parsed.append(result)
    parsed.sort(key=lambda result: result.get("price") is None)
    return parsed


async def _retrieve_jan_alternative_sources(
    client: httpx.AsyncClient,
    key: str,
    items: list[dict],
    generic_name: str,
    strength: str,
    dosage_form: str,
) -> list[dict]:
    candidates = _candidate_urls(items, "jan")
    pages = await _fetch(client, key, [item["url"] for item in candidates])
    page_by_url = {page.get("url"): page for page in pages if isinstance(page, dict)}
    records = []
    for item in candidates:
        page = page_by_url.get(item["url"])
        if not page:
            log_event("jan_source", source=item["url"], status="unavailable", result_count=0, extraction_status="failed", reason="fetch_returned_no_page")
            continue
        text = str(page.get("text") or page.get("content") or "")
        if not text.strip():
            log_event("jan_source", source=page.get("final_url") or item["url"], status="unavailable", result_count=0, extraction_status="failed", reason="unreadable_or_empty_content")
            continue
        record = _parse_page(page, "jan", item)
        exact_match = (
            record.get("ingredient_key") == _ingredient_key(generic_name)
            and record.get("strength_key") == _strength_key(strength)
            and record.get("form_key") == _form_key(dosage_form)
        )
        # PDF tender documents commonly contain bid values but not the published
        # retail MRP. Accept a PDF price only when it is explicitly labelled as MRP.
        is_pdf = urlparse(str(page.get("final_url") or item["url"])).path.lower().endswith(".pdf")
        explicit_pdf_mrp = bool(re.search(r"(?i)\bMRP\b[^\n]{0,35}(?:₹|Rs\.?\s*)?\d+(?:,\d{3})*(?:\.\d{1,2})?", text))
        if not exact_match or not record.get("price") or (is_pdf and not explicit_pdf_mrp):
            reason = "attribute_mismatch" if not exact_match else "official_mrp_not_extracted"
            log_event("jan_source", source=page.get("final_url") or item["url"], status="unavailable", result_count=1, extraction_status="rejected", reason=reason)
            continue
        records.append(record)
        log_event("jan_source", source=page.get("final_url") or item["url"], status="success", result_count=1, extraction_status="extracted", reason=None)
    return records


def _public_price(value: Decimal | None) -> float | None:
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)) if value is not None else None


def _empty_savings(message: str) -> dict:
    return {"available": False, "savings_per_pack": None, "savings_percentage": None, "currency": "INR", "comparison_status": "unavailable", "message": message}


def _public_medicine(
    brand: dict | None,
    jan: dict | None,
    query: str,
    match: bool,
    jan_sources_checked: list[str] | None = None,
) -> dict:
    brand = brand or {}
    jan = jan or {}
    retrieval_date = datetime.now(timezone.utc).date().isoformat()
    matched = bool(match)
    savings = _empty_savings("Savings are unavailable because a trustworthy, like-for-like comparison could not be established.")
    jan_comparable_price = None
    if matched:
        jan_comparable_price = _price_for_pack(jan, brand["pack_key"])
        diff = (brand["price"] - jan_comparable_price).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if diff >= 0:
            percentage = (diff * Decimal(100) / brand["price"]).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            pack_normalized = brand["pack_key"] != jan["pack_key"]
            savings = {
                "available": True,
                "savings_per_pack": float(diff),
                "savings_percentage": float(percentage),
                "currency": "INR",
                "comparison_status": "normalized_to_branded_pack_size" if pack_normalized else "retrieved_sources_comparable",
                "message": (
                    "Jan Aushadhi price was normalized to the branded pack size using both published unit counts. "
                    "Original listed prices and pack sizes are shown below; sources were not independently audited."
                    if pack_normalized else
                    "Potential savings per comparable pack; prices were retrieved from linked sources and were not independently audited."
                ),
            }
        else:
            savings = _empty_savings("The Jan Aushadhi price is not lower than the branded MRP.")
    verification = (
        "Retrieved from trustworthy sources; ingredients, strength, dosage form, and pack size matched automatically. Prices were not independently audited."
        if matched else "Live source results retrieved; product equivalence or both prices could not be established."
    )
    brand_status = (
        "price_basis_unverified" if brand.get("price") and not brand.get("pack_key")
        else "retrieved_from_trustworthy_source_not_independently_audited" if brand.get("price")
        else "price_not_found_on_source" if brand.get("source_url")
        else "unavailable"
    )
    jan_status = (
        "pack_size_missing" if jan.get("price") and not jan.get("pack_key")
        else "retrieved_from_trustworthy_source_not_independently_audited" if jan.get("price")
        else "unavailable"
    )
    return {
        "name": brand.get("name") or jan.get("name") or query,
        "product_name": brand.get("name") or jan.get("product_description") or jan.get("name") or query,
        "ingredients": brand.get("ingredients") or jan.get("ingredients"),
        "branded_product_name": brand.get("brand_name"),
        "strength": brand.get("strength") or jan.get("strength"),
        "dosage_form": brand.get("dosage_form") or jan.get("dosage_form"),
        "pack_size": brand.get("pack_size"),
        "branded_unit_size": brand.get("pack_size"),
        "market_mrp": _public_price(brand.get("price")),
        "market_price_status": brand_status,
        "market_price_source_url": brand.get("source_url"),
        "market_price_date": brand.get("source_date"),
        "jan_aushadhi_price": _public_price(jan.get("price")),
        "jan_comparable_pack_price": _public_price(jan_comparable_price),
        "jan_price_status": jan_status,
        "jan_price_reason": None if jan.get("price") else "no_verified_official_mrp_found",
        "jan_price_date": jan.get("source_date"),
        "jan_price_source_url": jan.get("source_url"),
        "jan_price_sources_checked": jan_sources_checked or [],
        "jan_product_description": jan.get("product_description") or jan.get("name"),
        "jan_ingredients": jan.get("ingredients"),
        "jan_strength": jan.get("strength"),
        "jan_dosage_form": jan.get("dosage_form"),
        "jan_unit_size": jan.get("pack_size"),
        "comparison_pack_size": brand.get("pack_size") if matched else None,
        "jan_price_retrieved_at": jan.get("retrieved_at"),
        "branded_mrp": _public_price(brand.get("price")),
        "branded_mrp_status": brand_status,
        "branded_mrp_date": brand.get("source_date"),
        "branded_mrp_source_url": brand.get("source_url"),
        "net_savings": savings,
        "verification_status": verification,
        "as_of_date": retrieval_date,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "branded_mrp_retrieved_at": brand.get("retrieved_at"),
        "jan_price_retrieved_at": jan.get("retrieved_at"),
    }


async def search_medicine(
    query: str,
    api_key: str,
    search_queries: list[str] | None = None,
    *,
    generic_name: str | None = None,
    strength: str | None = None,
    dosage_form: str | None = None,
    brand_name: str | None = None,
) -> dict:
    if search_queries is None:
        brand_query = f'{query} medicine manufacturer MRP composition strength dosage form pack size India'
    else:
        if not search_queries:
            raise ValueError("At least one branded medicine search query is required")
        brand_query = search_queries[0]
    has_jan_attributes = bool(generic_name and strength and dosage_form)
    if has_jan_attributes:
        jan_query = (
            f"{generic_name} {strength} {dosage_form} PMBJP Jan Aushadhi MRP "
            "official PMBI Product MRP List unit size pack quantity"
        )
    else:
        jan_query = f"{generic_name or query} {strength or ''} {dosage_form or ''} Jan Aushadhi PMBI official product list MRP unit size".strip()
    log_event("tinyfish_search", status="started", query_count=3 if has_jan_attributes else 2, jan_strategy="official_pmbi_targeted")
    try:
        async with make_tinyfish_client() as client:
            brand_results, jan_results = await asyncio.gather(
                _search(client, api_key, brand_query),
                _search(client, api_key, jan_query, JAN_OFFICIAL_DOMAINS),
            )
            brand_results = [item for item in brand_results if _trusted_source(item["url"], "brand")]
            jan_results = [item for item in jan_results if _trusted_source(item["url"], "jan")]
            initial_jan_result_count = len(jan_results)
            if has_jan_attributes:
                pack_term = "strip pack quantity" if _form_key(dosage_form) in ("tablet", "capsule") else "pack unit quantity"
                brand_target_query = (
                    f"{generic_name} {strength} {dosage_form} {pack_term} "
                    "MRP manufacturer India"
                )
                targeted_brand_results = await _search(client, api_key, brand_target_query)
                targeted_brand_results = [item for item in targeted_brand_results if _trusted_source(item["url"], "brand")]
                brand_results = list({item["url"]: item for item in brand_results + targeted_brand_results}.values())
                log_event(
                    "tinyfish_search", status="success" if targeted_brand_results else "no_match",
                    strategy="generic_strength_form_pack_targeted",
                    result_count=len(targeted_brand_results),
                    extraction_status="pending_fetch" if targeted_brand_results else "not_started",
                    reason=None if targeted_brand_results else "no_trusted_product_search_results",
                )
            log_event(
                "jan_source", source="tinyfish_search", status="success" if jan_results else "no_match",
                result_count=len(jan_results), extraction_status="pending_fetch" if jan_results else "not_started",
                reason=None if jan_results else "no_official_search_results",
            )
            log_event("tinyfish_search", status="success", branded_candidates=len(brand_results), jan_candidates=len(jan_results))
            brand_records = await _retrieve_kind(client, api_key, brand_results, "brand")
            if brand_name:
                candidate_count = len(brand_records)
                brand_records = [record for record in brand_records if _brand_matches(brand_name, record)]
                for record in brand_records:
                    if not record.get("brand_name"):
                        record["brand_name"] = brand_name
                log_event(
                    "brand_validation",
                    status="matched" if brand_records else "no_match",
                    requested_brand=brand_name,
                    candidate_count=candidate_count,
                    matching_count=len(brand_records),
                    reason=None if brand_records else "requested_brand_not_found_in_product_evidence",
                )
            if has_jan_attributes and brand_records and brand_records[0].get("pack_size"):
                jan_pack_query = (
                    f"{generic_name} {strength} {dosage_form} {brand_records[0]['pack_size']} "
                    "Jan Aushadhi PMBI official MRP product list unit size"
                )
                pack_results = await _search(client, api_key, jan_pack_query, JAN_OFFICIAL_DOMAINS)
                pack_results = [item for item in pack_results if _trusted_source(item["url"], "jan")]
                jan_results.extend(pack_results)
                log_event(
                    "jan_source", source="tinyfish_search_pack_targeted",
                    status="success" if pack_results else "no_match", result_count=len(pack_results),
                    extraction_status="pending_fetch" if pack_results else "not_started",
                    reason=None if pack_results else "no_official_pack_specific_search_results",
                )
            # Explicitly fetch the live official listing/portfolio pages even when
            # Search returns only a generic portal result or no indexed product row.
            for source_url in (PMBI_PRODUCT_LIST_URL, *JAN_PORTFOLIO_URLS):
                if not any(item.get("url") == source_url for item in jan_results):
                    jan_results.append({"url": source_url, "title": "Official PMBI/Jan Aushadhi product listing"})
            jan_results = list({item["url"]: item for item in jan_results}.values())
            jan_document_records = (
                await _retrieve_jan_alternative_sources(
                    client, api_key, jan_results, generic_name or "", strength or "", dosage_form or ""
                ) if has_jan_attributes else await _retrieve_kind(client, api_key, jan_results, "jan")
            )
            log_event("tinyfish_fetch", status="success", pages_fetched=len(brand_records) + len(jan_document_records))
            if has_jan_attributes:
                listed_pages = await _fetch(client, api_key, [PMBI_PRODUCT_LIST_URL])
                listed_page = next((page for page in listed_pages if page.get("url") == PMBI_PRODUCT_LIST_URL), None)
                listed_text = str((listed_page or {}).get("text") or (listed_page or {}).get("content") or "")
                listed_records = _extract_pmbi_table_rows(listed_text, generic_name, strength, dosage_form) if listed_text else []
                if not listed_page:
                    listed_status, listed_extraction, listed_reason = "unavailable", "failed", "fetch_returned_no_page"
                elif not listed_text.strip():
                    listed_status, listed_extraction, listed_reason = "unavailable", "failed", "empty_page_content"
                elif listed_records:
                    listed_status, listed_extraction, listed_reason = "success", "extracted", None
                else:
                    listed_status, listed_extraction, listed_reason = "no_product_data", "dynamic_search_requires_postback", "static_fetch_contains_search_form_without_results"
                log_event(
                    "jan_source", source=PMBI_PRODUCT_LIST_URL,
                    status=listed_status,
                    result_count=len(listed_records),
                    extraction_status=listed_extraction,
                    reason=listed_reason,
                )
                async with make_pmbi_client() as pmbi_client:
                    pmbi_records = await _search_pmbi_product_list(pmbi_client, generic_name, strength, dosage_form)
                pmbi_records = pmbi_records or listed_records
            else:
                pmbi_records = []
    except TinyFishAuthenticationError:
        log_event("tinyfish_error", status="authentication_error")
        raise
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            log_event("tinyfish_error", status="authentication_error")
            raise TinyFishAuthenticationError from None
        log_event("tinyfish_error", status="upstream_error", http_status=exc.response.status_code)
        raise TinyFishUnavailableError from None
    except (httpx.RequestError, ValueError, TypeError, KeyError):
        log_event("tinyfish_error", status="upstream_error")
        raise TinyFishUnavailableError from None

    jan_records = pmbi_records or jan_document_records
    jan_sources_checked = list(dict.fromkeys(
        [PMBI_PRODUCT_LIST_URL, *JAN_PORTFOLIO_URLS]
        + [item["url"] for item in jan_results]
    ))
    log_event(
        "jan_source_summary", status="unavailable" if not jan_records else "candidate_records_found",
        sources_checked=jan_sources_checked,
        tinyfish_search_results=initial_jan_result_count,
        extracted_records=len(jan_records),
        extraction_status="no_verified_product_price" if not jan_records else "records_validated",
    )
    if brand_name and not brand_records:
        log_event("validation", status="unavailable", reason="requested_brand_not_found")
        log_event("savings_calculation", status="unavailable", reason="Requested brand was not verified")
        return {
            "status": "not_found",
            "message": f"No trustworthy listing for the requested brand {brand_name} was found. Try another brand or search by the generic ingredient.",
            "jan_price_status": "unavailable" if not jan_records else "available_without_brand_match",
            "jan_price_sources_checked": jan_sources_checked,
        }
    if not jan_records:
        log_event("jan_price", status="unavailable", source="pmbi.co.in", reason="no_official_product_price_extracted")
    elif not any(item.get("pack_key") for item in jan_records):
        log_event("jan_price", status="partial", source="pmbi.co.in", reason="pack_size_missing")
    else:
        log_event("jan_price", status="success", source="pmbi.co.in", result_count=len(jan_records))
    pair = next(((brand, jan) for brand in brand_records for jan in jan_records if _comparable(brand, jan)), None)
    if pair:
        log_event("validation", status="comparable", reason="ingredients_strength_form_and_pack_match")
        log_event("savings_calculation", status="success", currency="INR")
        return {"status": "matched", "match_source": "tinyfish", "requires_confirmation": True, "medicine": _public_medicine(pair[0], pair[1], query, True, jan_sources_checked)}

    brand = brand_records[0] if brand_records else None
    jan = jan_records[0] if jan_records else None
    if brand or jan:
        log_event("validation", status="partial", reason="Comparable product or both prices not established")
        log_event("savings_calculation", status="unavailable", reason="Insufficient comparable price evidence")
        return {"status": "matched", "match_source": "tinyfish", "requires_confirmation": True, "medicine": _public_medicine(brand, jan, query, False, jan_sources_checked)}
    log_event("validation", status="unavailable", reason="No trustworthy price evidence")
    log_event("savings_calculation", status="unavailable", reason="Insufficient comparable price evidence")
    return {
        "status": "not_found",
        "message": "No trustworthy source with a usable medicine price was found. Try a more specific brand name or strength.",
        "jan_price_status": "unavailable",
        "jan_price_reason": "no_verified_official_mrp_found",
        "jan_price_sources_checked": jan_sources_checked,
    }

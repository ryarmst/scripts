#!/usr/bin/env python3
"""
contentful_graphql_dump.py
--------------------------
Comprehensively dumps all GraphQL data from a Contentful space.
Schema-driven: behaviour is determined by introspection, not hard-coded
type names or field assumptions.

Improvements over the v2 dumper:
  * Adaptive page sizing on TOO_COMPLEX_QUERY (halves and retries before
    falling back to sys-only). This was the bug that turned ~8 600 entries
    into sys.id-only rows in the original dumps.
  * Selection builder skips nested xxxCollection AND xxxCursorCollection
    at every depth (cursor variant is pure dead weight; collection variant
    is recovered via a per-item second pass).
  * Optional second-pass per-item links/references dump with explicit
    small `limit:` values, so we don't lose graph edges.
  * Rich-text body { json links { ... } } selection (opt-in via
    --rich-text-links) - surfaces embedded entry / asset IDs.
  * Exponential backoff with jitter on 429 / 5xx, respects
    X-Contentful-RateLimit-Reset.
  * Captures and logs X-Contentful-Graphql-Query-Cost per request.
  * Token may be passed via --token-env <ENV_VAR_NAME> to avoid leaking
    via process listings / shell history.
  * Selection field names are sanitised against a strict GraphQL-name
    regex (defence in depth against a malicious schema response).
  * Per-collection final adaptive page size is recorded in the summary
    so a follow-up run can start from a known-good size.

Usage:
    python3 contentful_graphql_dump.py --space <SPACE_ID>
        --token <TOKEN> | --token-env CTFL_TOKEN [options]

Output layout (same as before, with a few additions):
    <out>/<space>_<env>_<timestamp>/
        introspection_raw.json          Raw introspection response
        schema_summary.json             Parsed schema summary
        environments_probe.json         Environment enumeration results
        security_checks.json            Token scope / endpoint checks
        collections/
            <TypeName>.json             Paginated collection dump
        collection_links/               (optional, with --link-pass)
            <TypeName>.json             Per-item nested-ref expansion
        singletons/
            <field_name>.json
        singletons_by_id/
            <field_name>.json
        _summary.json                   Run manifest
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime

try:
    import requests
except ImportError:
    sys.exit("[!] Install requests: pip install requests")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GQL_URL     = "https://graphql.contentful.com/content/v1/spaces/{space}/environments/{env}"
PREVIEW_URL = "https://graphql.preview.contentful.com/content/v1/spaces/{space}/environments/{env}"
CMA_URL     = "https://api.contentful.com/spaces/{space}/environments/{env}/entries?limit=1"

COMMON_ENVIRONMENTS = [
    "master", "staging", "stage", "dev", "development",
    "preview", "qa", "test", "uat", "release", "production",
    "sandbox", "demo", "main", "next", "v2",
]

# GraphQL kinds that can be selected directly as scalar values
SCALAR_KINDS = {"SCALAR", "ENUM"}

# Fields that are pure system/meta overhead with no data value
SKIP_FIELDS = {
    "contentfulMetadata", "linkedFrom", "_id",
}

# Rich-text / JSON body fields resolve to a wrapper OBJECT with a "json" sub-field.
RICH_TEXT_WRAPPER_SUFFIXES = (
    "Body", "Description", "Answer", "Text", "Content",
    "Headline", "Summary", "Copy", "Detail",
)

# Contentful GraphQL error codes that are non-fatal -- they indicate dirty
# data (broken refs, missing locales, etc.) but the response still contains
# valid items.
NON_FATAL_GQL_CODES = {
    "UNRESOLVABLE_LINK",
    "UNRESOLVABLE_RESOURCE_LINK",
    "UNEXPECTED_LINKED_CONTENT_TYPE",
    "TYPE_NOT_FOUND",
}

# Complexity / size codes that mean "the SHAPE of this query is too heavy".
# These warrant adaptive page-size retries before falling back to sys-only.
COMPLEXITY_GQL_CODES = {
    "TOO_COMPLEX_QUERY",
    "QUERY_TOO_BIG",
    "RESPONSE_TOO_BIG",
}

# Strict GraphQL name pattern (https://spec.graphql.org/October2021/#Name).
# Used to sanitise field and type names before stitching them into queries.
NAME_RE = re.compile(r"^[_A-Za-z][_0-9A-Za-z]*$")

# Default cap on per-request retries when the server signals a transient
# failure (429, 5xx, network blip).
DEFAULT_HTTP_RETRIES = 5

# Full introspection with deep TypeRef fragment.
INTROSPECTION_QUERY = """{
  __schema {
    queryType { name }
    mutationType { name }
    subscriptionType { name }
    types {
      kind
      name
      description
      fields(includeDeprecated: true) {
        name
        description
        isDeprecated
        deprecationReason
        type { ...TR }
        args {
          name
          type { ...TR }
          defaultValue
        }
      }
      inputFields {
        name
        type { ...TR }
        defaultValue
      }
      enumValues(includeDeprecated: true) { name }
      possibleTypes { name kind }
    }
  }
}
fragment TR on __Type {
  kind name
  ofType {
    kind name
    ofType {
      kind name
      ofType {
        kind name
        ofType {
          kind name
          ofType { kind name ofType { kind name ofType { kind name } } }
        }
      }
    }
  }
}"""


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _backoff_sleep(attempt: int, base: float = 0.5, cap: float = 30.0,
                   reset_hint: float = 0.0) -> float:
    """
    Compute a sleep duration for retry attempt `attempt` (1-indexed).
    Honours Contentful's X-Contentful-RateLimit-Reset hint if provided
    (seconds), otherwise uses decorrelated jitter exponential backoff.
    Returns the slept seconds (useful for logging).
    """
    if reset_hint > 0:
        # Server told us when to retry. Add a touch of jitter so a fleet
        # of clients doesn't synchronise.
        delay = reset_hint + random.uniform(0, 0.25)
    else:
        # Decorrelated jitter: rand(base, prev * 3) capped at `cap`
        prev = base * (2 ** max(0, attempt - 1))
        delay = min(cap, random.uniform(base, max(base, prev * 3)))
    time.sleep(delay)
    return delay


def gql(url: str, token: str, query: str, delay: float = 0.0,
        max_retries: int = DEFAULT_HTTP_RETRIES) -> dict:
    """
    POST a GraphQL query.  Returns:
      { _status, _cost, _request_id, data?, errors? }
    or { _error: str } on a final transport failure.

    Retries automatically on 429 and 5xx with exponential backoff
    (honouring Retry-After / X-Contentful-RateLimit-Reset when present).
    """
    if delay:
        time.sleep(delay)

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type":  "application/json",
                    "User-Agent":    "contentful-dump/3.0",
                },
                json={"query": query},
                timeout=60,
            )
        except requests.RequestException as e:
            last_exc = e
            # Network blip -> backoff and retry, but only up to max_retries.
            if attempt >= max_retries:
                return {"_error": f"{type(e).__name__}: {e}"}
            _backoff_sleep(attempt)
            continue

        cost       = r.headers.get("X-Contentful-Graphql-Query-Cost")
        request_id = r.headers.get("X-Contentful-Request-Id")
        reset      = r.headers.get("X-Contentful-RateLimit-Reset") \
                     or r.headers.get("Retry-After")
        try:
            reset_hint = float(reset) if reset else 0.0
        except ValueError:
            reset_hint = 0.0

        # 429 / 5xx -> retry with backoff
        if r.status_code == 429 or 500 <= r.status_code < 600:
            if attempt < max_retries:
                slept = _backoff_sleep(attempt, reset_hint=reset_hint)
                print(f"        [retry {attempt}/{max_retries}] HTTP {r.status_code} "
                      f"req={request_id} -> slept {slept:.2f}s")
                continue
            # Out of retries - fall through to return the response so the
            # caller can record it.

        # Parse body. Some 4xx responses still have a JSON body with
        # `errors` (e.g. TOO_COMPLEX_QUERY arrives as HTTP 400). Be tolerant.
        try:
            body = r.json()
        except ValueError:
            body = {"_raw_text": r.text[:2000]}

        try:
            cost_i = int(cost) if cost is not None else None
        except ValueError:
            cost_i = None

        return {
            "_status":     r.status_code,
            "_cost":       cost_i,
            "_request_id": request_id,
            **body,
        }

    # Should not get here, but for completeness:
    return {"_error": f"exhausted retries: {last_exc}"}


def http_get(url: str, token: str, delay: float = 0.0,
             max_retries: int = DEFAULT_HTTP_RETRIES) -> dict:
    """GET with the same retry policy. Used only for security probes."""
    if delay:
        time.sleep(delay)

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "User-Agent":    "contentful-dump/3.0",
                },
                timeout=20,
            )
        except requests.RequestException as e:
            if attempt >= max_retries:
                return {"_error": f"{type(e).__name__}: {e}"}
            _backoff_sleep(attempt)
            continue

        reset = r.headers.get("X-Contentful-RateLimit-Reset") \
                or r.headers.get("Retry-After")
        try:
            reset_hint = float(reset) if reset else 0.0
        except ValueError:
            reset_hint = 0.0

        if r.status_code == 429 or 500 <= r.status_code < 600:
            if attempt < max_retries:
                _backoff_sleep(attempt, reset_hint=reset_hint)
                continue

        return {"_status": r.status_code, "_body": r.text[:2000]}

    return {"_error": "exhausted retries"}


def save(path: str, data) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# Type resolution helpers
# ---------------------------------------------------------------------------

def unwrap_type(t: dict) -> dict:
    """Peel NON_NULL and LIST wrappers to reach the named base type."""
    while t and t.get("kind") in ("NON_NULL", "LIST"):
        t = t.get("ofType")
    return t or {}


def base_kind(t: dict) -> str:
    return unwrap_type(t).get("kind", "")


def base_name(t: dict) -> str:
    return unwrap_type(t).get("name", "")


def safe_name(name: str) -> str:
    """
    Return `name` if it matches the GraphQL-name regex, else "".
    Used as a defence-in-depth check: an attacker who controls the schema
    response cannot inject raw GraphQL via a crafted field name.
    """
    if isinstance(name, str) and NAME_RE.match(name):
        return name
    return ""


# ---------------------------------------------------------------------------
# Schema parsing
# ---------------------------------------------------------------------------

def parse_schema(raw: dict) -> tuple:
    """
    Returns:
        query_root_fields   list[str]  - field names on the Query type
        types_by_name       dict       - name -> full type object
    """
    schema = (raw.get("data") or {}).get("__schema") or {}
    query_type_name   = (schema.get("queryType") or {}).get("name", "Query")
    types_by_name     = {}
    query_root_fields = []

    for t in schema.get("types") or []:
        name = t.get("name", "")
        if not name or name.startswith("__"):
            continue
        types_by_name[name] = t
        if name == query_type_name:
            query_root_fields = [f["name"] for f in (t.get("fields") or [])]

    return query_root_fields, types_by_name


# ---------------------------------------------------------------------------
# Selection set builder
# ---------------------------------------------------------------------------

def is_collection_typename(name: str) -> bool:
    """True for both xxxCollection and xxxCursorCollection variants."""
    return bool(name) and (name.endswith("Collection")
                           or name.endswith("CursorCollection"))


def is_cursor_collection_typename(name: str) -> bool:
    return bool(name) and name.endswith("CursorCollection")


def build_selection(type_name: str, types_by_name: dict, depth: int = 0,
                    visited: set = None,
                    rich_text_links: bool = False,
                    nested_collections: list = None) -> str:
    """
    Recursively build a GraphQL selection set string for a given OBJECT type.

    Rules
    -----
    * Scalars and enums are always selected.
    * `sys` always expands to the standard sub-selection.
    * Asset objects expand to their useful scalar fields.
    * Rich-text wrapper objects (detected by suffix + presence of `json` field)
      expand as `fieldName { json }`, or `fieldName { json links { ... } }`
      when --rich-text-links is enabled.
    * **Nested xxxCollection AND xxxCursorCollection sub-fields are skipped
      at every depth.** The original v2 builder skipped them only at
      depth >= 1, which meant item-level (depth 0) nested collections were
      inlined - and four or more of them at default limit 100 each blew
      through Contentful's 100,101 complexity ceiling. Those references
      are recovered via a second-pass per-item query (see
      dump_collection_links).  If `nested_collections` is provided, the
      names of skipped collection fields are appended for the caller.
    * Other OBJECT fields are recursed into up to depth 2, then dropped.
    * Union/Interface fields get `{ __typename }` only.
    * Cycle guard via `visited` set.
    * Any field with a non-GraphQL-name is dropped (defence in depth).
    """
    if visited is None:
        visited = set()
    if type_name in visited:
        return ""
    type_name_safe = safe_name(type_name)
    if not type_name_safe:
        return ""
    visited = visited | {type_name_safe}

    t = types_by_name.get(type_name_safe)
    if not t:
        return ""

    parts = []

    for f in (t.get("fields") or []):
        fname = safe_name(f.get("name", ""))
        if not fname or fname in SKIP_FIELDS:
            continue

        ftype  = f.get("type") or {}
        fkind  = base_kind(ftype)
        fbname = base_name(ftype)

        # --- Scalar / Enum ---
        if fkind in SCALAR_KINDS:
            parts.append(fname)
            continue

        # --- OBJECT ---
        if fkind == "OBJECT":

            # sys block - always the same
            if fname == "sys":
                parts.append(
                    "sys { id firstPublishedAt publishedAt environmentId spaceId }"
                )
                continue

            # linkedFrom - skip at all depths (huge, recursive)
            if fname == "linkedFrom":
                continue

            # Collection / CursorCollection sub-fields - skip at every depth.
            # This is the central fix vs. v2. We recover the data via a
            # second pass that hits one item at a time with explicit limits.
            if is_collection_typename(fbname):
                # CursorCollection is always pure dead weight (duplicate of
                # the regular Collection through a different pagination
                # interface).  Skip and don't even record it.
                if is_cursor_collection_typename(fbname):
                    continue
                if nested_collections is not None and depth == 0:
                    nested_collections.append({
                        "field":     fname,
                        "type":      fbname,
                    })
                continue

            # Asset - emit a fixed useful selection
            if fbname == "Asset":
                parts.append(
                    f"{fname} {{ url title fileName contentType size width height }}"
                )
                continue

            # Rich-text wrapper: field name ends with a known suffix AND
            # the resolved type has a `json` field.
            if fname.endswith(RICH_TEXT_WRAPPER_SUFFIXES) and fbname:
                sub_t = types_by_name.get(fbname, {})
                sub_field_names = {sf["name"] for sf in (sub_t.get("fields") or [])}
                if "json" in sub_field_names:
                    if rich_text_links and "links" in sub_field_names:
                        # Useful, but adds complexity. Each links { entries
                        # { block { sys { id } } } } costs 1 per link, but
                        # exposes the entry/asset graph for the body.
                        parts.append(
                            f"{fname} {{ json links {{ "
                            f"entries {{ "
                            f"  block {{ sys {{ id }} __typename }} "
                            f"  inline {{ sys {{ id }} __typename }} "
                            f"  hyperlink {{ sys {{ id }} __typename }} "
                            f"}} "
                            f"assets {{ "
                            f"  block {{ sys {{ id }} url title }} "
                            f"  hyperlink {{ sys {{ id }} url title }} "
                            f"}} "
                            f"}} }}"
                        )
                    else:
                        parts.append(f"{fname} {{ json }}")
                    continue

            # General OBJECT: recurse if within depth budget
            if depth < 2 and fbname not in visited:
                sub_sel = build_selection(
                    fbname, types_by_name, depth + 1, visited,
                    rich_text_links=rich_text_links,
                    nested_collections=None,   # only record top-level
                )
                if sub_sel:
                    parts.append(f"{fname} {{ {sub_sel} }}")
            continue

        # --- UNION / INTERFACE ---
        if fkind in ("UNION", "INTERFACE"):
            parts.append(f"{fname} {{ __typename sys {{ id }} }}")
            continue

        # --- LIST of scalars (e.g. [String]) ---
        if fkind == "LIST":
            inner = unwrap_type(ftype.get("ofType") or {})
            if inner.get("kind") in SCALAR_KINDS:
                parts.append(fname)
            continue

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Collection and singleton discovery
# ---------------------------------------------------------------------------

def get_required_args(field: dict) -> list:
    """
    Return a list of args that have a NON_NULL type and no default value.
    Each entry: { name, base_type_name, base_type_kind }
    """
    required = []
    for a in (field.get("args") or []):
        t = a.get("type") or {}
        if t.get("kind") != "NON_NULL":
            continue
        if a.get("defaultValue") is not None:
            continue
        required.append({
            "name":            a["name"],
            "base_type_name":  base_name(t),
            "base_type_kind":  base_kind(t),
        })
    return required


def discover_entries(query_root_fields: list, types_by_name: dict,
                     rich_text_links: bool = False) -> tuple:
    """
    Walk all root Query fields and categorise them. Returns
        (collections, arg_free_singletons, id_singletons)
    Each dict has:
        { root_field, item_type, selection,
          nested_collections: [...]   # only for collection entries
          required_args: [...]        # only for id_singletons }
    """
    query_type = None
    for t in types_by_name.values():
        field_names = {f["name"] for f in (t.get("fields") or [])}
        if query_root_fields and query_root_fields[0] in field_names:
            query_type = t
            break

    field_map = {f["name"]: f for f in ((query_type or {}).get("fields") or [])}

    collections          = []
    arg_free_singletons  = []
    id_singletons        = []
    seen_types           = set()

    for rf in query_root_fields:
        rf_safe = safe_name(rf)
        if not rf_safe:
            continue
        f = field_map.get(rf_safe)
        if not f:
            continue

        ret_name = base_name(f.get("type") or {})
        ret_kind = base_kind(f.get("type") or {})

        if ret_kind != "OBJECT":
            continue

        ret_type        = types_by_name.get(ret_name, {})
        ret_field_names = {ff["name"] for ff in (ret_type.get("fields") or [])}

        # ---- Paginated root collection ----
        # CursorCollection roots get skipped entirely - same data, different
        # pagination, and we'd just be duplicating work.
        if (
            rf_safe.endswith("Collection")
            and not is_cursor_collection_typename(rf_safe)
            and {"items", "total", "skip"}.issubset(ret_field_names)
        ):
            items_field = next(
                (ff for ff in (ret_type.get("fields") or []) if ff["name"] == "items"),
                None,
            )
            if not items_field:
                continue
            item_type_name = base_name(items_field.get("type") or {})
            if not safe_name(item_type_name) or item_type_name in seen_types:
                continue

            nested_collections = []
            selection = build_selection(
                item_type_name, types_by_name,
                rich_text_links=rich_text_links,
                nested_collections=nested_collections,
            )
            if not selection:
                continue

            seen_types.add(item_type_name)
            collections.append({
                "root_field":         rf_safe,
                "item_type":          item_type_name,
                "selection":          selection,
                "nested_collections": nested_collections,
            })

        # ---- Singleton ----
        elif not rf_safe.endswith("Collection"):
            if ret_name in seen_types:
                continue

            required  = get_required_args(f)
            selection = build_selection(
                ret_name, types_by_name,
                rich_text_links=rich_text_links,
            )
            if not selection:
                continue

            if not required:
                seen_types.add(ret_name)
                arg_free_singletons.append({
                    "root_field": rf_safe,
                    "item_type":  ret_name,
                    "selection":  selection,
                })
            else:
                id_singletons.append({
                    "root_field":     rf_safe,
                    "item_type":      ret_name,
                    "selection":      selection,
                    "required_args":  required,
                })

    return collections, arg_free_singletons, id_singletons


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def gql_error_code(err: dict) -> str:
    return (((err.get("extensions") or {}).get("contentful") or {})
            .get("code", ""))


def classify_gql_errors(gql_errs: list) -> tuple:
    """
    Split GraphQL errors into (fatal, non_fatal, complexity).
    * non_fatal: dirty data (broken refs etc.) - keep data, record warning
    * complexity: query shape too heavy - adaptive page-size retry
    * fatal: everything else (auth, syntax, unknown field, etc.)
    """
    fatal, non_fatal, complexity = [], [], []
    for e in gql_errs or []:
        code = gql_error_code(e)
        if code in NON_FATAL_GQL_CODES:
            non_fatal.append(e)
        elif code in COMPLEXITY_GQL_CODES:
            complexity.append(e)
        else:
            fatal.append(e)
    return fatal, non_fatal, complexity


# ---------------------------------------------------------------------------
# Collection dump (adaptive page size)
# ---------------------------------------------------------------------------

def _try_fetch_page(url: str, token: str, rf: str, selection: str,
                    skip: int, limit: int, delay: float) -> dict:
    """One attempt at a page. Returns the gql() response."""
    q = (f"{{ {rf}(limit: {limit}, skip: {skip}) "
         f"{{ total items {{ {selection} }} }} }}")
    return gql(url, token, q, delay=delay)


def dump_collection(url: str, token: str, entry: dict,
                    initial_page_size: int, delay: float,
                    min_page_size: int = 1) -> dict:
    """
    Paginate through a collection until exhausted, with **adaptive page
    sizing** on complexity errors.

    Algorithm:
      1. Start at `initial_page_size`.
      2. For each window starting at `skip`:
         a. Try the full selection at `current_limit`.
         b. If TOO_COMPLEX_QUERY: halve current_limit, retry the SAME
            window. Floor at `min_page_size`.
         c. If still failing at the floor: record the error and fall back
            to sys-only for this window only. Do NOT abort the whole
            collection.
         d. On success, advance `skip` by current_limit.
      3. Once a smaller page size succeeds, keep it for the rest of the
         dump (no point re-trying the bigger size on every page; in
         practice complexity is shape-driven, not data-driven).

    Returns:
      {
        root_field, item_type,
        total_reported, total_fetched, pages,
        initial_page_size, final_page_size,
        errors: [...], warnings: [...], warning_count, sys_only_pages,
        items: [...]
      }
    """
    rf            = entry["root_field"]
    selection     = entry["selection"]
    all_items     = []
    skip          = 0
    total         = None
    errors        = []
    warnings      = []
    sys_only_pages = []
    page          = 0
    current_limit = max(1, min(initial_page_size, 1000))
    sys_only_sel  = "sys { id }"
    # Once we've discovered that even the floor page size is too complex
    # for the rich selection, stop trying it - go straight to sys-only.
    rich_unusable = False

    while True:
        page += 1

        # --- Step 0: shortcut when prior pages proved rich is unusable ---
        if rich_unusable:
            resp_so = _try_fetch_page(url, token, rf, sys_only_sel,
                                      skip, current_limit, delay)
            so_data = (resp_so.get("data") or {}).get(rf) or {}
            if total is None:
                total = so_data.get("total")
            items_so = so_data.get("items") or []
            for it in items_so:
                if isinstance(it, dict):
                    it["_sys_only"] = True
            all_items.extend(items_so)
            sys_only_pages.append({
                "page": page, "skip": skip, "limit": current_limit,
                "fetched": len(items_so),
            })
            print(f"        page {page}: [sys-only, rich unusable] +{len(items_so)} "
                  f"({skip}..{skip+len(items_so)})")
            if not items_so or len(all_items) >= (total or 0):
                break
            skip += current_limit
            continue

        # --- Step 1: try the current limit with the full selection ---
        resp = _try_fetch_page(url, token, rf, selection, skip, current_limit, delay)

        # Transport-level failure -> can't continue this collection
        if "_error" in resp:
            errors.append({
                "page": page, "skip": skip, "limit": current_limit,
                "error": resp["_error"],
                "phase": "transport",
            })
            break

        gql_errs = resp.get("errors") or []
        fatal_errs, non_fatal_errs, complexity_errs = classify_gql_errors(gql_errs)
        data = (resp.get("data") or {}).get(rf)
        cost = resp.get("_cost")

        # --- Step 2: complexity blowup -> halve and retry the same window ---
        if complexity_errs and (data is None or not data.get("items")):
            halved = current_limit
            inner_attempts = 0
            while inner_attempts < 12 and halved > min_page_size:
                halved = max(min_page_size, halved // 2)
                inner_attempts += 1
                print(f"        page {page}: cost {cost} too high at limit "
                      f"{current_limit}; retrying at limit {halved}")
                resp = _try_fetch_page(url, token, rf, selection, skip, halved, delay)
                if "_error" in resp:
                    break
                gql_errs = resp.get("errors") or []
                fatal_errs, non_fatal_errs, complexity_errs = classify_gql_errors(gql_errs)
                data = (resp.get("data") or {}).get(rf)
                cost = resp.get("_cost")
                if data is not None and data.get("items") is not None:
                    current_limit = halved
                    break
                if not complexity_errs:
                    # Some other error class - stop trying to shrink
                    break

            # Did we eventually get data?
            if (data is None or data.get("items") is None) and complexity_errs:
                # Even the floor is too complex. Note this so we don't keep
                # burning requests trying to shrink on every subsequent page,
                # and fall back to sys-only for this window. We DO NOT abort
                # the whole collection - we record the gap and keep going so
                # the rest of the data isn't lost.
                rich_unusable = True
                resp_so = _try_fetch_page(url, token, rf, sys_only_sel,
                                          skip, current_limit, delay)
                so_data = (resp_so.get("data") or {}).get(rf) or {}
                if total is None:
                    total = so_data.get("total")
                items_so = so_data.get("items") or []
                # Mark each item so it's obvious in the output that we
                # only have sys.id for it.
                for it in items_so:
                    if isinstance(it, dict):
                        it["_sys_only"] = True
                all_items.extend(items_so)
                sys_only_pages.append({
                    "page": page, "skip": skip, "limit": current_limit,
                    "fetched": len(items_so),
                })
                errors.append({
                    "page": page, "skip": skip, "limit": current_limit,
                    "gql_errors": complexity_errs,
                    "phase": "complexity_fallback",
                    "note": ("fell back to sys-only for this window after "
                             "adaptive shrinking hit the floor"),
                })
                print(f"        page {page}: [sys-only fallback] +{len(items_so)} "
                      f"({skip}..{skip+len(items_so)})")
                if not items_so or len(all_items) >= (total or 0):
                    break
                skip += current_limit
                continue

        # --- Step 3: fatal errors -> record and (try once) sys-only fallback ---
        if (resp.get("_status", 200) >= 400 and not data) or (fatal_errs and not data):
            resp_so = _try_fetch_page(url, token, rf, sys_only_sel,
                                      skip, current_limit, delay)
            so_data = (resp_so.get("data") or {}).get(rf) or {}
            if not (resp_so.get("errors") or []) and so_data:
                if total is None:
                    total = so_data.get("total")
                items_so = so_data.get("items") or []
                for it in items_so:
                    if isinstance(it, dict):
                        it["_sys_only"] = True
                all_items.extend(items_so)
                sys_only_pages.append({
                    "page": page, "skip": skip, "limit": current_limit,
                    "fetched": len(items_so),
                })
                errors.append({
                    "page": page, "skip": skip, "limit": current_limit,
                    "gql_errors": fatal_errs,
                    "phase": "fatal_fallback",
                    "note": "sys-only fallback after fatal errors",
                })
                print(f"        [fatal-fallback] page {page}: +{len(items_so)} (sys-only)")
                if not items_so or len(all_items) >= (total or 0):
                    break
                skip += current_limit
                continue

            errors.append({
                "page": page, "skip": skip, "limit": current_limit,
                "gql_errors": fatal_errs,
                "http_status": resp.get("_status"),
                "phase": "fatal",
            })
            break

        if data is None:
            break

        if total is None:
            total = data.get("total", 0)
        items = data.get("items") or []
        all_items.extend(items)

        if non_fatal_errs:
            warnings.append({
                "page":   page,
                "skip":   skip,
                "limit":  current_limit,
                "count":  len(non_fatal_errs),
                "sample": non_fatal_errs[:5],
            })
        if fatal_errs:
            errors.append({
                "page": page, "skip": skip, "limit": current_limit,
                "gql_errors": fatal_errs,
                "phase": "partial",
                "note": "errors returned alongside data; continuing",
            })

        fetched     = len(all_items)
        warn_suffix = f"  ({len(non_fatal_errs)} broken refs)" if non_fatal_errs else ""
        cost_suffix = f"  cost={cost}" if cost is not None else ""
        print(f"        page {page}: +{len(items)}  [{fetched}/{total}]"
              f"  limit={current_limit}{cost_suffix}{warn_suffix}")

        if not items or fetched >= (total or 0):
            break
        skip += current_limit

    return {
        "root_field":         rf,
        "item_type":          entry["item_type"],
        "total_reported":     total,
        "total_fetched":      len(all_items),
        "pages":              page,
        "initial_page_size":  initial_page_size,
        "final_page_size":    current_limit,
        "errors":             errors,
        "warnings":           warnings,
        "warning_count":      sum(w["count"] for w in warnings),
        "sys_only_pages":     sys_only_pages,
        "items":              all_items,
    }


# ---------------------------------------------------------------------------
# Per-item nested-collection expansion (the recovery pass for the data we
# deliberately dropped during the main item-level selection)
# ---------------------------------------------------------------------------

def build_nested_collection_selection(coll_field_type: str,
                                       types_by_name: dict,
                                       rich_text_links: bool = False) -> str:
    """
    Given the TYPE NAME of a nested xxxCollection (e.g. "BlockEntryCollection"),
    return a selection set for `items { ... }` inside it.

    The nested collection type has shape:
        { items: [InnerType], total, limit, skip }
    We build a selection on InnerType using build_selection at a fresh
    visited set, but with depth boosted so that the inner item itself
    does NOT recurse into yet more nested collections (one level only).
    """
    coll_type = types_by_name.get(coll_field_type) or {}
    items_field = next(
        (ff for ff in (coll_type.get("fields") or []) if ff["name"] == "items"),
        None,
    )
    if not items_field:
        return ""
    inner_type = base_name(items_field.get("type") or {})
    if not safe_name(inner_type):
        return ""
    # Use depth=1 so the inner build_selection will NOT recurse into
    # nested collections from this item (it would otherwise blow up too).
    sel = build_selection(
        inner_type, types_by_name, depth=1,
        rich_text_links=rich_text_links,
    )
    return sel


def dump_collection_links(url: str, token: str,
                          collection_result: dict,
                          types_by_name: dict,
                          ids_subset: list,
                          link_pass_limit: int,
                          delay: float,
                          rich_text_links: bool = False,
                          batch_size: int = 5) -> dict:
    """
    Second-pass: for each item we have a sys.id for, fetch its nested
    xxxCollection sub-fields (which the main dump deliberately omitted
    to keep complexity under control). One request per item, with the
    full set of that item's nested collections included.

    Args:
        ids_subset:    list of sys.id values to expand
        link_pass_limit: limit applied to each nested collection inside
                       an item (Contentful default is 100; we typically
                       use 25-50 here)
        batch_size:    number of items per outer request (via GraphQL
                       aliases). Defaults to 5 - low because each item
                       can contain many nested collections.

    Returns:
      {
        item_type, root_field, nested_collections: [...],
        ids_known, ids_queried, results: [{id, data, errors, warnings}, ...],
        requests_used, final_batch_size
      }
    """
    item_type        = collection_result["item_type"]
    root_field       = collection_result["root_field"]
    nested           = collection_result.get("nested_collections") or []

    if not nested:
        return {"item_type": item_type, "skipped": True,
                "reason": "no nested collections discovered",
                "results": []}

    # Build the per-item selection: sys { id } plus each nested collection
    # at the requested limit, with the appropriate inner selection.
    inner_parts = ["sys { id }"]
    nested_meta = []
    for nc in nested:
        nfield   = safe_name(nc["field"])
        ntypename = safe_name(nc["type"])
        if not nfield or not ntypename:
            continue
        inner_sel = build_nested_collection_selection(
            ntypename, types_by_name, rich_text_links=rich_text_links,
        )
        if not inner_sel:
            continue
        inner_parts.append(
            f"{nfield}(limit: {link_pass_limit}) "
            f"{{ total skip limit items {{ {inner_sel} }} }}"
        )
        nested_meta.append({"field": nfield, "type": ntypename})

    if len(inner_parts) == 1:
        # Only sys remained - nothing to expand
        return {"item_type": item_type, "skipped": True,
                "reason": "no resolvable nested collection selections",
                "results": []}

    inner_sel_block = " ".join(inner_parts)

    # We need the singleton-style root_field that takes an `id` arg. For
    # paginated collection `xxxCollection`, the matching singleton is
    # typically the same name without the "Collection" suffix (camelCase).
    # We try to derive it; if it doesn't match a known field, the caller
    # can supply an override via collection_result["link_singleton_field"].
    sing_field = collection_result.get("link_singleton_field")
    if not sing_field:
        if root_field.endswith("Collection"):
            stem = root_field[:-len("Collection")]
            sing_field = stem  # e.g. blockColumnLayoutCollection -> blockColumnLayout
    sing_field = safe_name(sing_field or "")

    if not sing_field:
        return {"item_type": item_type, "skipped": True,
                "reason": f"couldn't derive singleton field from {root_field!r}",
                "results": []}

    results        = []
    requests_used  = 0
    errors_total   = 0
    warnings_total = 0
    current_batch  = max(1, batch_size)
    i              = 0
    capped         = list(ids_subset)

    print(f"        link-pass: {len(capped)} items, nested fields="
          f"{[n['field'] for n in nested_meta]}, "
          f"per-collection limit={link_pass_limit}, batch={current_batch}")

    while i < len(capped):
        subset = capped[i : i + current_batch]
        parts  = []
        for offset, sid in enumerate(subset):
            sid_esc = sid.replace("\\", "\\\\").replace('"', '\\"')
            alias   = f"e{i + offset}"
            parts.append(f'{alias}: {sing_field}(id: "{sid_esc}") '
                         f'{{ {inner_sel_block} }}')
        q = "{ " + " ".join(parts) + " }"
        r = gql(url, token, q, delay=delay)
        requests_used += 1

        # If the batch itself failed (HTTP 400 / complexity), halve and retry.
        gql_errs = r.get("errors") or []
        _, _, complexity_errs = classify_gql_errors(gql_errs)
        if ("_error" in r
            or (r.get("_status", 200) >= 400 and not r.get("data"))
            or (complexity_errs and not r.get("data"))):
            if current_batch > 1:
                new_batch = max(1, current_batch // 2)
                print(f"          link-pass batch of {current_batch} rejected "
                      f"({r.get('_status')}/{len(complexity_errs)} complexity errs), "
                      f"retrying with batch size {new_batch}")
                current_batch = new_batch
                continue
            # batch of 1 still failing - record per-id and skip
            for sid in subset:
                results.append({
                    "id": sid, "data": None,
                    "errors": [{
                        "message": "link-pass request failed at batch size 1",
                        "status": r.get("_status"),
                        "raw_errors": gql_errs,
                        "raw_error": r.get("_error"),
                    }],
                })
                errors_total += 1
            i += current_batch
            continue

        data = r.get("data") or {}
        errs_by_alias = {}
        for e in gql_errs:
            path = e.get("path") or []
            if path:
                errs_by_alias.setdefault(path[0], []).append(e)

        for offset, sid in enumerate(subset):
            alias      = f"e{i + offset}"
            alias_data = data.get(alias)
            alias_errs = errs_by_alias.get(alias, [])
            fatal, non_fatal, _ = classify_gql_errors(alias_errs)
            results.append({
                "id":       sid,
                "data":     alias_data,
                "errors":   fatal or None,
                "warnings": non_fatal or None,
            })
            if fatal:
                errors_total += 1
            if non_fatal:
                warnings_total += len(non_fatal)

        i += current_batch
        if i % (current_batch * 4) == 0 or i >= len(capped):
            print(f"          link-pass {min(i, len(capped))}/{len(capped)} "
                  f"({requests_used} reqs, batch={current_batch})")

    return {
        "item_type":         item_type,
        "root_field":        root_field,
        "singleton_field":   sing_field,
        "nested_collections": nested_meta,
        "ids_known":         len(capped),
        "ids_queried":       len(capped),
        "requests_used":     requests_used,
        "final_batch_size":  current_batch,
        "per_collection_limit": link_pass_limit,
        "errors_total":      errors_total,
        "warnings_total":    warnings_total,
        "results":           results,
        "skipped":           False,
    }


# ---------------------------------------------------------------------------
# Singleton dumps
# ---------------------------------------------------------------------------

def dump_singleton(url: str, token: str, entry: dict, delay: float) -> dict:
    rf  = entry["root_field"]
    sel = entry["selection"]
    q   = f"{{ {rf} {{ {sel} }} }}"
    r   = gql(url, token, q, delay=delay)
    gql_errs = r.get("errors") or []
    fatal_errs, non_fatal_errs, complexity_errs = classify_gql_errors(gql_errs)

    # If the singleton selection itself is too complex, retry with sys-only.
    data = (r.get("data") or {}).get(rf)
    if complexity_errs and data is None:
        r2 = gql(url, token, f"{{ {rf} {{ sys {{ id }} }} }}", delay=delay)
        data = (r2.get("data") or {}).get(rf)
        if data is not None:
            data = {**data, "_sys_only": True}

    return {
        "root_field": rf,
        "item_type":  entry["item_type"],
        "data":       data,
        "errors":     fatal_errs or None,
        "warnings":   non_fatal_errs or None,
        "complexity": complexity_errs or None,
        "_cost":      r.get("_cost"),
    }


def harvest_ids_for_type(coll_dir: str, item_type: str) -> list:
    """
    Open the collection JSON for `item_type` and return all sys.id values
    (deduped, in original order).
    """
    fpath = os.path.join(coll_dir, f"{item_type}.json")
    if not os.path.exists(fpath):
        return []
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    seen, ids = set(), []
    for item in (data.get("items") or []):
        sys_block = item.get("sys") if isinstance(item, dict) else None
        sid = (sys_block or {}).get("id")
        if sid and sid not in seen:
            seen.add(sid)
            ids.append(sid)
    return ids


def dump_id_singleton(url: str, token: str, entry: dict,
                       ids: list, delay: float,
                       max_calls: int = 500,
                       batch_size: int = 25) -> dict:
    """
    Call a singleton endpoint for each ID, batching multiple IDs per
    request using GraphQL aliases. On complexity / 400, halve batch_size
    and retry. Same adaptive principle as dump_collection_links but for
    schema-level singleton endpoints.
    """
    rf       = entry["root_field"]
    sel      = entry["selection"]
    required = entry["required_args"]

    if not required:
        return {"root_field": rf, "results": [], "skipped": True,
                "reason": "no required args"}

    # Pick the ID-like required arg
    id_arg = next(
        (a for a in required
         if a["base_type_name"] in ("String", "ID")
         and a["name"].lower() in ("id", "slug")),
        None,
    )
    if id_arg is None:
        id_arg = next(
            (a for a in required if a["base_type_name"] in ("String", "ID")),
            None,
        )
    if id_arg is None:
        return {
            "root_field":    rf, "item_type": entry["item_type"],
            "skipped":       True, "reason": "no resolvable ID-like required arg",
            "required_args": required, "results": [],
        }

    other_required = [a for a in required if a["name"] != id_arg["name"]]
    if other_required:
        return {
            "root_field":    rf, "item_type": entry["item_type"],
            "skipped":       True, "reason": "additional required args unfilled",
            "required_args": required, "results": [],
        }

    arg_name = safe_name(id_arg["name"])
    if not arg_name or not safe_name(rf):
        return {
            "root_field": rf, "item_type": entry["item_type"],
            "skipped": True, "reason": "unsafe field/arg name",
            "results": [],
        }

    capped         = ids[:max_calls]
    results        = []
    errors_total   = 0
    warnings_total = 0
    requests_used  = 0
    current_batch  = max(1, batch_size)

    print(f"        querying {len(capped)} IDs (of {len(ids)} known)"
          + (f" - capped at {max_calls}" if len(ids) > max_calls else "")
          + f", batch size: {current_batch}")

    def build_batch_query(id_subset, base_idx):
        parts = []
        for offset, sid in enumerate(id_subset):
            sid_escaped = sid.replace("\\", "\\\\").replace('"', '\\"')
            alias = f"e{base_idx + offset}"
            parts.append(
                f'{alias}: {rf}({arg_name}: "{sid_escaped}") {{ {sel} }}'
            )
        return "{ " + " ".join(parts) + " }"

    i = 0
    while i < len(capped):
        subset = capped[i : i + current_batch]
        q = build_batch_query(subset, i)
        r = gql(url, token, q, delay=delay)
        requests_used += 1

        gql_errs = r.get("errors") or []
        _, _, complexity_errs = classify_gql_errors(gql_errs)

        if ("_error" in r
            or (r.get("_status", 200) >= 400 and not r.get("data"))
            or (complexity_errs and not r.get("data"))):
            if current_batch > 1:
                new_batch = max(1, current_batch // 2)
                print(f"          batch of {current_batch} rejected "
                      f"(HTTP {r.get('_status')}, complexity={len(complexity_errs)}), "
                      f"retrying with batch size {new_batch}")
                current_batch = new_batch
                continue
            for sid in subset:
                results.append({
                    "id": sid, "data": None,
                    "errors": [{
                        "message": "request failed at batch size 1",
                        "status": r.get("_status"),
                        "raw_errors": gql_errs,
                        "raw_error": r.get("_error"),
                    }],
                    "warnings": None,
                })
                errors_total += 1
            i += current_batch
            continue

        data = r.get("data") or {}
        errs_by_alias = {}
        for e in gql_errs:
            path = e.get("path") or []
            if path:
                errs_by_alias.setdefault(path[0], []).append(e)

        for offset, sid in enumerate(subset):
            alias      = f"e{i + offset}"
            alias_data = data.get(alias)
            alias_errs = errs_by_alias.get(alias, [])
            fatal, non_fatal, _ = classify_gql_errors(alias_errs)
            results.append({
                "id":       sid,
                "data":     alias_data,
                "errors":   fatal or None,
                "warnings": non_fatal or None,
            })
            if fatal:
                errors_total += 1
            if non_fatal:
                warnings_total += len(non_fatal)

        i += current_batch
        if i % (current_batch * 4) == 0 or i >= len(capped):
            print(f"          {min(i, len(capped))}/{len(capped)} "
                  f"({requests_used} reqs)")

    return {
        "root_field":       rf,
        "item_type":        entry["item_type"],
        "id_arg":           arg_name,
        "ids_known":        len(ids),
        "ids_queried":      len(capped),
        "requests_used":    requests_used,
        "final_batch_size": current_batch,
        "results":          results,
        "errors_total":     errors_total,
        "warnings_total":   warnings_total,
        "skipped":          False,
    }


# ---------------------------------------------------------------------------
# Security checks
# ---------------------------------------------------------------------------

def run_security_checks(space: str, env: str, token: str,
                         url: str, delay: float) -> dict:
    results = {}

    r      = gql(url, token,
                 "{ __schema { mutationType { name } subscriptionType { name } } }",
                 delay=delay)
    schema = (r.get("data") or {}).get("__schema") or {}
    results["mutation_type"]     = schema.get("mutationType")
    results["subscription_type"] = schema.get("subscriptionType")

    purl = PREVIEW_URL.format(space=space, env=env)
    pr   = gql(purl, token, "{ __typename }", delay=delay)
    results["preview_endpoint"] = {
        "url":        purl,
        "accessible": bool(pr.get("data")),
        "status":     pr.get("_status"),
    }
    if pr.get("data"):
        print("    [!] WARNING: token works on preview endpoint - "
              "drafts may be exposed")

    cma_url = CMA_URL.format(space=space, env=env)
    cr      = http_get(cma_url, token, delay=delay)
    results["cma_endpoint"] = {
        "url":        cma_url,
        "accessible": cr.get("_status") == 200,
        "status":     cr.get("_status"),
    }
    if cr.get("_status") == 200:
        print("    [!] WARNING: token works on CMA (Management API) - "
              "write access possible!")

    return results


# ---------------------------------------------------------------------------
# Environment probe
# ---------------------------------------------------------------------------

def probe_environments(space: str, token: str, delay: float) -> dict:
    results = {}
    for env in COMMON_ENVIRONMENTS:
        url = GQL_URL.format(space=space, env=env)
        r   = gql(url, token, "{ __typename }", delay=delay)
        ok  = bool(r.get("data"))
        results[env] = {"accessible": ok, "status": r.get("_status")}
        print(f"    {env:<22} {'OK' if ok else 'HTTP ' + str(r.get('_status', 'err'))}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_token(args) -> str:
    """Resolve token from --token, --token-env, or fail."""
    if args.token and args.token_env:
        sys.exit("[!] Pass exactly one of --token / --token-env")
    if args.token:
        return args.token
    if args.token_env:
        tok = os.environ.get(args.token_env)
        if not tok:
            sys.exit(f"[!] Environment variable {args.token_env!r} is not set")
        return tok
    sys.exit("[!] Provide --token or --token-env <ENV_VAR_NAME>")


def main():
    ap = argparse.ArgumentParser(
        description="Comprehensive Contentful GraphQL data dumper (v3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--space",         required=True,           help="Contentful Space ID")
    ap.add_argument("--token",         default=None,            help="Bearer token (use --token-env in scripted runs to avoid leaking via the process list)")
    ap.add_argument("--token-env",     default=None,            help="Name of env var holding the bearer token")
    ap.add_argument("--env",           default="master",        help="Environment (default: master)")
    ap.add_argument("--out",           default="./dump",        help="Output root (default: ./dump)")
    ap.add_argument("--delay",         type=float, default=0.3, help="Seconds between requests (default: 0.3)")
    ap.add_argument("--page-size",     type=int,   default=100, help="Initial items per page 1-1000 (default: 100). Adaptive: halves on TOO_COMPLEX_QUERY.")
    ap.add_argument("--min-page-size", type=int,   default=1,   help="Floor for adaptive page-size shrinking (default: 1)")
    ap.add_argument("--skip-types",    nargs="*",  default=[],  help="Type names to skip")
    ap.add_argument("--only-types",    nargs="*",  default=[],  help="Only dump these types")
    ap.add_argument("--no-env-probe",  action="store_true",     help="Skip environment enumeration")
    ap.add_argument("--no-sec-checks", action="store_true",     help="Skip security checks")
    ap.add_argument("--rich-text-links", action="store_true",   help="Include `links { entries assets }` in rich-text bodies (slightly more complex, but surfaces embedded refs)")
    ap.add_argument("--link-pass",     action="store_true",     help="After the main collection dump, run a second pass per item to fetch nested xxxCollection references that were deliberately omitted to keep complexity under control")
    ap.add_argument("--link-pass-limit", type=int, default=25,  help="Items per nested collection inside the link pass (default: 25)")
    ap.add_argument("--link-pass-batch", type=int, default=5,   help="Items per outer request in the link pass (default: 5)")
    ap.add_argument("--max-id-calls",  type=int,   default=500, help="Max per-ID singleton calls per type (default: 500)")
    ap.add_argument("--id-batch-size", type=int,   default=25,  help="Initial alias-batch size for per-ID singleton queries")
    args = ap.parse_args()

    token = _resolve_token(args)

    url = GQL_URL.format(space=args.space, env=args.env)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(args.out, f"{args.space}_{args.env}_{ts}")
    os.makedirs(out, exist_ok=True)

    summary = {
        "space":   args.space,
        "env":     args.env,
        "url":     url,
        "started": datetime.now().isoformat(),
        "args":    {k: v for k, v in vars(args).items() if k != "token"},
        "collections":   [],
        "singletons":    [],
        "id_singletons": [],
        "link_pass":     [],
        "errors":        [],
    }

    # ------------------------------------------------------------------ 1. Introspection
    print(f"\n[1] Introspecting {url} ...")
    raw = gql(url, token, INTROSPECTION_QUERY, delay=args.delay)

    if "_error" in raw or raw.get("_status", 200) >= 400:
        sys.exit(f"[!] Introspection failed: status={raw.get('_status')} "
                 f"err={raw.get('_error')} body_keys={list(raw.keys())}")
    save(os.path.join(out, "introspection_raw.json"), raw)
    print(f"    Saved introspection_raw.json (cost={raw.get('_cost')})")
    if raw.get("errors"):
        print(f"    [!] Partial introspection errors: "
              f"{[e.get('message') for e in raw['errors']][:3]}")

    # ------------------------------------------------------------------ 2. Parse + discover
    print(f"\n[2] Parsing schema and discovering query surface ...")
    query_root_fields, types_by_name = parse_schema(raw)
    collections, arg_free_singletons, id_singletons = discover_entries(
        query_root_fields, types_by_name,
        rich_text_links=args.rich_text_links,
    )

    def _filter(lst):
        if args.only_types:
            lst = [e for e in lst if e["item_type"] in args.only_types]
        if args.skip_types:
            lst = [e for e in lst if e["item_type"] not in args.skip_types]
        return lst

    collections         = _filter(collections)
    arg_free_singletons = _filter(arg_free_singletons)
    id_singletons       = _filter(id_singletons)

    schema_summary = {
        "total_schema_types":      len(types_by_name),
        "collections_found":       len(collections),
        "arg_free_singletons":     len(arg_free_singletons),
        "id_singletons":           len(id_singletons),
        "mutation_type":           (raw.get("data") or {}).get("__schema", {}).get("mutationType"),
        "subscription_type":       (raw.get("data") or {}).get("__schema", {}).get("subscriptionType"),
        "collections": [
            {"root_field": e["root_field"], "item_type": e["item_type"],
             "nested_collections": [n["field"] for n in e.get("nested_collections", [])],
             "selection_preview": e["selection"][:160] + "..."}
            for e in collections
        ],
        "arg_free_singletons_list": [
            {"root_field": e["root_field"], "item_type": e["item_type"]}
            for e in arg_free_singletons
        ],
        "id_singletons_list": [
            {"root_field": e["root_field"], "item_type": e["item_type"],
             "required_args": [a["name"] + ": " + a["base_type_name"] + "!"
                               for a in e["required_args"]]}
            for e in id_singletons
        ],
    }
    save(os.path.join(out, "schema_summary.json"), schema_summary)
    print(f"    {len(collections)} collections, "
          f"{len(arg_free_singletons)} arg-free singletons, "
          f"{len(id_singletons)} ID-required singletons")

    # ------------------------------------------------------------------ 3. Security checks
    if not args.no_sec_checks:
        print(f"\n[3] Security checks ...")
        sec = run_security_checks(args.space, args.env, token, url, args.delay)
        save(os.path.join(out, "security_checks.json"), sec)
        print(f"    mutationType        = {sec['mutation_type']}")
        print(f"    preview accessible  = {sec['preview_endpoint']['accessible']}")
        print(f"    CMA accessible      = {sec['cma_endpoint']['accessible']}")
        summary["security_checks"] = sec
    else:
        print(f"\n[3] Security checks skipped.")

    # ------------------------------------------------------------------ 4. Environment probe
    if not args.no_env_probe:
        print(f"\n[4] Probing {len(COMMON_ENVIRONMENTS)} common environments ...")
        env_results = probe_environments(args.space, token, args.delay)
        save(os.path.join(out, "environments_probe.json"), env_results)
        accessible = [e for e, v in env_results.items() if v["accessible"]]
        summary["environments_accessible"] = accessible
        print(f"    Accessible: {accessible}")
    else:
        print(f"\n[4] Environment probe skipped.")

    # ------------------------------------------------------------------ 5. Collections (adaptive)
    coll_dir = os.path.join(out, "collections")
    os.makedirs(coll_dir, exist_ok=True)
    print(f"\n[5] Dumping {len(collections)} collections (adaptive page size) ...")

    for i, entry in enumerate(collections, 1):
        tname = entry["item_type"]
        nested_names = [n["field"] for n in entry.get("nested_collections", [])]
        nested_hint = (f"  [{len(nested_names)} nested coll fields skipped: "
                       f"{nested_names[:3]}{'...' if len(nested_names) > 3 else ''}]"
                       if nested_names else "")
        print(f"  [{i:>3}/{len(collections)}] {tname} ({entry['root_field']}){nested_hint}")
        result = dump_collection(url, token, entry,
                                 initial_page_size=args.page_size,
                                 delay=args.delay,
                                 min_page_size=args.min_page_size)
        fname  = f"{tname}.json"
        save(os.path.join(coll_dir, fname), result)
        record = {
            "item_type":         tname,
            "root_field":        entry["root_field"],
            "total_reported":    result["total_reported"],
            "total_fetched":     result["total_fetched"],
            "initial_page_size": result["initial_page_size"],
            "final_page_size":   result["final_page_size"],
            "file":              os.path.join("collections", fname),
            "had_errors":        bool(result["errors"]),
            "warning_count":     result.get("warning_count", 0),
            "sys_only_pages":    len(result.get("sys_only_pages", [])),
            "nested_collections_skipped": nested_names,
        }
        summary["collections"].append(record)
        if result["errors"]:
            summary["errors"].append({"type": tname, "errors": result["errors"]})

    # ------------------------------------------------------------------ 5b. Link pass (optional)
    if args.link_pass:
        link_dir = os.path.join(out, "collection_links")
        os.makedirs(link_dir, exist_ok=True)
        link_eligible = [e for e in collections if e.get("nested_collections")]
        print(f"\n[5b] Link pass: expanding nested refs for "
              f"{len(link_eligible)} collection types ...")
        for i, entry in enumerate(link_eligible, 1):
            tname = entry["item_type"]
            ids = harvest_ids_for_type(coll_dir, tname)
            if not ids:
                print(f"  [{i:>3}/{len(link_eligible)}] {tname}: no IDs - skipping")
                continue
            # Load the main collection result to give to dump_collection_links
            coll_path = os.path.join(coll_dir, f"{tname}.json")
            with open(coll_path) as fh:
                coll_result = json.load(fh)
            print(f"  [{i:>3}/{len(link_eligible)}] {tname}: {len(ids)} items")
            result = dump_collection_links(
                url, token, coll_result, types_by_name, ids,
                link_pass_limit=args.link_pass_limit,
                delay=args.delay,
                rich_text_links=args.rich_text_links,
                batch_size=args.link_pass_batch,
            )
            fname = f"{tname}.json"
            save(os.path.join(link_dir, fname), result)
            summary["link_pass"].append({
                "item_type":      tname,
                "file":           os.path.join("collection_links", fname),
                "skipped":        result.get("skipped", False),
                "reason":         result.get("reason"),
                "ids_queried":    result.get("ids_queried", 0),
                "requests_used":  result.get("requests_used", 0),
                "errors_total":   result.get("errors_total", 0),
                "warnings_total": result.get("warnings_total", 0),
            })

    # ------------------------------------------------------------------ 6. Singletons
    sing_dir = os.path.join(out, "singletons")
    os.makedirs(sing_dir, exist_ok=True)
    if arg_free_singletons:
        print(f"\n[6a] Dumping {len(arg_free_singletons)} arg-free singletons ...")
        for i, entry in enumerate(arg_free_singletons, 1):
            tname = entry["item_type"]
            print(f"  [{i:>3}/{len(arg_free_singletons)}] {tname} ({entry['root_field']}) ...")
            result = dump_singleton(url, token, entry, args.delay)
            fname  = f"{entry['root_field']}.json"
            save(os.path.join(sing_dir, fname), result)
            summary["singletons"].append({
                "item_type":  tname,
                "root_field": entry["root_field"],
                "file":       os.path.join("singletons", fname),
                "had_data":   result["data"] is not None,
                "had_errors": bool(result["errors"]),
                "complexity_warned": bool(result.get("complexity")),
            })
    else:
        print(f"\n[6a] No argument-free singletons found.")

    # ------------------------------------------------------------------ 6b. ID singletons
    id_sing_dir = os.path.join(out, "singletons_by_id")
    if id_singletons:
        os.makedirs(id_sing_dir, exist_ok=True)
        print(f"\n[6b] Dumping {len(id_singletons)} ID-required singletons ...")
        for i, entry in enumerate(id_singletons, 1):
            tname = entry["item_type"]
            req_summary = ", ".join(a["name"] + ": " + a["base_type_name"] + "!"
                                    for a in entry["required_args"])
            print(f"  [{i:>3}/{len(id_singletons)}] {tname} ({entry['root_field']}) "
                  f"requires ({req_summary}) ...")

            ids = harvest_ids_for_type(coll_dir, tname)
            if not ids:
                print(f"        no IDs from collection dump - skipping")
                summary["id_singletons"].append({
                    "item_type":     tname,
                    "root_field":    entry["root_field"],
                    "skipped":       True,
                    "reason":        "no IDs harvested from collection dump",
                    "required_args": req_summary,
                })
                continue

            result = dump_id_singleton(
                url, token, entry, ids, args.delay,
                max_calls=args.max_id_calls,
                batch_size=args.id_batch_size,
            )
            fname = f"{entry['root_field']}.json"
            save(os.path.join(id_sing_dir, fname), result)
            summary["id_singletons"].append({
                "item_type":      tname,
                "root_field":     entry["root_field"],
                "file":           os.path.join("singletons_by_id", fname),
                "skipped":        result.get("skipped", False),
                "reason":         result.get("reason"),
                "ids_known":      result.get("ids_known"),
                "ids_queried":    result.get("ids_queried"),
                "errors_total":   result.get("errors_total"),
                "warnings_total": result.get("warnings_total"),
                "required_args":  req_summary,
            })
    else:
        print(f"\n[6b] No ID-required singletons found.")

    # ------------------------------------------------------------------ 7. Final summary
    summary["finished"]              = datetime.now().isoformat()
    summary["total_collections"]     = len(collections)
    summary["total_singletons"]      = len(arg_free_singletons)
    summary["total_id_singletons"]   = len(id_singletons)
    summary["collections_with_data"] = sum(
        1 for c in summary["collections"] if (c.get("total_fetched") or 0) > 0
    )
    summary["collections_with_sys_only_gaps"] = sum(
        1 for c in summary["collections"] if c.get("sys_only_pages")
    )
    summary["total_warnings"] = sum(
        c.get("warning_count", 0) for c in summary["collections"]
    )
    save(os.path.join(out, "_summary.json"), summary)

    print(f"\n{'='*60}")
    print(f"  Output                     : {out}/")
    print(f"  Collections dumped         : {summary['total_collections']}")
    print(f"  Collections w/ data        : {summary['collections_with_data']}")
    print(f"  Collections w/ sys-only gap: {summary['collections_with_sys_only_gaps']}")
    print(f"  Arg-free singletons        : {summary['total_singletons']}")
    print(f"  ID-required singletons     : {summary['total_id_singletons']}")
    if args.link_pass:
        print(f"  Link-pass types            : {len(summary['link_pass'])}")
    print(f"  Errors                     : {len(summary['errors'])}")
    print(f"  Warnings (broken refs etc) : {summary['total_warnings']}")
    if not args.no_sec_checks:
        sec = summary.get("security_checks", {})
        print(f"  Mutations                  : {sec.get('mutation_type')}")
        print(f"  Preview accessible         : {sec.get('preview_endpoint', {}).get('accessible')}")
        print(f"  CMA accessible             : {sec.get('cma_endpoint', {}).get('accessible')}")
    if not args.no_env_probe:
        print(f"  Accessible envs            : {summary.get('environments_accessible', [])}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

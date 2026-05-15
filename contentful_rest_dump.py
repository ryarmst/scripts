#!/usr/bin/env python3
"""
contentful_rest_dump.py
-----------------------
Comprehensive Contentful REST API enumeration for an authorized security test.

Probes every documented Contentful REST endpoint across the Content Delivery
API (CDA), Content Preview API (CPA), and Content Management API (CMA), across
every accessible environment, and produces a Markdown summary report.

Usage:
    python3 contentful_rest_dump.py --space <SPACE_ID> --token <TOKEN> [options]

Options:
    --space         Contentful Space ID (required)
    --token         Bearer token - CDA, CPA, or CMA (required)
    --env           Primary environment to start from (default: master)
    --out           Output root directory (default: ./rest_dump)
    --delay         Seconds between requests (default: 0.3)
    --page-size     Items per page (default: 1000, max 1000)
    --no-env-probe  Skip enumerating other environments
    --max-pages     Safety cap on pagination per endpoint (default: 100)

Output layout:
    <out>/<space>_<timestamp>/
        token_scope.json            Which APIs the token can reach
        environments.json           Per-environment accessibility
        endpoints_index.json        Map of every endpoint hit, status, file
        cda/<env>/<endpoint>.json   CDA responses
        cpa/<env>/<endpoint>.json   CPA responses
        cma/<env>/<endpoint>.json   CMA responses
        cma/space/<endpoint>.json   CMA space-level (no env) responses
        cma/org/<endpoint>.json     CMA organization-level responses
        REPORT.md                   Human-readable findings summary
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from urllib.parse import quote

try:
    import requests
except ImportError:
    sys.exit("[!] Install requests: pip install requests")


# ---------------------------------------------------------------------------
# API hosts
# ---------------------------------------------------------------------------

CDA_HOST = "https://cdn.contentful.com"
CPA_HOST = "https://preview.contentful.com"
CMA_HOST = "https://api.contentful.com"

COMMON_ENVIRONMENTS = [
    "master", "staging", "stage", "dev", "development",
    "preview", "qa", "test", "uat", "release", "production",
    "sandbox", "demo", "main", "next", "v2",
]

# Endpoints that exist at the environment level on CDA / CPA
# (read-only delivery surface)
DELIVERY_ENV_ENDPOINTS = [
    "content_types",
    "entries",
    "assets",
    "locales",
    "tags",
]

# Endpoints that exist at the environment level on CMA
# (read-write management surface — read-only access still leaks a lot)
CMA_ENV_ENDPOINTS = [
    "content_types",
    "entries",
    "assets",
    "locales",
    "tags",
    "editor_interfaces",
    "extensions",
    "app_installations",
    "ui_extensions",
    "entry_tasks",
    "scheduled_actions",
    "releases",
    "environment_aliases",
    "content_types/published",
]

# Endpoints that exist at the space level on CMA (no environment in the path)
CMA_SPACE_ENDPOINTS = [
    "",                          # GET /spaces/<id> -> space metadata
    "environments",              # list environments
    "environment_aliases",
    "api_keys",                  # CDA token definitions
    "preview_api_keys",          # CPA token definitions
    "personal_access_tokens",    # PATs (org-scoped, but sometimes visible)
    "roles",
    "webhook_definitions",
    "webhook_definitions/calls",
    "memberships",               # space members
    "users",
    "teams",
    "team_space_memberships",
    "uploads",                   # uploaded files awaiting processing
]

# Endpoints at the org level on CMA — usually requires org-scoped tokens
CMA_ORG_ENDPOINTS_PATHS = [
    "/users/me",                                  # current user identity
    "/organizations",                             # orgs the token can see
    "/spaces",                                    # all spaces token can see
]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def http_get(url: str, token: str, delay: float = 0.0,
             params: dict = None, extra_headers: dict = None) -> dict:
    """Perform an authenticated GET. Returns a normalised result dict."""
    time.sleep(delay)
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent":    "contentful-rest-dump/1.0",
        "Accept":        "application/vnd.contentful.management.v1+json, application/vnd.contentful.delivery.v1+json, application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
    except Exception as e:
        return {"_error": str(e), "url": url, "status": None}

    result = {
        "url":        r.url,
        "status":     r.status_code,
        "headers":    dict(r.headers),
    }
    # Attempt JSON parse, fall back to text
    try:
        result["body"] = r.json()
        result["_json"] = True
    except ValueError:
        result["body"]  = r.text[:5000]
        result["_json"] = False
    return result


def save(path: str, data) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def safe_filename(s: str) -> str:
    return s.replace("/", "_").replace(":", "_").strip("_") or "_root"


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

def paginate(base_url: str, token: str, delay: float,
             page_size: int, max_pages: int) -> dict:
    """
    Walk a paginated collection endpoint (CDA/CMA both use skip/limit/total).
    Returns the assembled response with all items concatenated.
    """
    all_items   = []
    skip        = 0
    total       = None
    pages_used  = 0
    page_errors = []
    last_meta   = {}

    while pages_used < max_pages:
        pages_used += 1
        params = {"limit": page_size, "skip": skip}
        r = http_get(base_url, token, delay=delay, params=params)

        if r.get("_error") or not r.get("_json"):
            page_errors.append({"page": pages_used, "skip": skip, "result": r})
            break

        if r["status"] >= 400:
            page_errors.append({"page": pages_used, "skip": skip,
                                "status": r["status"], "body": r["body"]})
            break

        body  = r["body"] or {}
        items = body.get("items")
        if items is None:
            # Not a paginated collection — return the single response
            return {
                "paginated":   False,
                "first_url":   base_url,
                "status":      r["status"],
                "body":        body,
                "items":       None,
                "total":       None,
                "pages":       1,
                "page_errors": [],
            }

        if total is None:
            total = body.get("total", len(items))
        last_meta = {k: v for k, v in body.items() if k != "items"}
        all_items.extend(items)

        fetched = len(all_items)
        print(f"          page {pages_used}: +{len(items)}  [{fetched}/{total}]")

        if not items or fetched >= (total or 0):
            break
        skip += page_size

    return {
        "paginated":   True,
        "first_url":   base_url,
        "status":      200 if all_items and not page_errors else (page_errors[-1].get("status") if page_errors else 200),
        "meta":        last_meta,
        "items":       all_items,
        "total":       total,
        "fetched":     len(all_items),
        "pages":       pages_used,
        "page_errors": page_errors,
    }


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def probe_endpoint(host: str, path: str, token: str, delay: float) -> dict:
    """Single GET, no pagination — used for accessibility tests."""
    url = f"{host}{path}"
    return http_get(url, token, delay=delay, params={"limit": 1})


def probe_token_scope(space: str, env: str, token: str, delay: float) -> dict:
    """
    Determine which APIs the token can access by hitting one canonical
    endpoint on each.
    """
    results = {}

    print("    Probing CDA  ...")
    cda = probe_endpoint(CDA_HOST,
                         f"/spaces/{space}/environments/{env}/entries",
                         token, delay)
    results["cda"] = {"status": cda["status"], "accessible": cda["status"] == 200}

    print("    Probing CPA  ...")
    cpa = probe_endpoint(CPA_HOST,
                         f"/spaces/{space}/environments/{env}/entries",
                         token, delay)
    results["cpa"] = {"status": cpa["status"], "accessible": cpa["status"] == 200}

    print("    Probing CMA (space)  ...")
    cma_space = probe_endpoint(CMA_HOST, f"/spaces/{space}", token, delay)
    results["cma_space"] = {"status": cma_space["status"],
                            "accessible": cma_space["status"] == 200}

    print("    Probing CMA (entries) ...")
    cma_env = probe_endpoint(CMA_HOST,
                             f"/spaces/{space}/environments/{env}/entries",
                             token, delay)
    results["cma_env"] = {"status": cma_env["status"],
                          "accessible": cma_env["status"] == 200}

    print("    Probing /users/me ...")
    me = probe_endpoint(CMA_HOST, "/users/me", token, delay)
    results["users_me"] = {"status": me["status"],
                           "accessible": me["status"] == 200}
    if me["status"] == 200 and isinstance(me.get("body"), dict):
        results["users_me"]["identity"] = {
            "email":    me["body"].get("email"),
            "firstName": me["body"].get("firstName"),
            "lastName":  me["body"].get("lastName"),
            "sys_id":   (me["body"].get("sys") or {}).get("id"),
        }

    return results


def probe_environments(space: str, token: str, delay: float,
                       apis: dict) -> dict:
    """
    For each common environment name, test accessibility across whichever
    APIs the token can reach.
    """
    out = {}
    for env in COMMON_ENVIRONMENTS:
        out[env] = {}
        if apis["cda"]["accessible"]:
            r = probe_endpoint(CDA_HOST,
                               f"/spaces/{space}/environments/{env}/entries",
                               token, delay)
            out[env]["cda"] = {"status": r["status"],
                               "accessible": r["status"] == 200}
        if apis["cpa"]["accessible"]:
            r = probe_endpoint(CPA_HOST,
                               f"/spaces/{space}/environments/{env}/entries",
                               token, delay)
            out[env]["cpa"] = {"status": r["status"],
                               "accessible": r["status"] == 200}
        if apis["cma_env"]["accessible"]:
            r = probe_endpoint(CMA_HOST,
                               f"/spaces/{space}/environments/{env}/entries",
                               token, delay)
            out[env]["cma"] = {"status": r["status"],
                               "accessible": r["status"] == 200}

        results_str = " ".join(
            f"{api}={'OK' if v.get('accessible') else v.get('status')}"
            for api, v in out[env].items()
        )
        print(f"    {env:<22} {results_str or '(no API to probe)'}")

    return out


# ---------------------------------------------------------------------------
# Endpoint dumping
# ---------------------------------------------------------------------------

def dump_endpoint_list(host: str, base_path: str, endpoints: list,
                       token: str, out_dir: str, delay: float,
                       page_size: int, max_pages: int,
                       index: list, label: str) -> None:
    """
    Dump a list of endpoints under base_path, saving each to out_dir.
    Records each attempt (success or fail) into index.
    """
    for ep in endpoints:
        full_path = f"{base_path}/{ep}".rstrip("/")
        url       = f"{host}{full_path}"
        print(f"      {full_path}")
        result    = paginate(url, token, delay, page_size, max_pages)

        fname     = safe_filename(ep) + ".json"
        fpath     = os.path.join(out_dir, fname)
        save(fpath, result)

        record = {
            "label":       label,
            "url":         url,
            "path":        full_path,
            "file":        fpath,
            "status":      result.get("status"),
            "paginated":   result.get("paginated"),
            "total":       result.get("total"),
            "fetched":     result.get("fetched"),
            "had_errors":  bool(result.get("page_errors")),
        }
        index.append(record)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(out: str, summary: dict) -> str:
    """Produce a human-readable Markdown findings report."""
    lines = []
    lines.append(f"# Contentful REST Enumeration Report")
    lines.append(f"")
    lines.append(f"- **Space ID:** `{summary['space']}`")
    lines.append(f"- **Primary environment:** `{summary['env']}`")
    lines.append(f"- **Started:** {summary['started']}")
    lines.append(f"- **Finished:** {summary['finished']}")
    lines.append(f"- **Output directory:** `{out}`")
    lines.append("")

    # --- Token scope ---
    lines.append("## Token Scope")
    lines.append("")
    scope = summary.get("token_scope", {})
    for api, info in scope.items():
        marker = "✅" if info.get("accessible") else "❌"
        lines.append(f"- {marker} **{api.upper()}** — HTTP {info.get('status')}")
        if api == "users_me" and info.get("identity"):
            ident = info["identity"]
            lines.append(f"    - Identity leaked: `{ident.get('email')}` ({ident.get('firstName')} {ident.get('lastName')}, sys.id `{ident.get('sys_id')}`)")
    lines.append("")

    # --- Severity callouts ---
    findings = []
    if scope.get("cma_space", {}).get("accessible") or scope.get("cma_env", {}).get("accessible"):
        findings.append("🔴 **CRITICAL: Token has Content Management API (CMA) access.** This grants read/write to content, content types, webhooks, API keys, and roles. A CDA or CPA token should NEVER work against `api.contentful.com`.")
    if scope.get("users_me", {}).get("accessible"):
        ident = scope["users_me"].get("identity", {})
        findings.append(f"🔴 **CRITICAL: Token is a Personal Access Token (PAT) or OAuth token.** `/users/me` returned a user identity (`{ident.get('email')}`). This token inherits the full permissions of that user account.")
    if scope.get("cpa", {}).get("accessible"):
        findings.append("🟠 **HIGH: Token has Content Preview API (CPA) access.** This exposes unpublished/draft content. If the same token is shipped in production clients, all drafts are publicly readable.")

    # Multi-env access
    envs = summary.get("environments", {})
    accessible_envs = {
        api: [e for e, v in envs.items() if v.get(api, {}).get("accessible")]
        for api in ("cda", "cpa", "cma")
    }
    for api, env_list in accessible_envs.items():
        if len(env_list) > 1:
            findings.append(f"🟡 **MEDIUM: Token accesses multiple environments via {api.upper()}:** `{', '.join(env_list)}`. Tokens should be scoped to a single environment.")

    if findings:
        lines.append("## Key Findings")
        lines.append("")
        for f in findings:
            lines.append(f"- {f}")
        lines.append("")

    # --- Environment matrix ---
    if envs:
        lines.append("## Environment Accessibility Matrix")
        lines.append("")
        lines.append("| Environment | CDA | CPA | CMA |")
        lines.append("|---|---|---|---|")
        for env, apis in envs.items():
            cda = "✅" if apis.get("cda", {}).get("accessible") else (str(apis.get("cda", {}).get("status", "—")))
            cpa = "✅" if apis.get("cpa", {}).get("accessible") else (str(apis.get("cpa", {}).get("status", "—")))
            cma = "✅" if apis.get("cma", {}).get("accessible") else (str(apis.get("cma", {}).get("status", "—")))
            lines.append(f"| `{env}` | {cda} | {cpa} | {cma} |")
        lines.append("")

    # --- Endpoint dump summary ---
    lines.append("## Endpoint Dump Summary")
    lines.append("")
    index = summary.get("endpoints_index", [])

    # Group by label
    by_label = {}
    for r in index:
        by_label.setdefault(r["label"], []).append(r)

    for label, recs in by_label.items():
        ok_count   = sum(1 for r in recs if r.get("status") == 200)
        item_count = sum(r.get("fetched") or 0 for r in recs)
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"- Endpoints probed: **{len(recs)}**")
        lines.append(f"- Successful (HTTP 200): **{ok_count}**")
        lines.append(f"- Total items collected: **{item_count}**")
        lines.append("")
        lines.append("| Path | Status | Total | Fetched | File |")
        lines.append("|---|---|---|---|---|")
        for r in recs:
            path_disp  = r["path"] or "/"
            status     = r.get("status")
            total      = r.get("total") if r.get("total") is not None else "—"
            fetched    = r.get("fetched") if r.get("fetched") is not None else "—"
            file_rel   = os.path.relpath(r["file"], out)
            lines.append(f"| `{path_disp}` | {status} | {total} | {fetched} | `{file_rel}` |")
        lines.append("")

    # --- Top-level interesting files ---
    interesting = [
        ("API key definitions",        "cma/space/api_keys.json"),
        ("Preview API key definitions","cma/space/preview_api_keys.json"),
        ("Personal access tokens",     "cma/space/personal_access_tokens.json"),
        ("Webhook definitions",        "cma/space/webhook_definitions.json"),
        ("Roles",                      "cma/space/roles.json"),
        ("Space members",              "cma/space/memberships.json"),
        ("Users",                      "cma/space/users.json"),
        ("Environments list",          "cma/space/environments.json"),
        ("Locales",                    f"cda/{summary['env']}/locales.json"),
        ("Tags",                       f"cda/{summary['env']}/tags.json"),
        ("Content types",              f"cda/{summary['env']}/content_types.json"),
    ]
    pointers = []
    for label, relpath in interesting:
        full = os.path.join(out, relpath)
        if os.path.exists(full):
            pointers.append((label, relpath))
    if pointers:
        lines.append("## Highest-Value Files to Inspect")
        lines.append("")
        for label, relpath in pointers:
            lines.append(f"- **{label}** → `{relpath}`")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Generated by `contentful_rest_dump.py`")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Comprehensive Contentful REST API enumeration and dump.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--space",        required=True,           help="Contentful Space ID")
    ap.add_argument("--token",        required=True,           help="Bearer token (CDA, CPA, or CMA)")
    ap.add_argument("--env",          default="master",        help="Primary environment (default: master)")
    ap.add_argument("--out",          default="./rest_dump",   help="Output root (default: ./rest_dump)")
    ap.add_argument("--delay",        type=float, default=0.3, help="Seconds between requests")
    ap.add_argument("--page-size",    type=int, default=1000,  help="Items per page (max 1000)")
    ap.add_argument("--max-pages",    type=int, default=100,   help="Safety cap on pagination per endpoint")
    ap.add_argument("--no-env-probe", action="store_true",     help="Skip enumerating other environments")
    args = ap.parse_args()

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(args.out, f"{args.space}_{ts}")
    os.makedirs(out, exist_ok=True)

    summary = {
        "space":           args.space,
        "env":             args.env,
        "started":         datetime.now().isoformat(),
        "endpoints_index": [],
    }

    # ------------------------------------------------------------------ 1. Scope
    print(f"\n[1] Determining token scope ...")
    scope = probe_token_scope(args.space, args.env, args.token, args.delay)
    save(os.path.join(out, "token_scope.json"), scope)
    summary["token_scope"] = scope

    # ------------------------------------------------------------------ 2. Environments
    if not args.no_env_probe:
        # Only probe environments for APIs the token reaches
        env_check_apis = {
            "cda":     scope.get("cda",     {"accessible": False}),
            "cpa":     scope.get("cpa",     {"accessible": False}),
            "cma_env": scope.get("cma_env", {"accessible": False}),
        }
        if any(v["accessible"] for v in env_check_apis.values()):
            print(f"\n[2] Probing {len(COMMON_ENVIRONMENTS)} environments ...")
            env_results = probe_environments(args.space, args.token,
                                             args.delay, env_check_apis)
            save(os.path.join(out, "environments.json"), env_results)
            summary["environments"] = env_results
        else:
            print(f"\n[2] Skipping environment probe — no accessible API.")
            summary["environments"] = {}
    else:
        summary["environments"] = {}

    # Determine which environments to dump from (always include primary)
    envs_to_dump = {args.env}
    if summary.get("environments"):
        for env, apis in summary["environments"].items():
            for api, info in apis.items():
                if info.get("accessible"):
                    envs_to_dump.add(env)
    envs_to_dump = sorted(envs_to_dump)

    # ------------------------------------------------------------------ 3. CDA dump
    if scope.get("cda", {}).get("accessible"):
        print(f"\n[3] Dumping CDA endpoints across {len(envs_to_dump)} env(s) ...")
        for env in envs_to_dump:
            env_dir = os.path.join(out, "cda", env)
            os.makedirs(env_dir, exist_ok=True)
            print(f"  CDA / {env}")
            dump_endpoint_list(
                CDA_HOST,
                f"/spaces/{args.space}/environments/{env}",
                DELIVERY_ENV_ENDPOINTS,
                args.token, env_dir, args.delay,
                args.page_size, args.max_pages,
                summary["endpoints_index"],
                f"CDA / {env}",
            )
    else:
        print(f"\n[3] CDA not accessible — skipping.")

    # ------------------------------------------------------------------ 4. CPA dump
    if scope.get("cpa", {}).get("accessible"):
        print(f"\n[4] Dumping CPA (Preview) endpoints across {len(envs_to_dump)} env(s) ...")
        for env in envs_to_dump:
            env_dir = os.path.join(out, "cpa", env)
            os.makedirs(env_dir, exist_ok=True)
            print(f"  CPA / {env}")
            dump_endpoint_list(
                CPA_HOST,
                f"/spaces/{args.space}/environments/{env}",
                DELIVERY_ENV_ENDPOINTS,
                args.token, env_dir, args.delay,
                args.page_size, args.max_pages,
                summary["endpoints_index"],
                f"CPA / {env}",
            )
    else:
        print(f"\n[4] CPA not accessible — skipping.")

    # ------------------------------------------------------------------ 5. CMA env-level
    if scope.get("cma_env", {}).get("accessible"):
        print(f"\n[5] Dumping CMA environment endpoints across {len(envs_to_dump)} env(s) ...")
        for env in envs_to_dump:
            env_dir = os.path.join(out, "cma", env)
            os.makedirs(env_dir, exist_ok=True)
            print(f"  CMA / {env}")
            dump_endpoint_list(
                CMA_HOST,
                f"/spaces/{args.space}/environments/{env}",
                CMA_ENV_ENDPOINTS,
                args.token, env_dir, args.delay,
                args.page_size, args.max_pages,
                summary["endpoints_index"],
                f"CMA env / {env}",
            )
    else:
        print(f"\n[5] CMA environment-level not accessible — skipping.")

    # ------------------------------------------------------------------ 6. CMA space-level
    if scope.get("cma_space", {}).get("accessible"):
        print(f"\n[6] Dumping CMA space-level endpoints ...")
        space_dir = os.path.join(out, "cma", "space")
        os.makedirs(space_dir, exist_ok=True)
        dump_endpoint_list(
            CMA_HOST,
            f"/spaces/{args.space}",
            CMA_SPACE_ENDPOINTS,
            args.token, space_dir, args.delay,
            args.page_size, args.max_pages,
            summary["endpoints_index"],
            "CMA space",
        )
    else:
        print(f"\n[6] CMA space-level not accessible — skipping.")

    # ------------------------------------------------------------------ 7. CMA org/user-level
    if scope.get("users_me", {}).get("accessible"):
        print(f"\n[7] Dumping CMA org/user-level endpoints ...")
        org_dir = os.path.join(out, "cma", "org")
        os.makedirs(org_dir, exist_ok=True)
        for path in CMA_ORG_ENDPOINTS_PATHS:
            url    = f"{CMA_HOST}{path}"
            print(f"      {path}")
            result = paginate(url, args.token, args.delay,
                              args.page_size, args.max_pages)
            fname  = safe_filename(path) + ".json"
            fpath  = os.path.join(org_dir, fname)
            save(fpath, result)
            summary["endpoints_index"].append({
                "label":      "CMA org",
                "url":        url,
                "path":       path,
                "file":       fpath,
                "status":     result.get("status"),
                "paginated":  result.get("paginated"),
                "total":      result.get("total"),
                "fetched":    result.get("fetched"),
                "had_errors": bool(result.get("page_errors")),
            })
    else:
        print(f"\n[7] /users/me not accessible — skipping org endpoints.")

    # ------------------------------------------------------------------ 8. Index + report
    summary["finished"] = datetime.now().isoformat()
    save(os.path.join(out, "endpoints_index.json"),
         summary["endpoints_index"])

    report   = generate_report(out, summary)
    rep_path = os.path.join(out, "REPORT.md")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(report)

    save(os.path.join(out, "_summary.json"), summary)

    # ------------------------------------------------------------------ Console summary
    total_eps    = len(summary["endpoints_index"])
    ok_eps       = sum(1 for r in summary["endpoints_index"] if r.get("status") == 200)
    total_items  = sum((r.get("fetched") or 0) for r in summary["endpoints_index"])

    print(f"\n{'='*64}")
    print(f"  Output directory     : {out}/")
    print(f"  Report               : {rep_path}")
    print(f"  Endpoints attempted  : {total_eps}")
    print(f"  Endpoints successful : {ok_eps}")
    print(f"  Total items fetched  : {total_items}")
    print(f"  APIs accessible      : "
          + ", ".join(api for api, info in scope.items() if info.get("accessible")))
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()

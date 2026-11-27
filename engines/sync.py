"""Flat 3-argument day-level ledger sync & compact Bayesian health report engine."""

from __future__ import annotations

import pathlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .bayesian import BayesianTrackerEngine
from .common import (
    BJ_TZ,
    CLASSES,
    clean_str,
    is_valid_sales_date,
    iter_settled_prematch_cards,
)
from .conformal import ConformalEngine

_DRAFT_LINE_RE = re.compile(
    r"^\*\s*\[(H\d+)\b.*?stale_after:\s*([0-9T:\-+Z]+)",
    flags=re.IGNORECASE,
)
_NUM_OR_SPLIT_RE = re.compile(r"([+-]?)\s*(\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?")


def _clean_direction(raw: Any) -> str:
    """Strip trailing parenthetical notes like '平(纳入次选对冲)' -> '平'."""
    return re.sub(r"[（(][^）)]*[）)]$", "", clean_str(raw)).strip()


def _enrich_ah_regime(ah_regime: str, ah_pick: str) -> str:
    """Attach numeric line from card AH pick when regime text omitted digits (preserves 12-bucket accuracy)."""
    if not ah_regime or re.search(r"\d", ah_regime) or not ah_pick:
        return ah_regime
    cleaned = re.sub(r"[（(][^）)]*[）)]", " ", ah_pick)
    m = _NUM_OR_SPLIT_RE.search(cleaned)
    if not m:
        return ah_regime
    sign_s, a_s, b_s = m.groups()
    mag = (float(a_s) + float(b_s)) / 2.0 if b_s is not None else float(a_s)
    line_str = f"{-abs(mag):g}" if mag != 0.0 else "0.0"
    if ";" in ah_regime:
        first, rest = ah_regime.split(";", 1)
        return f"{first.strip()}({line_str}) ;{rest}"
    return f"{ah_regime.strip()}({line_str})"


def _load_wiki_draft_metadata(repo_root: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    """Auto-load H1..Hn stale_after metadata from skills/deep-analysis/wiki/index.md when present."""
    idx_path = repo_root / "skills" / "deep-analysis" / "wiki" / "index.md"
    if not idx_path.is_file():
        return {}
    meta: Dict[str, Dict[str, Any]] = {}
    for line in idx_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        m = _DRAFT_LINE_RE.search(line.strip())
        if m:
            hid, stale_after = m.group(1).upper(), m.group(2).strip()
            meta[hid] = {"entity_type": "draft", "stale_after": stale_after}
    return meta


def sync_day_ledgers(
    sales_date: str,
    repo_root: str,
    as_of: Optional[str] = None,
    entity_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    conformal_ledger_path: Optional[str] = None,
    bayesian_ledger_path: Optional[str] = None,
    alpha_target: float = 0.20,
    gamma: float = 0.05,
    prior_strength: float = 4.0,
    decay_gamma: float = 0.95,
) -> dict:
    """Idempotently sync one sales_date's settled prematch cards into Conformal + Bayesian ledgers and return compact health summary."""
    sd = clean_str(sales_date)
    if not is_valid_sales_date(sd):
        return {
            "status": "error",
            "error": {
                "error_code": "INVALID_SALES_DATE",
                "received": sales_date,
                "message": "sales_date must be YYYY-MM-DD.",
            },
        }

    root_raw = clean_str(repo_root)
    root = pathlib.Path(root_raw) if root_raw else None
    if root is None or not root.is_absolute() or not root.is_dir():
        return {
            "status": "error",
            "error": {
                "error_code": "INVALID_REPO_ROOT",
                "received": repo_root,
                "message": "repo_root must be an existing absolute directory path.",
            },
        }

    meta_file = root / "data" / sd / "meta.md"
    if not meta_file.is_file():
        return {
            "status": "error",
            "error": {
                "error_code": "META_NOT_FOUND",
                "received": str(meta_file),
                "message": f"Missing {meta_file}. Run match-scraper scores mode first.",
            },
        }

    c_path = conformal_ledger_path or str(root / "data" / "conformal_calibration.jsonl")
    b_path = bayesian_ledger_path or str(root / "data" / "bayesian_tracker.jsonl")

    conformal_records: List[Dict[str, Any]] = []
    bayesian_records: List[Dict[str, Any]] = []

    try:
        card_tuples = list(iter_settled_prematch_cards(str(root), sales_date=sd))
    except RuntimeError as exc:
        return {"status": "error", "error": {"error_code": "CARD_ITER_ERROR", "message": str(exc)}}

    for mid, card_sd, ko_dt, oc_dt, ident, dec, ms, norm_p, outcome in card_tuples:
        ko_iso = ko_dt.astimezone(BJ_TZ).isoformat()
        ct_iso = oc_dt.astimezone(BJ_TZ).isoformat()
        src_card = f"data/{card_sd}/cards/{mid}.yaml"
        league = clean_str(ident.get("league"))
        regime = ms.get("regime") if isinstance(ms.get("regime"), dict) else {}
        league_tier = clean_str(regime.get("league_tier"))
        # Enrich ah_tier_and_refund if card omitted numeric handicap line.
        ah_pick = ""
        td = dec.get("three_dimensions") if isinstance(dec.get("three_dimensions"), dict) else {}
        if td.get("margin"):
            ah_pick = clean_str(td.get("margin"))
        ah_regime = _enrich_ah_regime(clean_str(regime.get("ah_tier_and_refund")), ah_pick)

        conf_block = dec.get("conformal") if isinstance(dec.get("conformal"), dict) else {}
        c_rec: Dict[str, Any] = {
            "match_id": mid,
            "sales_date": card_sd,
            "kickoff_at": ko_iso,
            "cutoff_time": ct_iso,
            "league": league,
            "probs": [round(x, 6) for x in norm_p],
            "outcome": outcome,
            "source_card": src_card,
            "known_answer_replay": False,
        }
        pset = conf_block.get("prediction_set")
        if isinstance(pset, list) and pset:
            c_rec["prediction_set"] = [str(x).upper() for x in pset]
        if isinstance(conf_block.get("alpha_t"), (int, float)):
            c_rec["alpha_t"] = float(conf_block["alpha_t"])
        conformal_records.append(c_rec)

        seen_eids = set()
        primary = clean_str(dec.get("primary"))
        if primary in BayesianTrackerEngine._PRIMARY_DIR_LABELS:
            p_indices, p_wins = BayesianTrackerEngine._DIR_1X2_MAP[primary]
            bayesian_records.append({
                "match_id": mid,
                "sales_date": card_sd,
                "kickoff_at": ko_iso,
                "cutoff_time": ct_iso,
                "entity_id": "portfolio:1x2_primary",
                "direction": BayesianTrackerEngine._PRIMARY_DIR_LABELS[primary],
                "p_baseline": round(sum(norm_p[i] for i in p_indices), 6),
                "hit": 1 if outcome in p_wins else 0,
                "observation_role": "primary",
                "league": league,
                "league_tier": league_tier,
                "ah_tier_and_refund": ah_regime,
                "source_card": src_card,
                "known_answer_replay": False,
            })
            seen_eids.add("portfolio:1x2_primary")

        def _maybe_add(item: Any, role_override: Optional[str]) -> None:
            if not isinstance(item, dict):
                return
            eid = BayesianTrackerEngine.normalize_entity_id(item.get("page"))
            if not eid or not (eid.startswith("playbooks/") or eid.startswith("H")) or eid in seen_eids:
                return
            if role_override is not None:
                obs_role = role_override
            else:
                raw_role = clean_str(item.get("role")).lower()
                if raw_role not in ("primary", "residual"):
                    return
                obs_role = raw_role
            norm_dir = _clean_direction(item.get("direction"))
            if norm_dir not in BayesianTrackerEngine._DIR_1X2_MAP:
                return
            idxs, wins = BayesianTrackerEngine._DIR_1X2_MAP[norm_dir]
            bayesian_records.append({
                "match_id": mid,
                "sales_date": card_sd,
                "kickoff_at": ko_iso,
                "cutoff_time": ct_iso,
                "entity_id": eid,
                "direction": norm_dir,
                "p_baseline": round(sum(norm_p[i] for i in idxs), 6),
                "hit": 1 if outcome in wins else 0,
                "observation_role": obs_role,
                "league": league,
                "league_tier": league_tier,
                "ah_tier_and_refund": ah_regime,
                "source_card": src_card,
                "known_answer_replay": False,
            })
            seen_eids.add(eid)

        for hit_item in (ms.get("wiki_hits") or []):
            _maybe_add(hit_item, None)
        for sh_item in ((ms.get("wiki_scan") or {}).get("shadow_candidates") or []):
            _maybe_add(sh_item, "shadow")

    if conformal_records:
        c_res = ConformalEngine.record_outcomes(
            conformal_records,
            ledger_path=c_path,
            alpha_target=alpha_target,
            gamma=gamma,
        )
    else:
        c_res = {"status": "ok", "added_count": 0, "skipped": [], "total_rows": None, "latest_alpha_t": None}

    if bayesian_records:
        b_res = BayesianTrackerEngine.record_outcomes(
            bayesian_records,
            ledger_path=b_path,
            prior_strength=prior_strength,
            decay_gamma=decay_gamma,
        )
    else:
        b_res = {"status": "ok", "added_count": 0, "skipped": [], "total_rows": None}

    if c_res.get("status") != "ok":
        return {"status": "error", "stage": "conformal", "error": c_res.get("error")}
    if b_res.get("status") != "ok":
        return {"status": "error", "stage": "bayesian", "error": b_res.get("error")}

    resolved_as_of = clean_str(as_of) or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    merged_meta = _load_wiki_draft_metadata(root)
    if isinstance(entity_metadata, dict):
        merged_meta.update(entity_metadata)

    h_res = BayesianTrackerEngine.health_report(
        ledger_path=b_path,
        entity_metadata=merged_meta or None,
        as_of=resolved_as_of,
        prior_strength=prior_strength,
        decay_gamma=decay_gamma,
        include_contexts=False,
    )
    if h_res.get("status") != "ok":
        return {"status": "error", "stage": "health", "error": h_res.get("error")}

    portfolio_obj = None
    playbooks_compact: List[Dict[str, Any]] = []
    drafts_by_status: Dict[str, List[str]] = {}
    for ent in h_res.get("entities") or []:
        etype = ent.get("entity_type")
        if etype == "portfolio":
            portfolio_obj = {
                "entity_id": ent["entity_id"],
                "n_obs": ent["n_obs"],
                "hits": ent["hits"],
                "mean_p_baseline": ent["mean_p_baseline"],
                "posterior_mean": ent["posterior_mean"],
                "p_beat_market": ent["p_beat_market"],
                "decayed_n_eff": ent["decayed_n_eff"],
                "decayed_p_below_market": ent["decayed_p_below_market"],
                "cusum_drift_e": ent["cusum_drift_e"],
                "advisory_status": ent["advisory_status"],
            }
        elif etype == "draft":
            st = str(ent.get("advisory_status") or "COLLECTING")
            if ent.get("n_obs", 0) == 0:
                drafts_by_status.setdefault(st, []).append(ent["entity_id"])
            else:
                playbooks_compact.append({
                    "entity_id": ent["entity_id"],
                    "entity_type": "draft",
                    "n_obs": ent["n_obs"],
                    "hits": ent["hits"],
                    "posterior_mean": ent["posterior_mean"],
                    "p_beat_market": ent["p_beat_market"],
                    "e_pos": ent["e_pos"],
                    "e_neg": ent["e_neg"],
                    "advisory_status": st,
                })
        elif ent.get("n_obs", 0) > 0:
            playbooks_compact.append({
                "entity_id": ent["entity_id"],
                "n_obs": ent["n_obs"],
                "hits": ent["hits"],
                "posterior_mean": ent["posterior_mean"],
                "p_beat_market": ent["p_beat_market"],
                "e_pos": ent["e_pos"],
                "cusum_drift_e": ent["cusum_drift_e"],
                "advisory_status": ent["advisory_status"],
            })

    return {
        "status": "ok",
        "sales_date": sd,
        "settled_card_count": len(conformal_records),
        "conformal": {
            "added_count": c_res.get("added_count", 0),
            "skipped_count": len(c_res.get("skipped") or []),
            "total_rows": c_res.get("total_rows"),
            "latest_alpha_t": c_res.get("latest_alpha_t"),
            "empirical_lac_coverage": c_res.get("empirical_lac_coverage"),
        },
        "bayesian": {
            "added_count": b_res.get("added_count", 0),
            "skipped_count": len(b_res.get("skipped") or []),
            "total_rows": b_res.get("total_rows"),
        },
        "health": {
            "as_of": resolved_as_of,
            "total_rows": h_res.get("total_rows"),
            "role_counts": h_res.get("role_counts"),
            "context_counts": h_res.get("context_counts"),
            "alerts": h_res.get("alerts"),
            "portfolio": portfolio_obj,
            "playbooks": playbooks_compact,
            "zero_obs_drafts": drafts_by_status,
        },
    }

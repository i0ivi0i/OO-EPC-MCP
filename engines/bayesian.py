"""Pure-stdlib Market-Anchored Contextual Hierarchical Bayesian & Selection Engine."""

from __future__ import annotations

import hashlib
import pathlib
import random
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .common import (
    BJ_TZ,
    _is_strict_finite_number,
    _is_truthy_flag,
    atomic_write_ledger as _atomic_write_ledger,
    beta_ppf,
    clamp_prob,
    clean_str,
    is_valid_sales_date,
    iter_settled_prematch_cards,
    load_ledger as _load_ledger,
    parse_timestamp as _parse_timestamp,
    regularized_incomplete_beta,
    resolve_ledger_path as _resolve_ledger_path,
)
from .context import (
    DEFAULT_CTX_STRENGTH,
    DEFAULT_DECAY_GAMMA,
    DEFAULT_PRIOR_STRENGTH,
    DEFAULT_UCB_QUANTILE,
    P_MAX,
    P_MIN,
    VALID_CONTEXT_BUCKETS,
    VALID_OBSERVATION_ROLES,
    _evaluate_context_from_stats,
    normalize_context_bucket,
    resolve_context_parts,
)


class BayesianTrackerEngine:
    """Pure-stdlib Market-Anchored Beta Updating, Mixture E-Process, Page-CUSUM & Contextual Bandit Engine."""

    LAMBDA_GRID: Tuple[float, ...] = (0.10, 0.20, 0.35, 0.50)
    P_MIN: float = P_MIN
    P_MAX: float = P_MAX
    DEFAULT_PRIOR_STRENGTH: float = DEFAULT_PRIOR_STRENGTH
    DEFAULT_CTX_STRENGTH: float = DEFAULT_CTX_STRENGTH
    DEFAULT_DECAY_GAMMA: float = DEFAULT_DECAY_GAMMA
    DEFAULT_UCB_QUANTILE: float = DEFAULT_UCB_QUANTILE
    VALID_ENTITY_TYPES = ("draft", "playbook", "feature", "portfolio")
    VALID_SELECTION_MODES = ("bayes_ucb", "contextual_ts")
    _ENTITY_PATTERNS = (
        re.compile(r"^playbooks/[a-z0-9][a-z0-9-]*\.md$"),
        re.compile(r"^features/[a-z0-9][a-z0-9-]*\.md$"),
        re.compile(r"^H\d+$"),
        re.compile(r"^portfolio:[a-z0-9_]+$"),
    )
    _DIR_1X2_MAP: Dict[str, Tuple[Tuple[int, ...], Tuple[str, ...]]] = {
        "主胜": ((0,), ("H",)),
        "胜": ((0,), ("H",)),
        "平": ((1,), ("D",)),
        "平局": ((1,), ("D",)),
        "客胜": ((2,), ("A",)),
        "负": ((2,), ("A",)),
        "主不败": ((0, 1), ("H", "D")),
        "客不败": ((2, 1), ("A", "D")),
    }
    _PRIMARY_DIR_LABELS: Dict[str, str] = {"胜": "主胜", "平": "平", "负": "客胜"}

    @staticmethod
    def resolve_ledger_path(ledger_path: Optional[str]) -> Tuple[Optional[pathlib.Path], Optional[dict]]:
        return _resolve_ledger_path(ledger_path, "SPORTS_MATH_BAYESIAN_LEDGER", "data/bayesian_tracker.jsonl")

    @staticmethod
    def load_ledger(path: pathlib.Path) -> Tuple[List[Dict[str, Any]], Optional[dict]]:
        return _load_ledger(path)

    @staticmethod
    def atomic_write_ledger(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
        _atomic_write_ledger(path, rows, prefix=".bayesian_")

    @staticmethod
    def parse_timestamp(value: Any) -> Optional[datetime]:
        return _parse_timestamp(value)

    @classmethod
    def normalize_entity_id(cls, raw: Any) -> Optional[str]:
        if not isinstance(raw, str):
            return None
        s = re.split(r"[#\s§]", raw.strip(), maxsplit=1)[0].lstrip("./")
        for prefix in ("skills/deep-analysis/wiki/", "deep-analysis/wiki/", "wiki/"):
            if s.startswith(prefix):
                s = s[len(prefix):]
                break
        s = re.sub(r"^(playbooks|features)/\d{2}-", r"\1/", s)
        if re.match(r"^(playbooks|features)/[a-z0-9][a-z0-9-]*$", s):
            s = f"{s}.md"
        for pat in cls._ENTITY_PATTERNS:
            if pat.match(s):
                return s
        return None

    @classmethod
    def infer_entity_type(cls, entity_id: str, explicit_type: Optional[str] = None) -> str:
        if isinstance(explicit_type, str) and explicit_type.strip():
            et = {"hypothesis": "draft", "stable": "playbook"}.get(explicit_type.strip().lower(), explicit_type.strip().lower())
            if et in cls.VALID_ENTITY_TYPES:
                return et
        if entity_id.startswith("H"):
            return "draft"
        if entity_id.startswith("features/"):
            return "feature"
        if entity_id.startswith("portfolio:"):
            return "portfolio"
        return "playbook"

    @classmethod
    def clamp_baseline(cls, p_val: Any) -> Tuple[Optional[float], Optional[str]]:
        if not _is_strict_finite_number(p_val):
            return None, "INVALID_P_BASELINE"
        fv = float(p_val)
        if fv <= 0.0 or fv >= 1.0:
            return None, "INVALID_P_BASELINE"
        return round(clamp_prob(fv, cls.P_MIN, cls.P_MAX), 6), None

    @classmethod
    def sort_key(cls, row: Dict[str, Any]) -> Tuple[str, str, str, str]:
        dt = cls.parse_timestamp(row.get("kickoff_at"))
        ko_iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt else "9999-99-99T99:99:99Z"
        return (clean_str(row.get("sales_date")), ko_iso, clean_str(row.get("match_id")), clean_str(row.get("entity_id")))

    @classmethod
    def _group_by_entity_id(
        cls,
        rows: List[Dict[str, Any]],
        target_ids: Optional[set] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            if _is_truthy_flag(r.get("known_answer_replay")):
                continue
            norm = cls.normalize_entity_id(r.get("entity_id")) or clean_str(r.get("entity_id"))
            if norm and (target_ids is None or norm in target_ids):
                grouped.setdefault(norm, []).append(r)
        return grouped

    @classmethod
    def _filter_causal_rows(cls, rows: List[Dict[str, Any]], cutoff_dt: datetime) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        deadline = cutoff_dt - timedelta(hours=3)
        for r in rows:
            if _is_truthy_flag(r.get("known_answer_replay")):
                continue
            ct_dt = cls.parse_timestamp(r.get("cutoff_time"))
            ko_dt = cls.parse_timestamp(r.get("kickoff_at"))
            if ct_dt is not None and ko_dt is not None and ct_dt < ko_dt and ko_dt <= deadline:
                out.append(r)
        return out

    @classmethod
    def _validate_optional_context_bucket(
        cls, context_bucket: Optional[str]
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Validate optional context_bucket against VALID_CONTEXT_BUCKETS."""
        if context_bucket is not None and not isinstance(context_bucket, str):
            return None, {
                "error_code": "INVALID_CONTEXT_BUCKET",
                "received": context_bucket,
                "message": f"context_bucket must be a string matching the 12 canonical buckets or omitted, got {type(context_bucket).__name__}.",
            }
        if isinstance(context_bucket, str) and context_bucket.strip():
            cand_ctx = context_bucket.strip().lower()
            if cand_ctx not in VALID_CONTEXT_BUCKETS:
                return None, {
                    "error_code": "INVALID_CONTEXT_BUCKET",
                    "received": context_bucket,
                    "message": f"context_bucket must be one of the 12 canonical buckets or omitted, got {context_bucket!r}.",
                }
            return cand_ctx, None
        return None, None

    @staticmethod
    def _aggregate_roles(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {role: 0 for role in VALID_OBSERVATION_ROLES}
        for r in rows:
            role = clean_str(r.get("observation_role") or "primary").lower()
            if role in counts:
                counts[role] += 1
        return counts

    @staticmethod
    def _aggregate_context_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in rows:
            b = clean_str(r.get("context_bucket")).lower()
            if b in VALID_CONTEXT_BUCKETS:
                counts[b] = counts.get(b, 0) + 1
        return dict(sorted(counts.items()))

    @staticmethod
    def _validate_positive_hyperparams(**params: Any) -> Optional[dict]:
        for name, val in params.items():
            if not _is_strict_finite_number(val) or float(val) <= 0.0:
                return {
                    "error_code": "INVALID_HYPERPARAMETER",
                    "field": name,
                    "received": val,
                    "message": f"{name} must be a positive finite number, got {val!r}.",
                }
        return None

    @staticmethod
    def _build_observation_row(
        *,
        match_id: str,
        sales_date: str,
        kickoff_at: str,
        cutoff_time: str,
        entity_id: str,
        entity_type: str,
        direction: str,
        p_baseline: float,
        hit: int,
        source_card: str,
        league_tier: str,
        ah_tier: str,
        context_bucket: str,
        observation_role: str,
    ) -> Dict[str, Any]:
        return {
            "match_id": match_id,
            "sales_date": sales_date,
            "kickoff_at": kickoff_at,
            "cutoff_time": cutoff_time,
            "entity_id": entity_id,
            "entity_type": entity_type,
            "direction": direction,
            "p_baseline": p_baseline,
            "hit": hit,
            "source_card": source_card,
            "league_tier": league_tier,
            "ah_tier": ah_tier,
            "context_bucket": context_bucket,
            "observation_role": observation_role,
        }

    @classmethod
    def evaluate_entity(
        cls,
        entity_id: str,
        entity_type: str,
        rows: List[Dict[str, Any]],
        prior_strength: float = DEFAULT_PRIOR_STRENGTH,
        decay_gamma: float = DEFAULT_DECAY_GAMMA,
        stale_after: Optional[str] = None,
        as_of: Optional[str] = None,
        already_sorted: bool = False,
    ) -> Dict[str, Any]:
        sorted_rows = rows if already_sorted else sorted(rows, key=cls.sort_key)
        n_obs = len(sorted_rows)
        stale_dt = cls.parse_timestamp(stale_after) if stale_after else None
        as_of_dt = cls.parse_timestamp(as_of) if as_of else None

        if n_obs == 0:
            if entity_type == "draft" and stale_dt is not None and as_of_dt is not None and as_of_dt >= stale_dt:
                status = "STALE_REVIEW"
            else:
                status = "COLLECTING" if entity_type == "draft" else "HEALTHY"
            return {
                "entity_id": entity_id, "entity_type": entity_type,
                "n_obs": 0, "hits": 0, "mean_p_baseline": None,
                "posterior_mean": None, "ci90": None,
                "p_beat_market": None, "p_below_market": None,
                "e_pos": 1.0, "e_neg": 1.0, "anytime_p_pos": 1.0,
                "decayed_n_eff": 0.0, "decayed_posterior_mean": None,
                "decayed_p_below_market": None, "cusum_drift_e": 1.0,
                "advisory_status": status,
            }

        ys = [int(r["hit"]) for r in sorted_rows]
        p0s = [clamp_prob(float(r["p_baseline"]), cls.P_MIN, cls.P_MAX) for r in sorted_rows]
        hits = sum(ys)
        mean_p0 = sum(p0s) / n_obs

        m0 = max(0.5, float(prior_strength))
        a_full = m0 * mean_p0 + hits
        b_full = m0 * (1.0 - mean_p0) + (n_obs - hits)
        post_mean = a_full / (a_full + b_full)
        ci_lo = beta_ppf(0.05, a_full, b_full)
        ci_hi = beta_ppf(0.95, a_full, b_full)
        p_below = regularized_incomplete_beta(mean_p0, a_full, b_full)
        p_beat = 1.0 - p_below

        pos_wealths = [1.0] * len(cls.LAMBDA_GRID)
        neg_wealths = [1.0] * len(cls.LAMBDA_GRID)
        cusum_wealths = [1.0] * len(cls.LAMBDA_GRID)
        max_e_pos = 1.0
        for y_t, p_t in zip(ys, p0s):
            for idx, lam in enumerate(cls.LAMBDA_GRID):
                pos_wealths[idx] *= 1.0 + lam * (y_t - p_t) / p_t
                neg_factor = 1.0 - lam * (y_t - p_t) / (1.0 - p_t)
                neg_wealths[idx] *= neg_factor
                cusum_wealths[idx] = max(1.0, cusum_wealths[idx] * neg_factor)
            max_e_pos = max(max_e_pos, sum(pos_wealths) / len(cls.LAMBDA_GRID))

        e_pos = sum(pos_wealths) / len(cls.LAMBDA_GRID)
        e_neg = sum(neg_wealths) / len(cls.LAMBDA_GRID)
        cusum_drift_e = sum(cusum_wealths) / len(cls.LAMBDA_GRID)
        anytime_p_pos = min(1.0, 1.0 / max(1.0, max_e_pos))

        gamma = max(0.50, min(0.999, float(decay_gamma)))
        dec_n = dec_hits = dec_p0_sum = 0.0
        for idx, (y_t, p_t) in enumerate(zip(ys, p0s)):
            w = gamma ** (n_obs - 1 - idx)
            dec_n += w
            dec_hits += w * y_t
            dec_p0_sum += w * p_t
        dec_p0 = dec_p0_sum / dec_n
        a_dec = m0 * dec_p0 + dec_hits
        b_dec = m0 * (1.0 - dec_p0) + (dec_n - dec_hits)
        dec_post_mean = a_dec / (a_dec + b_dec)
        dec_p_below = regularized_incomplete_beta(dec_p0, a_dec, b_dec)

        if entity_type == "draft":
            if n_obs >= 5 and p_beat >= 0.90 and e_pos >= 20.0:
                status = "STRONG_EVIDENCE"
            elif n_obs >= 5 and p_beat >= 0.90 and e_pos >= 5.0:
                status = "PROMOTE_RECOMMENDED"
            elif n_obs >= 5 and p_below >= 0.85 and e_neg >= 5.0:
                status = "DEPRECATE_RECOMMENDED"
            elif stale_dt is not None and as_of_dt is not None and as_of_dt >= stale_dt:
                status = "STALE_REVIEW"
            else:
                status = "COLLECTING"
        else:
            if dec_n >= 8.0 and (dec_p_below >= 0.90 or cusum_drift_e >= 8.0):
                status = "DRIFT_ALERT"
            elif dec_n >= 5.0 and (dec_p_below >= 0.80 or cusum_drift_e >= 5.0):
                status = "DRIFT_WATCH"
            else:
                status = "HEALTHY"

        return {
            "entity_id": entity_id, "entity_type": entity_type,
            "n_obs": n_obs, "hits": hits, "mean_p_baseline": round(mean_p0, 6),
            "posterior_mean": round(post_mean, 6), "ci90": [round(ci_lo, 6), round(ci_hi, 6)],
            "p_beat_market": round(p_beat, 6), "p_below_market": round(p_below, 6),
            "e_pos": round(e_pos, 4), "e_neg": round(e_neg, 4),
            "anytime_p_pos": round(anytime_p_pos, 6), "decayed_n_eff": round(dec_n, 4),
            "decayed_posterior_mean": round(dec_post_mean, 6),
            "decayed_p_below_market": round(dec_p_below, 6),
            "cusum_drift_e": round(cusum_drift_e, 4), "advisory_status": status,
        }

    @classmethod
    def record_outcomes(
        cls,
        records: Any,
        ledger_path: Optional[str] = None,
        prior_strength: float = DEFAULT_PRIOR_STRENGTH,
        decay_gamma: float = DEFAULT_DECAY_GAMMA,
    ) -> dict:
        if not isinstance(records, list) or len(records) == 0:
            return {"status": "error", "error": {"error_code": "EMPTY_RECORDS", "message": "records must be a non-empty list."}}
        hp_err = cls._validate_positive_hyperparams(prior_strength=prior_strength, decay_gamma=decay_gamma)
        if hp_err is not None:
            return {"status": "error", "error": hp_err}

        lpath, l_err = cls.resolve_ledger_path(ledger_path)
        if l_err is not None:
            return {"status": "error", "error": l_err}
        assert lpath is not None

        existing, read_err = cls.load_ledger(lpath)
        if read_err is not None:
            return {"status": "error", "error": read_err}

        existing_keys = {
            (clean_str(r.get("match_id")), cls.normalize_entity_id(r.get("entity_id")) or clean_str(r.get("entity_id")))
            for r in existing
        }
        accepted: List[Dict[str, Any]] = []
        skipped: List[Dict[str, str]] = []
        touched_entities = set()

        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                skipped.append({"index": str(idx), "reason": "INVALID_RECORD_TYPE"})
                continue
            mid = clean_str(rec.get("match_id"))
            if not mid:
                skipped.append({"index": str(idx), "reason": "MISSING_MATCH_ID"})
                continue
            if _is_truthy_flag(rec.get("known_answer_replay")):
                skipped.append({"match_id": mid, "reason": "KNOWN_ANSWER_REPLAY_REJECTED"})
                continue
            norm_eid = cls.normalize_entity_id(rec.get("entity_id"))
            if norm_eid is None:
                skipped.append({"match_id": mid, "reason": "INVALID_ENTITY_ID"})
                continue
            if (mid, norm_eid) in existing_keys:
                skipped.append({"match_id": mid, "entity_id": norm_eid, "reason": "DUPLICATE_OBSERVATION"})
                continue

            sd_str = clean_str(rec.get("sales_date"))
            if not is_valid_sales_date(sd_str):
                skipped.append({"match_id": mid, "entity_id": norm_eid, "reason": "INVALID_SALES_DATE"})
                continue
            ko_dt = cls.parse_timestamp(rec.get("kickoff_at"))
            if ko_dt is None:
                skipped.append({"match_id": mid, "entity_id": norm_eid, "reason": "INVALID_KICKOFF_AT"})
                continue
            ct_dt = cls.parse_timestamp(rec.get("cutoff_time"))
            if ct_dt is None:
                skipped.append({"match_id": mid, "entity_id": norm_eid, "reason": "MISSING_OR_INVALID_CUTOFF"})
                continue
            if ct_dt >= ko_dt:
                skipped.append({"match_id": mid, "entity_id": norm_eid, "reason": "NON_PREMATCH_CUTOFF"})
                continue

            raw_hit = rec.get("hit")
            if isinstance(raw_hit, bool):
                hit_val = 1 if raw_hit else 0
            elif isinstance(raw_hit, int) and raw_hit in (0, 1):
                hit_val = int(raw_hit)
            else:
                skipped.append({"match_id": mid, "entity_id": norm_eid, "reason": "INVALID_HIT"})
                continue

            clamped_p0, p_err = cls.clamp_baseline(rec.get("p_baseline"))
            if p_err is not None or clamped_p0 is None:
                skipped.append({"match_id": mid, "entity_id": norm_eid, "reason": p_err or "INVALID_P_BASELINE"})
                continue

            obs_role = clean_str(rec.get("observation_role") or "primary").lower()
            if obs_role not in VALID_OBSERVATION_ROLES:
                skipped.append({"match_id": mid, "entity_id": norm_eid, "reason": "INVALID_OBSERVATION_ROLE"})
                continue

            direction_val = clean_str(rec.get("direction"))
            if direction_val and direction_val not in cls._DIR_1X2_MAP:
                skipped.append({"match_id": mid, "entity_id": norm_eid, "reason": "UNSUPPORTED_DIRECTION"})
                continue

            lt, at, cb = resolve_context_parts(
                context_bucket=rec.get("context_bucket"),
                league=rec.get("league"),
                league_tier=rec.get("league_tier"),
                ah_tier_and_refund=rec.get("ah_tier_and_refund") if rec.get("ah_tier_and_refund") is not None else rec.get("ah_tier"),
            )
            row_obj = cls._build_observation_row(
                match_id=mid,
                sales_date=sd_str,
                kickoff_at=ko_dt.astimezone(BJ_TZ).isoformat(),
                cutoff_time=ct_dt.astimezone(BJ_TZ).isoformat(),
                entity_id=norm_eid,
                entity_type=cls.infer_entity_type(norm_eid, rec.get("entity_type")),
                direction=direction_val,
                p_baseline=clamped_p0,
                hit=hit_val,
                source_card=clean_str(rec.get("source_card")),
                league_tier=lt,
                ah_tier=at,
                context_bucket=cb,
                observation_role=obs_role,
            )
            existing_keys.add((mid, norm_eid))
            accepted.append(row_obj)
            touched_entities.add(norm_eid)

        if not accepted:
            return {
                "status": "ok",
                "ledger_path": str(lpath),
                "added_count": 0,
                "skipped": skipped,
                "total_rows": len(existing),
                "updated_entities": [],
            }

        combined = sorted(existing + accepted, key=cls.sort_key)
        cls.atomic_write_ledger(lpath, combined)
        by_eid = cls._group_by_entity_id(combined, target_ids=touched_entities)

        updated_summaries = [
            cls.evaluate_entity(
                eid,
                cls.infer_entity_type(eid, by_eid[eid][-1].get("entity_type")),
                by_eid[eid],
                prior_strength=prior_strength,
                decay_gamma=decay_gamma,
                already_sorted=True,
            )
            for eid in sorted(touched_entities)
        ]

        return {
            "status": "ok",
            "ledger_path": str(lpath),
            "added_count": len(accepted),
            "skipped": skipped,
            "total_rows": len(combined),
            "updated_entities": updated_summaries,
        }

    @classmethod
    def health_report(
        cls,
        ledger_path: Optional[str] = None,
        entity_ids: Optional[List[str]] = None,
        entity_type: Optional[str] = None,
        entity_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        as_of: Optional[str] = None,
        prior_strength: float = DEFAULT_PRIOR_STRENGTH,
        decay_gamma: float = DEFAULT_DECAY_GAMMA,
        context_bucket: Optional[str] = None,
        include_contexts: bool = False,
        ctx_strength: float = DEFAULT_CTX_STRENGTH,
        ucb_quantile: float = DEFAULT_UCB_QUANTILE,
    ) -> dict:
        hp_err = cls._validate_positive_hyperparams(
            prior_strength=prior_strength, decay_gamma=decay_gamma,
            ctx_strength=ctx_strength, ucb_quantile=ucb_quantile,
        )
        if hp_err is not None:
            return {"status": "error", "error": hp_err}

        filter_ctx, ctx_err = cls._validate_optional_context_bucket(context_bucket)
        if ctx_err is not None:
            return {"status": "error", "error": ctx_err}

        lpath, l_err = cls.resolve_ledger_path(ledger_path)
        if l_err is not None:
            return {"status": "error", "error": l_err}
        assert lpath is not None

        rows, read_err = cls.load_ledger(lpath)
        if read_err is not None:
            return {"status": "error", "error": read_err}

        if entity_ids is not None and not isinstance(entity_ids, (list, tuple, set)):
            return {
                "status": "error",
                "error": {
                    "error_code": "INVALID_ENTITY_IDS",
                    "received": entity_ids,
                    "message": f"entity_ids must be a list/tuple/set of strings or None, got {type(entity_ids).__name__}.",
                },
            }

        non_replay_rows = [r for r in rows if not _is_truthy_flag(r.get("known_answer_replay"))]
        want = {cls.normalize_entity_id(x) or str(x) for x in entity_ids} if entity_ids is not None else None
        by_eid = cls._group_by_entity_id(non_replay_rows, target_ids=want)

        meta_norm: Dict[str, Dict[str, Any]] = {}
        if isinstance(entity_metadata, dict):
            for k, v in entity_metadata.items():
                norm_k = cls.normalize_entity_id(k) or str(k)
                if isinstance(v, dict):
                    meta_norm[norm_k] = v

        all_eids = set(by_eid.keys()) | set(meta_norm.keys())
        if want is not None:
            all_eids = all_eids & want

        filter_type = entity_type.strip().lower() if isinstance(entity_type, str) and entity_type.strip() else None
        entities: List[Dict[str, Any]] = []
        matched_rows: List[Dict[str, Any]] = []
        for eid in sorted(all_eids):
            erows = by_eid.get(eid, [])
            m = meta_norm.get(eid, {})
            explicit_t = m.get("entity_type") or (erows[-1].get("entity_type") if erows else None)
            etype = cls.infer_entity_type(eid, explicit_t)
            if filter_type and etype != filter_type:
                continue
            matched_rows.extend(erows)
            summary = cls.evaluate_entity(
                eid, etype, erows,
                prior_strength=prior_strength, decay_gamma=decay_gamma,
                stale_after=m.get("stale_after"), as_of=as_of,
            )
            show_contexts = bool(include_contexts) or (filter_ctx is not None)
            if show_contexts:
                by_ctx: Dict[str, List[Dict[str, Any]]] = {}
                for r in erows:
                    b = clean_str(r.get("context_bucket")).lower()
                    if filter_ctx:
                        if b == filter_ctx:
                            by_ctx.setdefault(b, []).append(r)
                    elif b in VALID_CONTEXT_BUCKETS:
                        by_ctx.setdefault(b, []).append(r)
                target_buckets = [filter_ctx] if filter_ctx else sorted(by_ctx.keys())
                ctx_map: Dict[str, Dict[str, Any]] = {}
                for bkt in target_buckets:
                    ev = _evaluate_context_from_stats(
                        by_ctx.get(bkt, []),
                        n_global=summary["n_obs"],
                        hits_global=summary["hits"],
                        mean_p0_global=summary["mean_p_baseline"],
                        prior_strength=prior_strength,
                        ctx_strength=ctx_strength,
                        ucb_quantile=ucb_quantile,
                    )
                    ctx_map[bkt] = {
                        "n_obs": ev["n_obs"],
                        "hits": ev["hits"],
                        "mean_p_baseline": ev["mean_p_baseline"],
                        "p_0c_anchor": ev["p_0c_anchor"],
                        "prior_center": ev["prior_center"],
                        "posterior_mean": ev["posterior_mean"],
                        "ci90": ev["ci90"],
                        "ucb90_excess": ev["ucb_excess"],
                        "p_beat_market": ev["p_beat_market"],
                    }
                summary["contexts"] = ctx_map
            entities.append(summary)

        alerts = {
            "promote_candidates": [e["entity_id"] for e in entities if e["advisory_status"] in ("PROMOTE_RECOMMENDED", "STRONG_EVIDENCE")],
            "deprecate_candidates": [e["entity_id"] for e in entities if e["advisory_status"] == "DEPRECATE_RECOMMENDED"],
            "stale_reviews": [e["entity_id"] for e in entities if e["advisory_status"] == "STALE_REVIEW"],
            "drift_watches": [e["entity_id"] for e in entities if e["advisory_status"] == "DRIFT_WATCH"],
            "drift_alerts": [e["entity_id"] for e in entities if e["advisory_status"] == "DRIFT_ALERT"],
        }

        scope_rows = matched_rows if (want is not None or filter_type is not None) else non_replay_rows
        return {
            "status": "ok",
            "algorithm": "Beta-MixtureEProcess-PageCUSUM",
            "ledger_path": str(lpath),
            "total_rows": len(non_replay_rows),
            "entity_count": len(entities),
            "context_filter": filter_ctx,
            "role_counts": cls._aggregate_roles(scope_rows),
            "context_counts": cls._aggregate_context_counts(scope_rows),
            "entities": entities,
            "alerts": alerts,
        }

    @classmethod
    def select_playbooks(
        cls,
        candidate_ids: List[str],
        context_bucket: Optional[str] = None,
        league: str = "",
        league_tier: str = "",
        ah_tier_and_refund: str = "",
        cutoff_time: Optional[str] = None,
        ledger_path: Optional[str] = None,
        mode: str = "bayes_ucb",
        seed_key: Optional[str] = None,
        ucb_quantile: float = DEFAULT_UCB_QUANTILE,
        prior_strength: float = DEFAULT_PRIOR_STRENGTH,
        ctx_strength: float = DEFAULT_CTX_STRENGTH,
        decay_gamma: float = DEFAULT_DECAY_GAMMA,
    ) -> dict:
        if not isinstance(candidate_ids, (list, tuple)):
            return {
                "status": "error",
                "error": {
                    "error_code": "INVALID_CANDIDATE_IDS",
                    "received": candidate_ids,
                    "message": "candidate_ids must be a list of playbook/feature/draft IDs.",
                },
            }
        hp_err = cls._validate_positive_hyperparams(
            ucb_quantile=ucb_quantile, prior_strength=prior_strength,
            ctx_strength=ctx_strength, decay_gamma=decay_gamma,
        )
        if hp_err is not None:
            return {"status": "error", "error": hp_err}

        _, ctx_err = cls._validate_optional_context_bucket(context_bucket)
        if ctx_err is not None:
            return {"status": "error", "error": ctx_err}

        sel_mode = clean_str(mode or "bayes_ucb").lower()
        if sel_mode not in cls.VALID_SELECTION_MODES:
            return {
                "status": "error",
                "error": {
                    "error_code": "INVALID_SELECTION_MODE",
                    "received": mode,
                    "message": f"mode must be one of {cls.VALID_SELECTION_MODES}, got {mode!r}.",
                },
            }

        cutoff_dt = cls.parse_timestamp(cutoff_time)
        if cutoff_dt is None:
            return {
                "status": "error",
                "error": {
                    "error_code": "MISSING_OR_INVALID_CUTOFF",
                    "received": cutoff_time,
                    "message": "Provide valid ISO-8601 cutoff_time to enforce causal zero-leakage filtering.",
                },
            }

        lpath, l_err = cls.resolve_ledger_path(ledger_path)
        if l_err is not None:
            return {"status": "error", "error": l_err}
        assert lpath is not None

        all_rows, read_err = cls.load_ledger(lpath)
        if read_err is not None:
            return {"status": "error", "error": read_err}

        causal_rows = cls._filter_causal_rows(all_rows, cutoff_dt)
        norm_bucket = normalize_context_bucket(
            context_bucket=context_bucket, league=league,
            league_tier=league_tier, ah_tier_and_refund=ah_tier_and_refund,
        )

        valid_ids: List[str] = []
        seen_ids = set()
        skipped: List[str] = []
        for raw_id in candidate_ids:
            nid = cls.normalize_entity_id(raw_id)
            if nid is None:
                skipped.append(str(raw_id))
            elif nid not in seen_ids:
                seen_ids.add(nid)
                valid_ids.append(nid)

        sorted_valid = sorted(valid_ids)
        seed_input = f"{norm_bucket}|{cutoff_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}|{','.join(sorted_valid)}|{seed_key or ''}"
        seed_digest = hashlib.sha256(seed_input.encode("utf-8")).hexdigest()
        rng = random.Random(int(seed_digest[:16], 16))
        by_eid = cls._group_by_entity_id(causal_rows, target_ids=seen_ids)

        eval_map: Dict[str, Dict[str, Any]] = {}
        for eid in sorted_valid:
            etype = cls.infer_entity_type(eid)
            erows = by_eid.get(eid, [])
            g_eval = cls.evaluate_entity(eid, etype, erows, prior_strength=prior_strength, decay_gamma=decay_gamma)
            ctx_rows = (
                [r for r in erows if clean_str(r.get("context_bucket")).lower() == norm_bucket]
                if norm_bucket in VALID_CONTEXT_BUCKETS
                else []
            )
            c_eval = _evaluate_context_from_stats(
                ctx_rows,
                n_global=g_eval["n_obs"],
                hits_global=g_eval["hits"],
                mean_p0_global=g_eval["mean_p_baseline"],
                prior_strength=prior_strength,
                ctx_strength=ctx_strength,
                ucb_quantile=ucb_quantile,
            )
            g_status = g_eval["advisory_status"]
            drift_pen = 0.15 if g_status == "DRIFT_ALERT" else (0.05 if g_status == "DRIFT_WATCH" else 0.0)
            exploit_score = round((c_eval["posterior_mean"] - c_eval["p_0c_anchor"]) - drift_pen, 6)
            ucb_excess = round(c_eval["ucb_excess"], 6)
            sampled_excess = round(rng.betavariate(c_eval["a_c"], c_eval["b_c"]) - c_eval["p_0c_anchor"], 6) if sel_mode == "contextual_ts" else ucb_excess

            eval_map[eid] = {
                "entity_id": eid,
                "entity_type": etype,
                "n_global": c_eval["n_global"],
                "n_context": c_eval["n_obs"],
                "p_0c_anchor": c_eval["p_0c_anchor"],
                "prior_center_c": c_eval["prior_center"],
                "posterior_mean_c": c_eval["posterior_mean"],
                "ucb_c": c_eval["ucb_c"],
                "ucb_excess": ucb_excess,
                "sampled_excess": sampled_excess,
                "exploit_score": exploit_score,
                "advisory_status_global": g_status,
                "_tie_p_beat": float(g_eval["p_beat_market"] or 0.0),
            }

        candidates_out = [{k: v for k, v in eval_map[eid].items() if k != "_tie_p_beat"} for eid in valid_ids]

        caller_rank = {eid: idx for idx, eid in enumerate(valid_ids)}
        primary_seat: Optional[str] = None
        residual_seat: Optional[str] = None
        shadow_candidates: List[str] = []
        tie_break_primary = False
        tie_break_residual = False
        score_key = "sampled_excess" if sel_mode == "contextual_ts" else "ucb_excess"
        if eval_map:
            ordered_items = [eval_map[eid] for eid in valid_ids]

            def _prefer(pool: List[Dict[str, Any]], pred: Any) -> List[Dict[str, Any]]:
                return [x for x in pool if pred(x)] or pool

            def _prefer_playable(pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                return _prefer(_prefer(pool, lambda x: x["entity_type"] != "draft"), lambda x: x["entity_type"] == "playbook")

            primary_pool = _prefer_playable(ordered_items)

            def _select_best_with_tie(
                cand_pool: List[Dict[str, Any]], score_fn: Any
            ) -> Tuple[Dict[str, Any], bool]:
                chosen = min(cand_pool, key=lambda x: (score_fn(x), caller_rank[x["entity_id"]]))
                best_sc = score_fn(chosen)
                is_tie = sum(1 for x in cand_pool if score_fn(x) == best_sc) > 1
                return chosen, is_tie

            def _p_score(x: Dict[str, Any]) -> Tuple[float, float]:
                return (-round(x["exploit_score"], 9), -round(x["_tie_p_beat"], 9))

            def _r_score(x: Dict[str, Any]) -> Tuple[float, float]:
                return (-round(x[score_key], 9), -round(x["exploit_score"], 9))

            best_primary, tie_break_primary = _select_best_with_tie(primary_pool, _p_score)
            primary_seat = best_primary["entity_id"]

            rem = [x for x in ordered_items if x["entity_id"] != primary_seat]
            if rem:
                residual_pool = _prefer_playable(rem)
                residual_candidates = _prefer(residual_pool, lambda x: x["advisory_status_global"] != "DRIFT_ALERT")
                best_residual, tie_break_residual = _select_best_with_tie(residual_candidates, _r_score)
                residual_seat = best_residual["entity_id"]

            shadow_items = [x for x in rem if x["entity_id"] != residual_seat]
            shadow_items.sort(key=lambda x: (_r_score(x), caller_rank[x["entity_id"]]))
            shadow_candidates = [x["entity_id"] for x in shadow_items]

        return {
            "status": "ok",
            "algorithm": "HierarchicalContextualBandit",
            "mode": sel_mode,
            "context_bucket": norm_bucket,
            "causal_rows": len(causal_rows),
            "exploration_triggered": len(valid_ids) >= 3,
            "primary_seat": primary_seat,
            "residual_seat": residual_seat,
            "shadow_candidates": shadow_candidates,
            "tie_break_policy": "caller_order",
            "tie_break_applied": {"primary": tie_break_primary, "residual": tie_break_residual},
            "seed_digest": seed_digest,
            "sampled_scores": {eid: eval_map[eid]["sampled_excess"] for eid in sorted_valid},
            "skipped": skipped,
            "candidates": candidates_out,
        }

    @classmethod
    def verify_ledger_file(cls, ledger_path: str, min_rows: int = 50) -> Tuple[bool, str]:
        p = pathlib.Path(ledger_path)
        rows, err = cls.load_ledger(p)
        if err is not None:
            return False, f"ERR {err}"
        if len(rows) < min_rows:
            return False, f"ERR row_count={len(rows)} < min_rows={min_rows}"
        seen_keys = set()
        prev_key: Optional[Tuple[str, str, str, str]] = None
        for idx, r in enumerate(rows):
            if _is_truthy_flag(r.get("known_answer_replay")):
                return False, f"ERR replay_row_in_ledger at row {idx}"
            mid, eid = clean_str(r.get("match_id")), clean_str(r.get("entity_id"))
            if not mid or not eid or (mid, eid) in seen_keys:
                return False, f"ERR duplicate_or_empty_key at row {idx}: {(mid, eid)!r}"
            seen_keys.add((mid, eid))
            if cls.normalize_entity_id(eid) != eid:
                return False, f"ERR non_canonical_entity_id at {mid}: {eid!r}"
            raw_hit = r.get("hit")
            if isinstance(raw_hit, bool) or raw_hit not in (0, 1):
                return False, f"ERR invalid hit at {(mid, eid)}: {raw_hit!r}"
            pb = r.get("p_baseline")
            if not _is_strict_finite_number(pb) or float(pb) < cls.P_MIN - 1e-9 or float(pb) > cls.P_MAX + 1e-9:
                return False, f"ERR out_of_bounds p_baseline at {(mid, eid)}: {pb!r}"
            ko_dt, ct_dt = cls.parse_timestamp(r.get("kickoff_at")), cls.parse_timestamp(r.get("cutoff_time"))
            if ko_dt is None or ct_dt is None or ct_dt >= ko_dt:
                return False, f"ERR invalid temporal gate at {(mid, eid)}"
            if "observation_role" in r and r["observation_role"] not in VALID_OBSERVATION_ROLES:
                return False, f"ERR invalid observation_role at {(mid, eid)}: {r['observation_role']!r}"
            key = cls.sort_key(r)
            if prev_key is not None and key < prev_key:
                return False, f"ERR unsorted ledger at {(mid, eid)}: {key} < {prev_key}"
            prev_key = key
        digest = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        return True, f"BAYES_LEDGER_OK {len(rows)} sha256={digest}"

    @classmethod
    def bootstrap_from_repo(cls, repo_root: str, output_path: str) -> Tuple[int, str]:
        """Deterministically build Bayesian observation JSONL from settled prematch cards with 12-bucket contexts."""
        rows: List[Dict[str, Any]] = []
        seen_keys = set()

        for mid, sales_date, ko_dt, oc_dt, ident, dec, ms, norm_p, outcome in iter_settled_prematch_cards(repo_root):
            ko_iso = ko_dt.astimezone(BJ_TZ).isoformat()
            ct_iso = oc_dt.astimezone(BJ_TZ).isoformat()
            src_card = f"data/{sales_date}/cards/{mid}.yaml"

            regime = ms.get("regime") or {}
            lt, at, cb = resolve_context_parts(
                league=ident.get("league", ""),
                league_tier=regime.get("league_tier", ""),
                ah_tier_and_refund=regime.get("ah_tier_and_refund", ""),
            )

            def _append_1x2_obs(
                eid: str,
                etype: str,
                dir_key: str,
                stored_dir: str,
                obs_role: str,
            ) -> None:
                if dir_key not in cls._DIR_1X2_MAP or (mid, eid) in seen_keys:
                    return
                indices, wins = cls._DIR_1X2_MAP[dir_key]
                raw_p = sum(norm_p[i] for i in indices)
                p0, _ = cls.clamp_baseline(min(raw_p, 1.0 - 1e-6))
                assert p0 is not None
                seen_keys.add((mid, eid))
                rows.append(
                    cls._build_observation_row(
                        match_id=mid, sales_date=sales_date, kickoff_at=ko_iso, cutoff_time=ct_iso,
                        entity_id=eid, entity_type=etype, direction=stored_dir, p_baseline=p0,
                        hit=1 if outcome in wins else 0, source_card=src_card,
                        league_tier=lt, ah_tier=at, context_bucket=cb, observation_role=obs_role,
                    )
                )

            primary = clean_str(dec.get("primary"))
            if primary in cls._PRIMARY_DIR_LABELS:
                _append_1x2_obs(
                    "portfolio:1x2_primary",
                    "portfolio",
                    primary,
                    cls._PRIMARY_DIR_LABELS[primary],
                    "primary",
                )

            def _append_candidate(item: Any, role_override: Optional[str]) -> None:
                if not isinstance(item, dict):
                    return
                eid = cls.normalize_entity_id(item.get("page"))
                if not eid or not (eid.startswith("playbooks/") or eid.startswith("H")):
                    return
                direction = re.sub(r"[（(][^）)]*[）)]$", "", clean_str(item.get("direction"))).strip()
                if role_override is not None:
                    obs_role = role_override
                else:
                    raw_role = clean_str(item.get("role")).lower()
                    obs_role = raw_role if raw_role in ("primary", "residual") else "primary"
                _append_1x2_obs(eid, cls.infer_entity_type(eid), direction, direction, obs_role)

            for h in (ms.get("wiki_hits") or []):
                _append_candidate(h, None)
            for s_item in ((ms.get("wiki_scan") or {}).get("shadow_candidates") or []):
                _append_candidate(s_item, "shadow")

        rows.sort(key=cls.sort_key)
        out_p = pathlib.Path(output_path)
        cls.atomic_write_ledger(out_p, rows)
        ok, msg = cls.verify_ledger_file(str(out_p), min_rows=0)
        if not ok:
            raise RuntimeError(f"Bayesian bootstrap failed verification: {msg}")
        return len(rows), msg


def select_playbooks(*args: Any, **kwargs: Any) -> dict:
    """Module-level convenience alias delegating to BayesianTrackerEngine.select_playbooks."""
    return BayesianTrackerEngine.select_playbooks(*args, **kwargs)

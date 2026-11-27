"""Pure-stdlib 12-bucket contextual normalization and logit-space single-shrinkage evaluator."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .common import (
    _is_truthy_flag,
    beta_ppf,
    clamp_prob,
    clean_str,
    logit,
    regularized_incomplete_beta,
    sigmoid,
)

LEAGUE_TIERS: Tuple[str, ...] = ("top5", "continental_cup", "national_team", "other_domestic")
AH_TIERS: Tuple[str, ...] = ("even_shallow", "mid", "deep")
VALID_CONTEXT_BUCKETS = frozenset(f"{lt}:{at}" for lt in LEAGUE_TIERS for at in AH_TIERS)
VALID_OBSERVATION_ROLES: Tuple[str, ...] = ("primary", "residual", "shadow")

DEFAULT_PRIOR_STRENGTH: float = 4.0
DEFAULT_CTX_STRENGTH: float = 4.0
DEFAULT_DECAY_GAMMA: float = 0.95
DEFAULT_UCB_QUANTILE: float = 0.90
P_MIN: float = 0.02
P_MAX: float = 0.98

_TOP5_EXACT_OR_PREFIX = re.compile(
    r"^(?:五大联赛|英超|西甲|德甲|意甲|法甲|西班牙甲|英格兰超|德国甲|意大利甲|法国甲)(?:[\s\(\[（A-Za-z0-9]|$)"
    r"|\b(?:english )?premier league\b|\bla liga\b|\bbundesliga\b|\bligue 1\b|\bitalian serie a\b",
    flags=re.IGNORECASE,
)
_CUP_KEYS = (
    "洲际", "欧冠", "欧罗巴", "欧联", "欧协", "欧会杯", "亚冠",
    "解放者", "南美杯", "世俱杯", "europa", "champions", "conference", "libertadores",
)
_NAT_KEYS = (
    "国家队", "欧国联", "世界杯", "欧洲杯", "美洲杯", "亚洲杯", "非洲杯",
    "世预赛", "欧预赛", "亚预赛", "南美预", "友谊赛", "国际赛",
    "nations league", "world cup",
)


def normalize_league_tier(league: Any = "", league_tier: Any = "") -> str:
    """Map league / regime.league_tier strings deterministically onto 4 canonical buckets."""
    lt_raw = clean_str(league_tier)
    lg_raw = clean_str(league)
    if lt_raw.lower() in LEAGUE_TIERS:
        return lt_raw.lower()
    if lg_raw.lower() in LEAGUE_TIERS:
        return lg_raw.lower()

    lt_low = lt_raw.lower()
    lg_low = lg_raw.lower()

    # Check explicit regime tier labels first so "巴西甲" + "次级联赛" never collides with "西甲"
    if any(k in lt_low for k in ("五大", "top5")):
        return "top5"
    if any(k in lt_low for k in ("洲际", "欧战", "continental")):
        return "continental_cup"
    if any(k in lt_low for k in ("国家队", "national")):
        return "national_team"
    if any(k in lt_low for k in ("次级", "非五大", "国内", "小联赛", "other", "domestic", "顶级联赛")):
        return "other_domestic"

    for key in _NAT_KEYS:
        if key in lg_low:
            return "national_team"
    for key in _CUP_KEYS:
        if key in lg_low:
            return "continental_cup"
    if _TOP5_EXACT_OR_PREFIX.search(lg_low) or (lg_low == "serie a"):
        return "top5"

    if (lg_raw and lg_low != "unknown") or (lt_raw and lt_low != "unknown"):
        return "other_domestic"
    return "unknown"


def _classify_ah_line(val: float) -> str:
    if val < 0.75 - 1e-9:
        return "even_shallow"
    if val <= 1.0 + 1e-9:
        return "mid"
    return "deep"


def normalize_ah_tier(ah_tier_and_refund: Any = "") -> str:
    """Classify multi-clause AH regime strings into even_shallow | mid | deep | unknown."""
    raw = clean_str(ah_tier_and_refund)
    if not raw or raw.lower() == "unknown":
        return "unknown"
    if raw.lower() in AH_TIERS:
        return raw.lower()

    head = re.split(r"非原", raw, maxsplit=1)[0]
    head = re.sub(r"退档自\s*[+-]?\d+(?:\.\d+)?(?:/[+-]?\d+(?:\.\d+)?)?", " ", head)
    raw_clauses = [c.strip() for c in re.split(r"[;；|]", head) if c.strip()]
    if not raw_clauses:
        return "unknown"

    def _strip_noise(text: str) -> str:
        t = re.sub(r"[,，、]\s*(?:香港)?马会.*$", " ", text)
        t = re.sub(r"\b(?:full|half|no)_refund\b", " ", t, flags=re.IGNORECASE)
        t = re.sub(r"\b188bet\b", " ", t, flags=re.IGNORECASE)
        t = re.sub(r"\bcid[:=\s]*\d+(?:\s*[/,]\s*\d+)*", " ", t, flags=re.IGNORECASE)
        t = re.sub(r"(?<![.\d-])\d+(?:\s*/\s*\d+)*\s*家", " ", t)
        t = re.sub(r"[（(]\s*\d+(?:\s*/\s*\d+)*\s*[）)]", " ", t)
        t = re.sub(r"(?<![.\d-])\d+(?:\s*/\s*\d+)+(?![.\d])", " ", t)
        return re.sub(r"^[，,\s]+|[，,\s]+$", "", re.sub(r"\s+", " ", t))

    cleaned = [(_strip_noise(c), c) for c in raw_clauses]
    cleaned = [(cl, orig) for cl, orig in cleaned if cl] or [(raw, raw)]

    non_hkjc = [item for item in cleaned if not re.match(r"^(?:香港)?马会", item[0])]
    pool = non_hkjc or cleaned

    mainstream = [item for item in pool if re.search(r"最新|现主流|终盘|当前主流|主流|六家|七家|多数", item[1])]
    arrowed = [item for item in pool if re.search(r"->|→|退至|升至|转为|转", item[1])]
    line_bearing = [
        item for item in pool
        if re.search(r"深盘|中盘|浅盘|平手|平半|半一|一球|球半|两球|\d(?:\.\d+)?/\d|\d\.\d{1,2}", item[0])
    ]
    chosen_clause = (mainstream or arrowed or line_bearing or pool)[-1][0]
    chosen_clause = re.split(r"[,，、]\s*(?:香港)?马会", chosen_clause, maxsplit=1)[0].strip() or chosen_clause
    step = re.split(r"->|→|退至|升至|转为|转", chosen_clause)[-1].strip() or chosen_clause

    def _classify_from_text(segment: str) -> Optional[str]:
        q_matches = [
            0.5 * (abs(float(m.group(1))) + abs(float(m.group(2))))
            for m in re.finditer(r"([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)", segment)
            if abs(float(m.group(1))) <= 6.0 and abs(float(m.group(2))) <= 6.0
        ]
        if q_matches:
            return _classify_ah_line(q_matches[-1])

        num_matches = [
            abs(float(m.group(1)))
            for m in re.finditer(r"([+-]?\d+(?:\.\d+)?)", segment)
            if abs(float(m.group(1))) <= 6.0 and abs(abs(float(m.group(1))) * 4.0 - round(abs(float(m.group(1))) * 4.0)) < 1e-6
        ]
        if num_matches:
            return _classify_ah_line(num_matches[-1])

        # Prefer explicit Chinese handicap line names over coarser tier labels (e.g. "中盘/客让半球" -> even_shallow)
        if re.search(r"两球|球半|一球半", segment):
            return "deep"
        if re.search(r"一球|半一", segment):
            return "mid"
        if re.search(r"平手|平半|半球", segment):
            return "even_shallow"
        if re.search(r"深盘|深让", segment):
            return "deep"
        if re.search(r"中盘|中让", segment):
            return "mid"
        if re.search(r"浅盘|均势", segment):
            return "even_shallow"
        return None

    return _classify_from_text(step) or _classify_from_text(chosen_clause) or _classify_from_text(head) or "unknown"


_ENRICH_NUM_OR_SPLIT_RE = re.compile(r"([+-]?)\s*(\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?")


def enrich_ah_regime(ah_regime: str, ah_pick: str) -> str:
    """Attach numeric line from card AH pick when regime text omitted digits (shared by sync and bootstrap)."""
    if not ah_regime or re.search(r"\d", ah_regime) or not ah_pick:
        return ah_regime
    cleaned = re.sub(r"[（(][^）)]*[）)]", " ", ah_pick)
    m = _ENRICH_NUM_OR_SPLIT_RE.search(cleaned)
    if not m:
        return ah_regime
    _, a_s, b_s = m.groups()
    mag = (float(a_s) + float(b_s)) / 2.0 if b_s is not None else float(a_s)
    line_str = f"{-abs(mag):g}" if mag != 0.0 else "0.0"
    if ";" in ah_regime:
        first, rest = ah_regime.split(";", 1)
        return f"{first.strip()}({line_str}) ;{rest}"
    return f"{ah_regime.strip()}({line_str})"


def resolve_context_parts(
    context_bucket: Optional[str] = None,
    league: Any = "",
    league_tier: Any = "",
    ah_tier_and_refund: Any = "",
) -> Tuple[str, str, str]:
    """Resolve (league_tier, ah_tier, context_bucket) cleanly with case-insensitive fast-path."""
    if isinstance(context_bucket, str) and context_bucket.strip():
        cb = context_bucket.strip().lower()
        if cb in VALID_CONTEXT_BUCKETS:
            lt, at = cb.split(":", 1)
            return lt, at, cb
    lt = normalize_league_tier(league, league_tier)
    at = normalize_ah_tier(ah_tier_and_refund)
    cb = f"{lt}:{at}" if (lt != "unknown" and at != "unknown") else "unknown"
    return lt, at, cb


def normalize_context_bucket(
    context_bucket: Optional[str] = None,
    league: Any = "",
    league_tier: Any = "",
    ah_tier_and_refund: Any = "",
) -> str:
    """Resolve explicit or raw metadata fields into one of 12 canonical buckets or 'unknown'."""
    return resolve_context_parts(context_bucket, league, league_tier, ah_tier_and_refund)[2]


def _evaluate_context_from_stats(
    ctx_rows: List[Dict[str, Any]],
    n_global: int,
    hits_global: int,
    mean_p0_global: Optional[float],
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    ctx_strength: float = DEFAULT_CTX_STRENGTH,
    ucb_quantile: float = DEFAULT_UCB_QUANTILE,
) -> Dict[str, Any]:
    """Compute single-shrinkage logit-space Beta posterior & Bayes-UCB from pre-aggregated global stats."""
    n_c = len(ctx_rows)
    s_c = sum(int(r["hit"]) for r in ctx_rows)
    mean_p0_c = (
        sum(clamp_prob(float(r["p_baseline"]), P_MIN, P_MAX) for r in ctx_rows) / n_c
        if n_c > 0
        else None
    )

    g_anchor = mean_p0_global if (n_global > 0 and mean_p0_global is not None) else 0.50
    p_0c_anchor = mean_p0_c if mean_p0_c is not None else g_anchor
    m0 = max(0.5, float(prior_strength))
    # Leave-context-out conjugate-smoothed transfer eliminates global/context double-counting
    # and the unsmoothed 0/1 -> P_MIN=0.02 logit cliff.
    n_other = max(0, n_global - n_c)
    s_other = max(0, min(n_other, hits_global - s_c))
    if n_other > 0:
        if n_c > 0 and mean_p0_c is not None and mean_p0_global is not None:
            raw_other_anchor = (n_global * mean_p0_global - n_c * mean_p0_c) / n_other
            g_other_anchor = clamp_prob(raw_other_anchor, P_MIN, P_MAX)
        else:
            g_other_anchor = clamp_prob(g_anchor, P_MIN, P_MAX)
        p_smooth_other = (s_other + m0 * g_other_anchor) / (n_other + m0)
        delta_logit = logit(p_smooth_other, P_MIN, P_MAX) - logit(g_other_anchor, P_MIN, P_MAX)
        mu_0c = clamp_prob(sigmoid(logit(p_0c_anchor, P_MIN, P_MAX) + delta_logit), P_MIN, P_MAX)
    else:
        mu_0c = clamp_prob(p_0c_anchor, P_MIN, P_MAX)
    m_ctx = max(0.5, float(ctx_strength))
    a_c = m_ctx * mu_0c + s_c
    b_c = m_ctx * (1.0 - mu_0c) + (n_c - s_c)

    post_mean_c = a_c / (a_c + b_c)
    ci_lo = beta_ppf(0.05, a_c, b_c)
    ci_hi = beta_ppf(0.95, a_c, b_c)
    ucb_c = beta_ppf(max(0.50, min(0.995, float(ucb_quantile))), a_c, b_c)
    p_below_c = regularized_incomplete_beta(clamp_prob(p_0c_anchor, P_MIN, P_MAX), a_c, b_c)

    return {
        "n_global": n_global,
        "hits_global": hits_global,
        "n_obs": n_c,
        "hits": s_c,
        "mean_p_baseline": round(mean_p0_c, 6) if mean_p0_c is not None else None,
        "p_0c_anchor": round(p_0c_anchor, 6),
        "prior_center": round(mu_0c, 6),
        "a_c": round(a_c, 6),
        "b_c": round(b_c, 6),
        "posterior_mean": round(post_mean_c, 6),
        "ci90": [round(ci_lo, 6), round(ci_hi, 6)],
        "ucb_c": round(ucb_c, 6),
        "ucb_excess": round(ucb_c - p_0c_anchor, 6),
        "p_beat_market": round(1.0 - p_below_c, 6),
    }


def evaluate_context(
    rows: List[Dict[str, Any]],
    entity_id: str,
    context_bucket: str,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    ctx_strength: float = DEFAULT_CTX_STRENGTH,
    ucb_quantile: float = DEFAULT_UCB_QUANTILE,
) -> Dict[str, Any]:
    """Compute single-shrinkage logit-space hierarchical Beta metrics for (entity_id, context_bucket)."""
    from .bayesian import BayesianTrackerEngine

    norm_eid = BayesianTrackerEngine.normalize_entity_id(entity_id) or str(entity_id)
    ent_rows = [
        r for r in rows
        if not _is_truthy_flag(r.get("known_answer_replay"))
        and (BayesianTrackerEngine.normalize_entity_id(r.get("entity_id")) or str(r.get("entity_id", ""))) == norm_eid
    ]
    n_global = len(ent_rows)
    hits_global = sum(int(r["hit"]) for r in ent_rows)
    mean_p0_global = (
        sum(clamp_prob(float(r["p_baseline"]), P_MIN, P_MAX) for r in ent_rows) / n_global
        if n_global > 0
        else None
    )
    ctx_rows = (
        [r for r in ent_rows if clean_str(r.get("context_bucket")).lower() == context_bucket]
        if context_bucket in VALID_CONTEXT_BUCKETS
        else []
    )
    return _evaluate_context_from_stats(
        ctx_rows,
        n_global=n_global,
        hits_global=hits_global,
        mean_p0_global=mean_p0_global,
        prior_strength=prior_strength,
        ctx_strength=ctx_strength,
        ucb_quantile=ucb_quantile,
    )

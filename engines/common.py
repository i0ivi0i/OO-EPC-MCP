"""Shared pure-stdlib constants, math primitives, validation, card iterator, and atomic JSONL helpers."""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

CLASSES: Tuple[str, str, str] = ("H", "D", "A")
BJ_TZ = timezone(timedelta(hours=8))
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TS_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2})(?::(\d{2}))?(?:\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?"
)


def clean_str(val: Any) -> str:
    """Safely normalize optional string values without turning None into 'None'."""
    return "" if val is None else str(val).strip()


def is_valid_sales_date(value: Any) -> bool:
    """Return True if value parses to a strict YYYY-MM-DD sales date string."""
    s = str(value or "").strip()
    return bool(s and _DATE_RE.match(s))


def _is_truthy_flag(val: Any) -> bool:
    """Return True if val is boolean True or a truthy replay string ('true', '1', 'yes')."""
    if val is True:
        return True
    if isinstance(val, str) and val.strip().lower() in ("true", "1", "yes"):
        return True
    return False


def _is_strict_finite_number(val: Any) -> bool:
    """Return True only for finite real numbers (explicitly rejecting bool, NaN, inf)."""
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def clamp_prob(p: float, p_min: float = 0.02, p_max: float = 0.98) -> float:
    """Clamp probability cleanly within [p_min, p_max]."""
    return max(p_min, min(p_max, float(p)))


def logit(p: float, p_min: float = 0.02, p_max: float = 0.98) -> float:
    """Log-odds transform after clamping away from 0/1 singularities."""
    pc = clamp_prob(p, p_min=p_min, p_max=p_max)
    return math.log(pc / (1.0 - pc))


def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _betacf(a: float, b: float, x: float) -> float:
    """Modified Lentz continued fraction for incomplete beta I_x(a, b)."""
    maxit = 200
    eps = 3.0e-12
    fpmin = 1.0e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(
    x: float,
    a: float,
    b: float,
    _log_beta: Optional[float] = None,
) -> float:
    """Evaluate regularized incomplete beta I_x(a, b) = P(Beta(a, b) <= x) using pure stdlib."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = _log_beta if _log_beta is not None else (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))
    bt = math.exp(log_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return max(0.0, min(1.0, bt * _betacf(a, b, x) / a))
    return max(0.0, min(1.0, 1.0 - bt * _betacf(b, a, 1.0 - x) / b))


def beta_ppf(q: float, a: float, b: float, iters: int = 60) -> float:
    """Invert regularized incomplete beta I_x(a, b) = q via monotonic bisection."""
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if regularized_incomplete_beta(mid, a, b, _log_beta=log_beta) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Extract the first timestamp from a string, defaulting missing tz to +08:00, and return UTC datetime."""
    if not value:
        return None
    match = _TS_RE.search(str(value).strip())
    if not match:
        return None
    date_part, hm_part, sec_part, tz_part = match.groups()
    hm_part = hm_part or "00:00"
    sec_part = sec_part or "00"
    try:
        base = datetime.strptime(f"{date_part} {hm_part}:{sec_part}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    if not tz_part:
        tz = BJ_TZ
    elif tz_part == "Z":
        tz = timezone.utc
    else:
        tz_clean = tz_part.replace(":", "")
        sign = 1 if tz_clean[0] == "+" else -1
        hrs = int(tz_clean[1:3])
        mins = int(tz_clean[3:5]) if len(tz_clean) >= 5 else 0
        tz = timezone(timedelta(hours=sign * hrs, minutes=sign * mins))
    return base.replace(tzinfo=tz).astimezone(timezone.utc)


def normalize_probs(probs: Any, context: str = "probs") -> Tuple[Optional[List[float]], Optional[dict]]:
    """Validate 3-class probabilities (list/tuple or H/D/A or home/draw/away dict; 0..1 or 0..100%) and normalize to sum 1.0."""
    seq: Any = probs
    if isinstance(probs, dict):
        lowered = {str(k).strip().lower(): v for k, v in probs.items()}
        if len(lowered) == len(probs) and set(lowered.keys()) == {"h", "d", "a"}:
            seq = [lowered["h"], lowered["d"], lowered["a"]]
        elif len(lowered) == len(probs) and set(lowered.keys()) == {"home", "draw", "away"}:
            seq = [lowered["home"], lowered["draw"], lowered["away"]]
        else:
            return None, {
                "error_code": "INVALID_PROBS_DICT",
                "field": context,
                "received": probs,
                "message": f"{context} dict must have exact keys {{'H','D','A'}} or {{'home','draw','away'}}, got {sorted(str(k) for k in probs.keys())!r}.",
            }
    elif not isinstance(probs, (list, tuple)) or len(probs) != 3:
        return None, {
            "error_code": "INVALID_PROBS_LENGTH",
            "field": context,
            "received": probs,
            "message": f"{context} must be a 3-element list [p_home, p_draw, p_away] (or {{'H','D','A'}}/{{'home','draw','away'}} dict), got {probs!r}.",
        }
    vals: List[float] = []
    for idx, raw in enumerate(seq):
        try:
            fval = float(raw)
        except (TypeError, ValueError):
            return None, {
                "error_code": "NON_NUMERIC_PROB",
                "field": f"{context}[{idx}]",
                "received": raw,
                "message": f"{context}[{idx}] must be numeric, got {raw!r}.",
            }
        if not math.isfinite(fval) or fval <= 0.0:
            return None, {
                "error_code": "PROB_OUT_OF_RANGE",
                "field": f"{context}[{idx}]",
                "received": fval,
                "message": f"{context}[{idx}] must be strictly positive, got {fval}.",
            }
        vals.append(fval)
    total = sum(vals)
    if total > 2.0:
        vals = [v / 100.0 for v in vals]
        total = sum(vals)
    if total < 0.85 or total > 1.15:
        return None, {
            "error_code": "PROBS_SUM_OUT_OF_RANGE",
            "field": context,
            "received": vals,
            "message": f"Normalized sum of {context} ({total:.4f}) is outside [0.85, 1.15].",
        }
    return [v / total for v in vals], None


def resolve_ledger_path(
    ledger_path: Optional[str],
    env_var: str,
    default_rel: str,
) -> Tuple[Optional[pathlib.Path], Optional[dict]]:
    """Resolve an absolute JSONL ledger path, allowing env_var fallback."""
    env_path = os.environ.get(env_var, "").strip()
    raw = (ledger_path or "").strip()
    if not raw:
        if not env_path:
            return None, {
                "error_code": "INVALID_LEDGER_PATH",
                "message": f"ledger_path is required when {env_var} is unset. Pass an absolute path to {default_rel}.",
            }
        p = pathlib.Path(env_path)
        if not p.is_absolute():
            return None, {
                "error_code": "INVALID_LEDGER_PATH",
                "message": f"{env_var} must be an absolute path, got {env_path!r}.",
            }
        return p, None
    p = pathlib.Path(raw)
    if not p.is_absolute():
        if env_path and pathlib.Path(env_path).is_absolute():
            return pathlib.Path(env_path), None
        return None, {
            "error_code": "INVALID_LEDGER_PATH",
            "received": raw,
            "message": f"ledger_path must be an absolute path (got relative {raw!r}) when {env_var} is unset.",
        }
    return p, None


def load_ledger(path: pathlib.Path) -> Tuple[List[Dict[str, Any]], Optional[dict]]:
    """Load JSONL ledger (tolerating leading UTF-8 BOM); return empty list if file does not exist yet."""
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as fh:
            for lineno, line in enumerate(fh, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError as exc:
                    return [], {
                        "error_code": "MALFORMED_LEDGER_LINE",
                        "line": lineno,
                        "message": f"Invalid JSON at {path}:{lineno}: {exc}",
                    }
                if isinstance(obj, dict):
                    rows.append(obj)
    except FileNotFoundError:
        return [], None
    except OSError as exc:
        return [], {"error_code": "LEDGER_READ_ERROR", "message": str(exc)}
    return rows, None


def atomic_write_ledger(path: pathlib.Path, rows: List[Dict[str, Any]], prefix: str = ".ledger_") -> None:
    """Atomically persist JSONL rows via same-directory temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=prefix, suffix=".jsonl.tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def iter_settled_prematch_cards(repo_root: str, sales_date: Optional[str] = None):
    """Yield validated settled prematch card tuples shared by ConformalEngine, BayesianTrackerEngine, and sync_day_ledgers."""
    try:
        import yaml  # Lazy import so online MCP stays 100% stdlib
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for card-based ledger sync/bootstrap (install with `pip install 'oo-epc-mcp[bootstrap]'`)."
        ) from exc

    root = pathlib.Path(repo_root)
    data_dir = root / "data"
    seen_ids = set()

    if sales_date is not None:
        target_dir = data_dir / sales_date
        day_dirs = [target_dir] if target_dir.is_dir() and _DATE_RE.match(sales_date) else []
    else:
        day_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir() and _DATE_RE.match(p.name)) if data_dir.is_dir() else []

    for day_dir in day_dirs:
        cards_dir = day_dir / "cards"
        if not cards_dir.is_dir():
            continue
        sales_date = day_dir.name
        scores_map: Dict[str, Tuple[int, int]] = {}
        for src_name in ("meta.md", "journal.md"):
            src_file = day_dir / src_name
            try:
                txt = src_file.read_text(encoding="utf-8-sig", errors="ignore")
            except OSError:
                continue
            for line in txt.splitlines():
                if not (line.startswith("|") or "[SCORE-PROVENANCE-GATE]" in line):
                    continue
                m_id = re.search(r"\b(\d{7})\b", line)
                if not m_id:
                    continue
                mid = m_id.group(1)
                if mid in scores_map:
                    continue
                m_ft = re.search(r"full_time=(\d{1,2}):(\d{1,2})", line)
                if m_ft:
                    scores_map[mid] = (int(m_ft.group(1)), int(m_ft.group(2)))
                    continue
                if any(k in line for k in ("完场", "✅", "❌")):
                    for m_sc in re.finditer(r"(?<!\d)(\d{1,2}):(\d{1,2})(?!\d)", line):
                        h, a = int(m_sc.group(1)), int(m_sc.group(2))
                        before = line[max(0, m_sc.start() - 6):m_sc.start()]
                        if h <= 12 and a <= 12 and not re.search(r"\d{2}-\d{2}\s*$", before):
                            scores_map[mid] = (h, a)
                            break

        for card_path in sorted(cards_dir.glob("*.yaml")):
            mid = card_path.stem
            if mid in seen_ids or mid not in scores_map:
                continue
            doc = yaml.safe_load(card_path.read_text(encoding="utf-8-sig", errors="replace"))
            if not isinstance(doc, dict):
                continue
            ann = doc.get("announce") or {}
            ca = ann.get("comprehensive_assessment") or {}
            ms = ann.get("memory_summary") or {}
            ident = ms.get("identity") or {}
            dref = ms.get("data_ref") or {}
            dec = ms.get("decision") or {}
            res = dref.get("research") or {}

            if (
                _is_truthy_flag(doc.get("known_answer_replay"))
                or _is_truthy_flag(ms.get("known_answer_replay"))
                or _is_truthy_flag(dref.get("known_answer_replay"))
            ):
                continue
            phase = str(res.get("analysis_phase", "prematch")).strip().lower()
            if phase not in ("", "prematch"):
                continue

            ko_dt = parse_timestamp(str(ident.get("kickoffTime", "")).strip())
            oc_dt = parse_timestamp(str(dref.get("odds_cutoff", "")).strip())
            if ko_dt is None or oc_dt is None or oc_dt >= ko_dt:
                continue

            probs_raw: Optional[List[float]] = None
            p_block = dec.get("probabilities")
            if isinstance(p_block, dict) and all(k in p_block for k in ("home", "draw", "away")):
                probs_raw = [p_block["home"], p_block["draw"], p_block["away"]]
            else:
                oo_str = str(ca.get("oo_epc_probabilities", ""))
                m_oo = re.search(r"胜\s*([0-9.]+)%\s*/\s*平\s*([0-9.]+)%\s*/\s*负\s*([0-9.]+)%", oo_str)
                if m_oo:
                    probs_raw = [float(m_oo.group(1)), float(m_oo.group(2)), float(m_oo.group(3))]
            if probs_raw is None:
                continue
            norm_p, p_err = normalize_probs(probs_raw, context=mid)
            if p_err is not None or norm_p is None:
                continue

            hg, ag = scores_map[mid]
            outcome = "H" if hg > ag else ("D" if hg == ag else "A")
            seen_ids.add(mid)
            yield mid, sales_date, ko_dt, oc_dt, ident, dec, ms, norm_p, outcome

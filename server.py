"""sports-math MCP Server entrypoint: thin FastMCP wrappers, legacy re-exports, and CLI."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional

_PKG_ROOT = str(pathlib.Path(__file__).resolve().parent)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from engines import (  # noqa: E402
    BayesianTrackerEngine,
    ConformalEngine,
    OoEpcCalculator,
    PoissonEngine,
    _betacf,
    _is_strict_finite_number,
    _is_truthy_flag,
    beta_ppf,
    evaluate_context,
    normalize_ah_tier,
    normalize_context_bucket,
    normalize_league_tier,
    regularized_incomplete_beta,
    select_playbooks,
)

try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    try:
        from fastmcp import FastMCP  # type: ignore[import-not-found]
    except ImportError:
        from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]

mcp = FastMCP(
    name="sports-math",
    description="Mathematical computation engine for sports betting odds devigging, score probability modeling, conformal calibration, and contextual Bayesian/e-process playbook selection.",
    version="1.4.0",
)


@mcp.tool()
def devig_oo_epc(home_odds: float, draw_odds: float, away_odds: float) -> dict:
    """Calculate unbiased 1X2 probabilities from decimal odds via Goto (2026) OO-EPC.

    Args:
        home_odds: Home win decimal odds (>= 1.0).
        draw_odds: Draw decimal odds (>= 1.0).
        away_odds: Away win decimal odds (>= 1.0).
    """
    parsed, err = OoEpcCalculator.validate_triple([home_odds, draw_odds, away_odds], context="devig_oo_epc")
    if err is not None:
        return {"status": "error", "algorithm": "Goto-OO-EPC", "error": err}
    assert parsed is not None
    return OoEpcCalculator.solve_triple(parsed)


@mcp.tool()
def devig_oo_epc_batch(items: List[Dict[str, Any]]) -> dict:
    """Batch OO-EPC devigging across bookmakers with opening/latest shifts and Pinnacle anchor gap.

    Args:
        items: List of dicts with cid, optional name, open ([h,d,a]), and latest ([h,d,a]).
    """
    return OoEpcCalculator.solve_batch(items)


@mcp.tool()
def poisson_dixon_coles(home_lambda: float, away_lambda: float, max_goals: int = 8, top_n: int = 6) -> dict:
    """Calculate bivariate Poisson match score probabilities with Dixon-Coles adjustments.

    Args:
        home_lambda: Expected home goals.
        away_lambda: Expected away goals.
        max_goals: Max goals per team grid (default 8).
        top_n: Top scorelines returned (default 6).
    """
    return PoissonEngine.calculate(home_lambda, away_lambda, max_goals=max_goals, top_n=top_n)


@mcp.tool()
def conformal_predict_1x2(
    probs: List[float],
    cutoff_time: Optional[str] = None,
    cutoff_sales_date: Optional[str] = None,
    alpha: float = 0.20,
    window_size: int = 150,
    ledger_path: Optional[str] = None,
) -> dict:
    """Compute causal 1X2 conformal set (LAC), Mondrian draw alert, ACI alpha_t, and Venn-Abers intervals.

    Args:
        probs: 3-element [p_home, p_draw, p_away] probabilities.
        cutoff_time: ISO-8601 causal cutoff timestamp.
        cutoff_sales_date: YYYY-MM-DD fallback cutoff.
        alpha: Target miscoverage rate (default 0.20).
        window_size: Rolling causal calibration window (default 150).
        ledger_path: Absolute path to data/conformal_calibration.jsonl.
    """
    return ConformalEngine.predict_1x2(
        probs=probs,
        cutoff_time=cutoff_time,
        cutoff_sales_date=cutoff_sales_date,
        alpha=alpha,
        window_size=window_size,
        ledger_path=ledger_path,
    )


@mcp.tool()
def conformal_record_outcomes(
    records: List[Dict[str, Any]],
    ledger_path: Optional[str] = None,
    alpha_target: float = 0.20,
    gamma: float = 0.05,
) -> dict:
    """Idempotently append settled pre-match 1X2 outcomes to conformal_calibration.jsonl.

    Args:
        records: List of settled match dicts (match_id, sales_date, kickoff_at, cutoff_time, league, probs, outcome, source_card, known_answer_replay).
        ledger_path: Absolute path to data/conformal_calibration.jsonl.
        alpha_target: Target miscoverage rate (default 0.20).
        gamma: ACI step size (default 0.05).
    """
    return ConformalEngine.record_outcomes(
        records=records,
        ledger_path=ledger_path,
        alpha_target=alpha_target,
        gamma=gamma,
    )


@mcp.tool()
def bayesian_record_outcomes(
    records: List[Dict[str, Any]],
    ledger_path: Optional[str] = None,
    prior_strength: float = 4.0,
    decay_gamma: float = 0.95,
) -> dict:
    """Idempotently append settled pre-match contextual Bayesian observations (primary/residual/shadow).

    Args:
        records: List of settled dicts (match_id, sales_date, kickoff_at, cutoff_time, entity_id, p_baseline, hit, source_card, optional direction in 1X2 single/double-chance 主胜/平/客胜/主不败/客不败, league, league_tier, ah_tier_and_refund, context_bucket, observation_role='primary'|'residual'|'shadow').
        ledger_path: Absolute path to data/bayesian_tracker.jsonl.
        prior_strength: Global shrinkage pseudo-count M_0 (default 4.0).
        decay_gamma: Exponential forgetting factor (default 0.95).
    """
    return BayesianTrackerEngine.record_outcomes(
        records=records,
        ledger_path=ledger_path,
        prior_strength=prior_strength,
        decay_gamma=decay_gamma,
    )


@mcp.tool()
def bayesian_health_report(
    ledger_path: Optional[str] = None,
    entity_ids: Optional[List[str]] = None,
    entity_type: Optional[str] = None,
    entity_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    as_of: Optional[str] = None,
    prior_strength: float = 4.0,
    decay_gamma: float = 0.95,
    context_bucket: Optional[str] = None,
    include_contexts: bool = False,
) -> dict:
    """Evaluate global & 12-bucket contextual Beta posteriors, e-processes, role_counts, context_counts, and Page-CUSUM drift alerts.

    Args:
        ledger_path: Absolute path to data/bayesian_tracker.jsonl.
        entity_ids: Optional filter list of entity IDs.
        entity_type: Optional filter ('draft', 'playbook', 'feature', 'portfolio').
        entity_metadata: Optional map {entity_id: {entity_type, stale_after}}.
        as_of: Optional ISO-8601 timestamp for evaluating stale_after deadlines.
        prior_strength: Global prior pseudo-count M_0 (default 4.0).
        decay_gamma: Exponential forgetting factor (default 0.95).
        context_bucket: Optional 12-bucket filter ('top5:mid', etc.; auto-enables contexts for that bucket).
        include_contexts: Attach per-entity contexts map (default False for compact output).
    """
    return BayesianTrackerEngine.health_report(
        ledger_path=ledger_path,
        entity_ids=entity_ids,
        entity_type=entity_type,
        entity_metadata=entity_metadata,
        as_of=as_of,
        prior_strength=prior_strength,
        decay_gamma=decay_gamma,
        context_bucket=context_bucket,
        include_contexts=include_contexts,
    )


@mcp.tool()
def bayesian_select_playbooks(
    candidate_ids: List[str],
    context_bucket: Optional[str] = None,
    league: str = "",
    league_tier: str = "",
    ah_tier_and_refund: str = "",
    cutoff_time: Optional[str] = None,
    ledger_path: Optional[str] = None,
    mode: str = "bayes_ucb",
    seed_key: Optional[str] = None,
    ucb_quantile: float = 0.90,
) -> dict:
    """Select exploitation primary_seat and Bayes-UCB / deterministic seeded CTS residual_seat under causal cutoff (caller_order tie-break).

    Args:
        candidate_ids: Candidate playbook/feature/draft IDs to rank (order breaks equal-score ties).
        context_bucket: Optional explicit 12-bucket key ('top5:mid', etc.).
        league: League display name (used when context_bucket is omitted).
        league_tier: League tier string from Step 1 regime.
        ah_tier_and_refund: AH regime string from Step 1 regime.
        cutoff_time: Strict ISO-8601 causal cutoff timestamp (enforces kickoff_at <= cutoff_time - 3h).
        ledger_path: Absolute path to data/bayesian_tracker.jsonl.
        mode: 'bayes_ucb' (default deterministic UCB) or 'contextual_ts' (SHA-256 seeded CTS).
        seed_key: Optional per-match salt (e.g. match_id) for contextual_ts RNG isolation.
        ucb_quantile: Beta quantile for Bayes-UCB residual scoring (default 0.90).
    """
    return BayesianTrackerEngine.select_playbooks(
        candidate_ids=candidate_ids,
        context_bucket=context_bucket,
        league=league,
        league_tier=league_tier,
        ah_tier_and_refund=ah_tier_and_refund,
        cutoff_time=cutoff_time,
        ledger_path=ledger_path,
        mode=mode,
        seed_key=seed_key,
        ucb_quantile=ucb_quantile,
    )


def _cli_main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--bootstrap-ledger", action="store_true")
    parser.add_argument("--bootstrap-bayesian-ledger", action="store_true")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default="data/conformal_calibration.jsonl")
    parser.add_argument("--verify-ledger", default=None)
    parser.add_argument("--verify-bayesian-ledger", default=None)
    parser.add_argument("--min-rows", type=int, default=50)
    args, _ = parser.parse_known_args()

    if args.bootstrap_ledger:
        _, msg = ConformalEngine.bootstrap_from_repo(args.repo_root, args.output)
        print(msg)
        return 0
    if args.bootstrap_bayesian_ledger:
        out_path = args.output if args.output != "data/conformal_calibration.jsonl" else "data/bayesian_tracker.jsonl"
        _, msg = BayesianTrackerEngine.bootstrap_from_repo(args.repo_root, out_path)
        print(msg)
        return 0
    if args.verify_ledger:
        ok, msg = ConformalEngine.verify_ledger_file(args.verify_ledger, min_rows=args.min_rows)
        print(msg)
        return 0 if ok else 1
    if args.verify_bayesian_ledger:
        ok, msg = BayesianTrackerEngine.verify_ledger_file(args.verify_bayesian_ledger, min_rows=args.min_rows)
        print(msg)
        return 0 if ok else 1
    mcp.run()
    return 0


__all__ = [
    "BayesianTrackerEngine", "ConformalEngine", "OoEpcCalculator", "PoissonEngine",
    "_betacf", "_is_strict_finite_number", "_is_truthy_flag", "beta_ppf",
    "bayesian_health_report", "bayesian_record_outcomes", "bayesian_select_playbooks",
    "conformal_predict_1x2", "conformal_record_outcomes",
    "devig_oo_epc", "devig_oo_epc_batch", "evaluate_context", "mcp",
    "normalize_ah_tier", "normalize_context_bucket", "normalize_league_tier",
    "os", "poisson_dixon_coles", "regularized_incomplete_beta", "select_playbooks",
]

if __name__ == "__main__":
    sys.exit(_cli_main())

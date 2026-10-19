"""Core portfolio: rules-based holdings the trading bot sweeps its excess into.

Paper-only. Pure logic lives in settings / ledger / rebalancer / weights / tilts
so it can be backtested and unit-tested without any network or broker.
Entry point: `python -m portfolio --help`. Roadmap: PORTFOLIO_PLAN.md.
"""

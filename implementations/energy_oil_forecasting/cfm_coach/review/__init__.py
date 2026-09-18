"""The coach's review agent: reads resolved v5.2 runs, finds repeated errors, proposes fixes.

Everything the weekly job writes lives under this package's ``data_dir``
(``cfm_coach/review/`` by default). It never edits the live streams, their
ledgers, or the agent package: numeric proposals become a *coached* forecast
replayed under a coach-owned ledger, and text proposals become drafted
challenger packages a human registers by hand.

See ``~/.claude/plans/review-the-plan-for-dazzling-kitten.md`` for the design.
"""

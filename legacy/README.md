# Legacy Artifacts

This directory keeps historical one-off scripts and generated outputs from the
first benchmark iteration.

- `scripts/` contains the old ad hoc runners, report generators, and repair
  scripts. They are kept for provenance, not as the active execution surface.
- `reports/` and `results/` are generated artifacts and are ignored by git.

The active benchmark v2 entry points live in:

- `benchmark/` for model/task/run specs
- `scripts/run_benchmark.py`
- `scripts/build_leaderboard.py`
- `scripts/import_legacy_results.py`
- `uv run mm-bench benchmark ...`

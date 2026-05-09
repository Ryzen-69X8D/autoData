"""
retrain_pipeline.py
====================
Programmatic (non-Airflow) orchestrator for the full MLOps pipeline.

Execution order:
  1. Ingest    — load NIFTY 50 CSV data
  2. Validate  — OHLCV sanity checks
  3. Preprocess — feature engineering + MinMaxScaler
  4. Train     — RandomForest on 80% of data
  5. Evaluate  — RMSE / MAE / R² on held-out 20%
  6. Compare   — new vs deployed metrics
  7. Deploy    — conditional model promotion
  8. Notify    — log / Slack / email summary

Use this module instead of dags/retrain_dag.py when:
  • Running locally without Docker / Airflow
  • Writing integration tests that exercise every step end-to-end
  • Scheduling via cron instead of Airflow
  • Building CI pipelines (GitHub Actions, etc.)

Usage:
    python pipeline/retrain_pipeline.py
    python pipeline/retrain_pipeline.py --ticker TCS.NS --start 2021-01-01
    python pipeline/retrain_pipeline.py --dry-run   # skips deployment
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# Make src/ importable whether run from the project root or the pipeline/ dir.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.ingest     import fetch_stock_data
from src.preprocess import preprocess_data
from src.train      import train_model
from src.evaluate   import evaluate_model, is_new_model_better
from src.deploy     import deploy_model
from src.utils      import (
    get_logger,
    load_json,
    save_json,
    validate_ohlcv,
    metrics_summary,
    notify,
    today_str,
)

log = get_logger(__name__)


# ── Pipeline configuration ────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """All tuneable knobs for one pipeline run."""

    # Data
    ticker:     str = "NIFTY_50"
    start_date: str = "2020-01-01"
    end_date:   str = "2024-12-31"

    # Paths  (relative to project root)
    raw_path:             str = "data/raw/stock_data.csv"
    processed_path:       str = "data/processed/processed_data.csv"
    model_new_path:       str = "models/random_forest_new.pkl"
    model_prod_path:      str = "models/random_forest.pkl"
    scaler_path:          str = "models/scaler.pkl"
    metrics_new_path:     str = "models/metrics_new.json"
    metrics_deployed_path: str = "models/metrics_deployed.json"

    # Behaviour
    dry_run:          bool = False   # skip deployment step
    force_deploy:     bool = False   # deploy even if metrics don't improve
    notify_channel:   str = "log"   # "log" | "slack" | "email"
    raise_on_failure: bool = True    # re-raise exceptions after logging


# ── Step results ──────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    name:      str
    success:   bool
    duration:  float          # seconds
    output:    dict = field(default_factory=dict)
    error:     Optional[str] = None


@dataclass
class PipelineReport:
    run_id:       str
    ticker:       str
    date_range:   str
    started_at:   str
    finished_at:  str = ""
    total_seconds: float = 0.0
    steps:        list[StepResult] = field(default_factory=list)
    deployed:     Optional[bool] = None
    success:      bool = False
    summary:      str = ""

    def add(self, result: StepResult) -> None:
        self.steps.append(result)

    def failed_steps(self) -> list[str]:
        return [s.name for s in self.steps if not s.success]

    def to_dict(self) -> dict:
        return {
            "run_id":       self.run_id,
            "ticker":       self.ticker,
            "date_range":   self.date_range,
            "started_at":   self.started_at,
            "finished_at":  self.finished_at,
            "total_seconds": self.total_seconds,
            "deployed":     self.deployed,
            "success":      self.success,
            "summary":      self.summary,
            "steps": [
                {
                    "name":     s.name,
                    "success":  s.success,
                    "duration": round(s.duration, 2),
                    "output":   s.output,
                    "error":    s.error,
                }
                for s in self.steps
            ],
        }


# ── Pipeline runner ───────────────────────────────────────────────────────────

class RetrainPipeline:
    """
    Orchestrates all pipeline steps.  Each step is wrapped in a timing /
    error-catching harness so one failure does not crash the entire run
    (unless cfg.raise_on_failure is True).
    """

    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        self.report = PipelineReport(
            run_id=f"run_{today_str('%Y%m%d_%H%M%S')}",
            ticker=self.cfg.ticker,
            date_range=f"{self.cfg.start_date} → {self.cfg.end_date}",
            started_at=datetime.utcnow().isoformat(),
        )
        self._t0 = time.time()

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> PipelineReport:
        """Executes all steps in order and returns a full report."""
        self._banner("START")

        steps = [
            ("ingest",     self._step_ingest),
            ("validate",   self._step_validate),
            ("preprocess", self._step_preprocess),
            ("train",      self._step_train),
            ("evaluate",   self._step_evaluate),
            ("compare",    self._step_compare),
            ("deploy",     self._step_deploy),
        ]

        for name, fn in steps:
            result = self._run_step(name, fn)
            self.report.add(result)
            if not result.success and self.cfg.raise_on_failure:
                self._finalise(overall_success=False)
                raise RuntimeError(
                    f"Pipeline aborted at step '{name}': {result.error}"
                )

        overall = all(s.success for s in self.report.steps)
        self._finalise(overall_success=overall)
        self._banner("DONE")
        return self.report

    # ── Steps ─────────────────────────────────────────────────────────────────

    def _step_ingest(self) -> dict:
        df = fetch_stock_data(
            ticker=self.cfg.ticker,
            start_date=self.cfg.start_date,
            end_date=self.cfg.end_date,
            output_path=self.cfg.raw_path,
        )
        if df.empty:
            raise RuntimeError("Ingestion returned an empty DataFrame.")
        return {"rows": len(df), "columns": df.columns.tolist()}

    def _step_validate(self) -> dict:
        import pandas as pd
        df = pd.read_csv(self.cfg.raw_path)
        ok, errors = validate_ohlcv(df)
        if not ok:
            raise ValueError(f"OHLCV validation failed: {errors}")
        return {"valid": True, "rows_checked": len(df)}

    def _step_preprocess(self) -> dict:
        result_df = preprocess_data(
            input_path=self.cfg.raw_path,
            output_path=self.cfg.processed_path,
            scaler_path=self.cfg.scaler_path,
        )
        return {
            "rows":     len(result_df),
            "features": result_df.columns.tolist(),
        }

    def _step_train(self) -> dict:
        rmse = train_model(
            input_path=self.cfg.processed_path,
            model_output_path=self.cfg.model_new_path,
            metrics_path=self.cfg.metrics_new_path,
        )
        return {"rmse": rmse}

    def _step_evaluate(self) -> dict:
        metrics = evaluate_model(
            model_path=self.cfg.model_new_path,
            data_path=self.cfg.processed_path,
            metrics_output_path=self.cfg.metrics_new_path,
        )
        log.info("Evaluation → %s", metrics_summary(metrics))
        return metrics

    def _step_compare(self) -> dict:
        better = is_new_model_better(
            new_metrics_path=self.cfg.metrics_new_path,
            deployed_metrics_path=self.cfg.metrics_deployed_path,
        )
        return {"new_is_better": better}

    def _step_deploy(self) -> dict:
        if self.cfg.dry_run:
            log.info("DRY-RUN: skipping deployment.")
            self.report.deployed = False
            return {"deployed": False, "reason": "dry_run"}

        # Check if forced or if new model is better
        better = is_new_model_better(
            new_metrics_path=self.cfg.metrics_new_path,
            deployed_metrics_path=self.cfg.metrics_deployed_path,
        )
        if not better and not self.cfg.force_deploy:
            log.info("New model is not better — keeping current deployed model.")
            self.report.deployed = False
            return {"deployed": False, "reason": "not_better"}

        deployed = deploy_model(
            new_model_path=self.cfg.model_new_path,
            deployed_model_path=self.cfg.model_prod_path,
            new_metrics_path=self.cfg.metrics_new_path,
            deployed_metrics_path=self.cfg.metrics_deployed_path,
        )
        self.report.deployed = deployed
        return {"deployed": deployed}

    # ── Harness ───────────────────────────────────────────────────────────────

    def _run_step(self, name: str, fn) -> StepResult:
        log.info("── Step: %-12s starting…", name.upper())
        t0 = time.time()
        try:
            output  = fn()
            elapsed = time.time() - t0
            log.info("   ✅  %-12s completed in %.2fs", name.upper(), elapsed)
            return StepResult(name=name, success=True, duration=elapsed, output=output or {})
        except Exception as exc:
            elapsed = time.time() - t0
            error_text = traceback.format_exc()
            log.error("   ❌  %-12s FAILED in %.2fs\n%s", name.upper(), elapsed, error_text)
            return StepResult(
                name=name,
                success=False,
                duration=elapsed,
                error=str(exc),
            )

    def _finalise(self, overall_success: bool) -> None:
        self.report.finished_at  = datetime.utcnow().isoformat()
        self.report.total_seconds = round(time.time() - self._t0, 1)
        self.report.success      = overall_success

        failed = self.report.failed_steps()
        if overall_success:
            self.report.summary = (
                f"Pipeline completed in {self.report.total_seconds}s. "
                f"Model {'DEPLOYED ✅' if self.report.deployed else 'NOT changed ⛔'}."
            )
        else:
            self.report.summary = (
                f"Pipeline FAILED after {self.report.total_seconds}s. "
                f"Failed steps: {', '.join(failed)}."
            )

        notify(self.report.summary, channel=self.cfg.notify_channel)

        # Persist the run report alongside the model artefacts
        report_path = os.path.join(
            os.path.dirname(self.cfg.model_prod_path),
            f"pipeline_report_{self.report.run_id}.json",
        )
        try:
            save_json(self.report.to_dict(), report_path)
        except Exception:
            pass   # non-fatal

    def _banner(self, label: str) -> None:
        line = "=" * 62
        log.info(line)
        log.info("  autoData MLOps Pipeline  |  %s  |  %s", label, today_str())
        log.info("  Ticker: %-10s  %s → %s", self.cfg.ticker,
                 self.cfg.start_date, self.cfg.end_date)
        if label == "DONE":
            log.info("  Duration: %.1fs  |  Deployed: %s",
                     self.report.total_seconds, self.report.deployed)
            log.info("  Summary: %s", self.report.summary)
        log.info(line)


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full stock prediction retraining pipeline."
    )
    p.add_argument("--ticker",     default="NIFTY_50",   help="NSE ticker (default: NIFTY_50)")
    p.add_argument("--start",      default="2020-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end",        default="2024-12-31", help="End date   YYYY-MM-DD")
    p.add_argument("--dry-run",    action="store_true",  help="Skip deployment step")
    p.add_argument("--force",      action="store_true",  help="Deploy even if not better")
    p.add_argument("--notify",     default="log",        choices=["log", "slack", "email"],
                   help="Notification channel")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    cfg = PipelineConfig(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        dry_run=args.dry_run,
        force_deploy=args.force,
        notify_channel=args.notify,
    )

    pipeline = RetrainPipeline(cfg)
    report   = pipeline.run()

    sys.exit(0 if report.success else 1)

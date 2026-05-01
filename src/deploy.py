import json
import os
import shutil


def deploy_model(
    new_model_path:      str,
    deployed_model_path: str,
    new_metrics_path:    str,
    deployed_metrics_path: str = None,
) -> bool:
    """
    Conditionally promotes the newly trained model to production.

    Strategy:
      - If no model is deployed yet           → always deploy.
      - If new RMSE < deployed RMSE           → deploy.
      - Otherwise                             → keep existing model.

    Returns True if the new model was deployed, False otherwise.
    """
    should_deploy = True

    if deployed_metrics_path and os.path.exists(deployed_metrics_path):
        with open(new_metrics_path)      as f: new_m = json.load(f)
        with open(deployed_metrics_path) as f: dep_m = json.load(f)

        new_rmse = new_m.get("rmse", float("inf"))
        dep_rmse = dep_m.get("rmse", float("inf"))

        if new_rmse >= dep_rmse:
            print(f"⛔  New model (RMSE={new_rmse:.6f}) is NOT better than "
                  f"deployed (RMSE={dep_rmse:.6f}). Skipping deployment.")
            should_deploy = False
        else:
            print(f"✅  New model (RMSE={new_rmse:.6f}) beats deployed "
                  f"(RMSE={dep_rmse:.6f}). Proceeding with deployment.")

    if should_deploy:
        os.makedirs(os.path.dirname(deployed_model_path), exist_ok=True)
        shutil.copy2(new_model_path, deployed_model_path)
        print(f"✅  Model deployed → {deployed_model_path}")

        # Also promote the metrics so the next run compares correctly
        if new_metrics_path and deployed_metrics_path:
            shutil.copy2(new_metrics_path, deployed_metrics_path)
            print(f"✅  Metrics promoted → {deployed_metrics_path}")

    return should_deploy


# ── Standalone execution ──────────────────────────────────────────────────────
if __name__ == "__main__":
    deploy_model(
        new_model_path        = "/opt/airflow/models/random_forest_new.pkl",
        deployed_model_path   = "/opt/airflow/models/random_forest.pkl",
        new_metrics_path      = "/opt/airflow/models/metrics_new.json",
        deployed_metrics_path = "/opt/airflow/models/metrics_deployed.json",
    )

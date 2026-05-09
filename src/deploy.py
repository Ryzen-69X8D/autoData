import json
import os
import shutil

def deploy_ensemble(new_model_dir: str, prod_model_dir: str):
    new_metrics_path = os.path.join(new_model_dir, "metrics_new.json")
    prod_metrics_path = os.path.join(prod_model_dir, "metrics_prod.json")
    
    should_deploy = True

    if os.path.exists(prod_metrics_path):
        with open(new_metrics_path) as f: new_m = json.load(f)
        with open(prod_metrics_path) as f: prod_m = json.load(f)
        
        if new_m.get("rmse", float("inf")) >= prod_m.get("rmse", float("inf")):
            print("⛔ New ensemble is NOT better. Skipping deployment.")
            should_deploy = False
        else:
            print("✅ New ensemble beats deployed model. Proceeding.")

    if should_deploy:
        os.makedirs(prod_model_dir, exist_ok=True)
        # Copy all models
        for file in ["lstm_model.pt", "xgb_model.json", "rf_model.pkl"]:
            shutil.copy2(os.path.join(new_model_dir, file), os.path.join(prod_model_dir, file))
        # Copy scaler and metrics
        scaler_src = os.path.join(new_model_dir, "..", "scaler.pkl")
        if os.path.exists(scaler_src):
            shutil.copy2(scaler_src, os.path.join(prod_model_dir, "scaler.pkl"))
        shutil.copy2(new_metrics_path, prod_metrics_path)
        print(f"✅ Ensemble deployed -> {prod_model_dir}")

    return should_deploy

if __name__ == "__main__":
    deploy_ensemble(
        os.path.join(os.path.dirname(__file__), "..", "models", "new"),
        os.path.join(os.path.dirname(__file__), "..", "models", "prod")
    )
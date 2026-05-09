import os
import shutil
import json

def deploy_model(new_model_dir: str, deployed_model_dir: str, new_metrics_path: str, deployed_metrics_path: str) -> bool:
    # 1. Compare Scores
    if not os.path.exists(deployed_metrics_path):
        better = True
    else:
        with open(new_metrics_path) as f: new = json.load(f)
        with open(deployed_metrics_path) as f: old = json.load(f)
        better = new.get("rmse", float("inf")) < old.get("rmse", float("inf"))

    # 2. Deploy if strictly better
    if better:
        os.makedirs(deployed_model_dir, exist_ok=True)
        
        # Move the 3 ensemble files
        for file in ["lstm_model.pt", "xgb_model.json", "rf_model.pkl"]:
            src = os.path.join(new_model_dir, file)
            dst = os.path.join(deployed_model_dir, file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        # Move the scaler 
        if os.path.exists("models/scaler.pkl"):
            shutil.copy2("models/scaler.pkl", os.path.join(deployed_model_dir, "scaler.pkl"))
            
        # Move the metrics
        shutil.copy2(new_metrics_path, deployed_metrics_path)
        return True
        
    return False
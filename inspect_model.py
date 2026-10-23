"""
SynRes-AI: Model Inspector Utility.
Loads and displays the internal structure of any .pkl model in models/
(Hyperparameters, number of decision trees, feature weights, etc.)
"""

import os
import sys
import pickle

# Set UTF-8 encoding for clean terminal display
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def inspect_model(filename="real_model_oncologyscreen_drs.pkl"):
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"File {filename} not found in {MODELS_DIR}")
        return

    print("=" * 70)
    print(f"  INSPECTING TRAINED BINARY MODEL: {filename}")
    print("=" * 70)

    with open(filepath, "rb") as f:
        model = pickle.load(f)

    print(f"• Model Class Type:       {type(model).__name__} ({type(model).__module__})")
    print(f"• Number of Boosted Trees: {getattr(model, 'n_estimators', 'N/A')}")
    print(f"• Tree Max Depth:          {getattr(model, 'max_depth', 'N/A')}")
    print(f"• Learning Rate (eta):     {getattr(model, 'learning_rate', 'N/A')}")
    print(f"• Regularization (L1/L2):  reg_alpha={getattr(model, 'reg_alpha', 0)}, reg_lambda={getattr(model, 'reg_lambda', 1)}")
    
    if hasattr(model, 'n_features_in_'):
        print(f"• Input Feature Count:     {model.n_features_in_} features (Symmetric DRS + Multi-Omics)")

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_idx = importances.argsort()[::-1][:8]
        print("\n• Top 8 Most Influential Feature Weights:")
        for rank, idx in enumerate(top_idx, 1):
            print(f"   [{rank}] Feature Index {idx:<4} -> Importance Weight: {importances[idx]:.5f}")

    print("\n[SUCCESS] The .pkl model is loaded and intact in Python memory!")
    print("=" * 70)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "real_model_oncologyscreen_drs.pkl"
    inspect_model(target)

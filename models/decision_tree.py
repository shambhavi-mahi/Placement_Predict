import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import config

from models.feature_engg import _apply_encoding

def _run_tree_models(df):
    os.makedirs(config.DECISION_TREE_DIR, exist_ok=True)
    enc_df, _, _, _ = _apply_encoding(df)
    drop_cols = ["StudentID", "IsAnomaly", "Salary Package", config.TARGET_COLUMN]
    feature_cols = [c for c in enc_df.columns if c not in drop_cols]
    
    data = enc_df.dropna(subset=feature_cols + [config.TARGET_COLUMN]).copy()
    
    X = data[feature_cols]
    y = data[config.TARGET_COLUMN].astype(int)
    
    split = int(0.8 * len(data))
    x_train, x_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]
    
    # 1. Single Trees (Overfitting vs Underfitting)
    full_tree = DecisionTreeClassifier(random_state=config.RANDOM_STATE)
    full_tree.fit(x_train, y_train)
    shallow_tree = DecisionTreeClassifier(max_depth=3, random_state=config.RANDOM_STATE)
    shallow_tree.fit(x_train, y_train)
    
    # 2. Bagging
    bagging = BaggingClassifier(random_state=config.RANDOM_STATE)
    bagging.fit(x_train, y_train)
    rf = RandomForestClassifier(random_state=config.RANDOM_STATE)
    rf.fit(x_train, y_train)
    
    # 3. Boosting
    adaboost = AdaBoostClassifier(random_state=config.RANDOM_STATE)
    adaboost.fit(x_train, y_train)
    gb = GradientBoostingClassifier(random_state=config.RANDOM_STATE)
    gb.fit(x_train, y_train)
    
    def acc(model):
        return {
            "train_acc": round(float(accuracy_score(y_train, model.predict(x_train))) * 100, 2),
            "val_acc": round(float(accuracy_score(y_val, model.predict(x_val))) * 100, 2)
        }
    
    plt.figure(figsize=(20, 10))
    plot_tree(shallow_tree, feature_names=feature_cols, class_names=["Not Placed", "Placed"], filled=True, fontsize=9)
    plt.title("Decision Tree (max_depth=3)")
    plt.savefig(f"{config.DECISION_TREE_DIR}/tree_visualization.png", dpi=120, bbox_inches="tight")
    plt.close()
    
    importance = pd.Series(shallow_tree.feature_importances_, index=feature_cols)
    top_features = importance[importance > 0].sort_values(ascending=False)
    
    plt.figure(figsize=(7, max(3, len(top_features) * 0.4)))
    top_features.sort_values().plot(kind="barh")
    plt.xlabel("Importance")
    plt.title("Feature Importance (shallow tree)")
    plt.tight_layout()
    plt.savefig(f"{config.DECISION_TREE_DIR}/feature_importance.png")
    plt.close()
    
    res = {
        "full": acc(full_tree),
        "shallow": acc(shallow_tree),
        "bagging": acc(bagging),
        "rf": acc(rf),
        "ada": acc(adaboost),
        "gb": acc(gb)
    }
    res["full"]["depth"] = full_tree.get_depth()
    return res

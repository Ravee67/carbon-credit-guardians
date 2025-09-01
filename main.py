import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
DATA_PATH = "creditcard.csv"  
MODEL_ARTIFACT = "fraud_model.joblib"

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset {path} not found. Download from Kaggle and place here.")
    df = pd.read_csv(path)
    return df

def preprocess(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    if "Time" in X.columns:
        X = X.drop(columns=["Time"])

    if "Amount" in X.columns:
        rs = RobustScaler()
        X["Amount"] = rs.fit_transform(X[["Amount"]])

    return X, y

def build_search_spaces():
    lr_pipe = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.1)),
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=2000,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1
        ))
    ])
    rf_pipe = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.1)),
        ("clf", RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=-1
        ))
    ])

    search_spaces = {
        "lr": {
            "pipeline": lr_pipe,
            "param_distributions": {
                "clf__C": np.logspace(-2, 2, 10),
                "clf__penalty": ["l1", "l2"]
            },
        },
        "rf": {
            "pipeline": rf_pipe,
            "param_distributions": {
                "clf__n_estimators": [200, 300, 400],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_split": [2, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__max_features": ["sqrt", "log2", None],
            },
        },
    }
    return search_spaces

def choose_threshold_by_fbeta(y_true, y_prob, beta=2.0) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.append(thresholds, 1.0)
    eps = 1e-12
    fbeta_scores = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls + eps)
    best_idx = np.nanargmax(fbeta_scores)
    return float(thresholds[best_idx])

def train_and_eval(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    spaces = build_search_spaces()

    best_model = None
    best_score = -np.inf
    best_name = None

    for name, cfg in spaces.items():
        pipe = cfg["pipeline"]
        params = cfg["param_distributions"]

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=params,
            n_iter=5, 
            scoring="average_precision",
            cv=cv,
            n_jobs=-1,
            verbose=1,
            random_state=RANDOM_STATE,
            refit=True
        )
        search.fit(X_train, y_train)

        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model = search.best_estimator_
            best_name = name

    y_prob = best_model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    th = choose_threshold_by_fbeta(y_test, y_prob, beta=2.0)
    y_pred = (y_prob >= th).astype(int)

    print("\n✅ Selected model:", best_name)
    print("ROC-AUC :", roc)
    print("PR-AUC  :", pr_auc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return {"model": best_model, "threshold": th}

def save_artifact(artifacts, path=MODEL_ARTIFACT):
    joblib.dump(artifacts, path)
    print(f"\n💾 Model saved at {path}")

def load_artifact(path=MODEL_ARTIFACT):
    return joblib.load(path)

def predict_label(df, artifact):
    probs = artifact["model"].predict_proba(df)[:, 1]
    th = artifact["threshold"]
    return (probs >= th).astype(int), probs

def main():
    df = load_data(DATA_PATH)
    X, y = preprocess(df)
    artifacts = train_and_eval(X, y)
    save_artifact(artifacts)

    sample = X.head(5)
    preds, probs = predict_label(sample, artifacts)
    print("\n🔍 Sample predictions:\n", pd.DataFrame({"fraud_prob": probs, "fraud_label": preds}))

if __name__ == "__main__":
    main()


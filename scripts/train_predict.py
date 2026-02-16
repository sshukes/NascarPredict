import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, roc_auc_score, brier_score_loss


def choose_models():
    try:
        from catboost import CatBoostRegressor, CatBoostClassifier  # type: ignore
        print("[OK] using CatBoost")
        return (
            CatBoostRegressor(verbose=False, depth=6, iterations=200),
            CatBoostClassifier(verbose=False, depth=6, iterations=200),
            CatBoostClassifier(verbose=False, depth=6, iterations=200),
            "catboost",
        )
    except Exception:
        print("[WARN] CatBoost missing; using scikit fallback")
        return (
            RandomForestRegressor(n_estimators=200, random_state=42),
            LogisticRegression(max_iter=1000),
            LogisticRegression(max_iter=1000),
            "scikit",
        )


def prep(df):
    features = [c for c in df.columns if c.startswith("drv_") or c in ["Start", "qual_speed"]]
    X = df[features].apply(pd.to_numeric, errors="coerce")
    all_missing = [c for c in X.columns if X[c].notna().sum() == 0]
    if all_missing:
        print(f"[WARN] dropping all-missing features before imputation: {', '.join(all_missing)}")
        features = [c for c in features if c not in all_missing]
        X = X[features]
    if not features:
        raise SystemExit("[ERROR] no usable feature columns after dropping all-missing features")
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X)
    return X, features, imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="data/featurized/data_featurized.csv")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--predict_future", action="store_true")
    ap.add_argument("--compare_actual", action="store_true")
    ap.add_argument("--year", type=int)
    ap.add_argument("--race", type=int)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--save_csv", default=None)
    args = ap.parse_args()

    Path("models").mkdir(exist_ok=True)
    df = pd.read_csv(args.infile)
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df = df.sort_values(["race_date", "year", "season_race_num"])

    reg, clf_top10, clf_dnf, family = choose_models()
    train_df = df[df["target_finish"].notna()].copy()
    if train_df.empty:
        raise SystemExit("[ERROR] no training rows with actual finish")

    split = int(len(train_df) * 0.8)
    tr = train_df.iloc[:split]
    te = train_df.iloc[split:] if split < len(train_df) else train_df.iloc[-1:]
    Xtr, feats, imp = prep(tr)
    Xte = imp.transform(te[feats].apply(pd.to_numeric, errors="coerce"))

    reg.fit(Xtr, tr["target_finish"])
    clf_top10.fit(Xtr, tr["target_top10"])
    clf_dnf.fit(Xtr, tr["target_dnf"])

    pred_finish = reg.predict(Xte)
    prob_top10 = clf_top10.predict_proba(Xte)[:, 1]
    prob_dnf = clf_dnf.predict_proba(Xte)[:, 1]
    print(f"MAE={mean_absolute_error(te['target_finish'], pred_finish):.3f}")
    if len(np.unique(te["target_top10"])) > 1:
        print(f"AUC_top10={roc_auc_score(te['target_top10'], prob_top10):.3f} Brier={brier_score_loss(te['target_top10'], prob_top10):.3f}")

    if args.train:
        import joblib

        joblib.dump(reg, "models/finish_model.pkl")
        joblib.dump(clf_top10, "models/top10_model.pkl")
        joblib.dump(clf_dnf, "models/dnf_model.pkl")
        Path("models/meta.json").write_text(json.dumps({"family": family}, indent=2), encoding="utf-8")
        Path("models/feature_cols.json").write_text(json.dumps(feats, indent=2), encoding="utf-8")
        print("[OK] models saved to models/")

    if args.predict or args.predict_future:
        sub = df[(df["year"] == args.year) & (df["season_race_num"] == args.race)].copy()
        if sub.empty:
            print(f"No rows found for year={args.year} race={args.race}. Run get_entries/build_dataset first.")
            return
        Xs = imp.transform(sub[feats].apply(pd.to_numeric, errors="coerce"))
        sub["pred_finish"] = reg.predict(Xs)
        sub["prob_top10"] = clf_top10.predict_proba(Xs)[:, 1]
        sub["prob_dnf"] = clf_dnf.predict_proba(Xs)[:, 1]
        sub["score"] = sub["prob_top10"] - sub["prob_dnf"]
        print("\nBest predicted finish")
        print(sub.sort_values("pred_finish")[["Driver", "pred_finish", "prob_top10", "prob_dnf"]].head(args.top).to_string(index=False))
        print("\nBest prob_top10 - prob_dnf")
        print(sub.sort_values("score", ascending=False)[["Driver", "score", "prob_top10", "prob_dnf"]].head(args.top).to_string(index=False))
        if args.save_csv:
            sub.to_csv(args.save_csv, index=False)

    if args.compare_actual:
        sub = train_df[(train_df["year"] == args.year) & (train_df["season_race_num"] == args.race)].copy()
        if sub.empty:
            print("[WARN] no actual rows for compare")
            return
        Xs = imp.transform(sub[feats].apply(pd.to_numeric, errors="coerce"))
        sub["pred_finish"] = reg.predict(Xs)
        print(f"Actual comparison MAE={mean_absolute_error(sub['target_finish'], sub['pred_finish']):.3f}")


if __name__ == "__main__":
    main()

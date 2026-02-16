import argparse
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError


def _read_csv_if_present(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except EmptyDataError:
        print(f"[WARN] {path} is empty; skipping")
        return pd.DataFrame()




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/data.csv")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    results = _read_csv_if_present("data/raw/results.csv")
    entries = _read_csv_if_present("data/raw/entries.csv")

    base = results.copy() if not results.empty else entries.copy()
    if base.empty:
        pd.DataFrame().to_csv(out, index=False)
        print("[WARN] empty dataset")
        return

    if not entries.empty:
        combo = pd.concat([base, entries], ignore_index=True, sort=False)
        if "driver_id" not in combo.columns and "Driver" in combo.columns:
            combo["driver_id"] = combo["Driver"].astype(str).str.lower().str.replace(" ", "_", regex=False)
        dedupe_cols = [c for c in ["sked_id", "driver_id"] if c in combo.columns]
        if dedupe_cols:
            base = combo.drop_duplicates(dedupe_cols, keep="last")
        else:
            print("[WARN] missing dedupe keys after merge; keeping combined rows")
            base = combo

    if Path("data/raw/qualifying.csv").exists() and "driver_id" in base.columns:
        q = _read_csv_if_present("data/raw/qualifying.csv")
        q = q[[c for c in ["sked_id", "driver_id", "Start", "qual_speed", "pole_speed", "qual_round"] if c in q.columns]]
        base = base.drop(columns=["Start", "qual_speed", "pole_speed", "qual_round"], errors="ignore").merge(q, on=["sked_id", "driver_id"], how="left")

    for p in ["data/enrich/race_meta.csv", "data/enrich/weather.csv"]:
        if Path(p).exists() and "sked_id" in base.columns:
            e = _read_csv_if_present(p)
            if "sked_id" in e.columns:
                base = base.merge(e, on="sked_id", how="left", suffixes=("", "_enrich"))

    if Path("data/enrich/track_meta.csv").exists() and "track" in base.columns:
        t = _read_csv_if_present("data/enrich/track_meta.csv")
        if "track_canonical" in t.columns:
            base = base.merge(t, left_on="track", right_on="track_canonical", how="left", suffixes=("", "_track"))

    if "driver_id" not in base.columns and "Driver" in base.columns:
        base["driver_id"] = base["Driver"].astype(str).str.lower().str.replace(" ", "_", regex=False)

    car = base["CarNumber"].astype(str) if "CarNumber" in base.columns else ""
    base["_k2"] = base["Driver"].astype(str) + "|" + car
    key_primary = [c for c in ["sked_id", "driver_id"] if c in base.columns]
    if key_primary:
        base = base.drop_duplicates(key_primary, keep="last")
    else:
        print("[WARN] primary key columns missing; skip primary de-dupe")

    key_fallback = [c for c in ["sked_id", "_k2"] if c in base.columns]
    if key_fallback:
        base = base.drop_duplicates(key_fallback, keep="last")
    else:
        print("[WARN] fallback key columns missing; skip fallback de-dupe")
    base = base.drop(columns=["_k2"], errors="ignore")

    sort_cols = [c for c in ["race_date", "year", "season_race_num", "driver_id"] if c in base.columns]
    if sort_cols:
        base = base.sort_values(sort_cols)
    else:
        print("[WARN] sort columns missing; writing unsorted dataset")
    base.to_csv(out, index=False)
    try:
        base.to_parquet("data/raw/data.parquet", index=False)
    except Exception:
        print("[WARN] pyarrow missing; parquet skipped")
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()

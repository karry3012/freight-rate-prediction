"""Freight rate prediction: time-aware validation, LightGBM on log(rate).

Run:  python freight_rate_model.py
Outputs: validation_predictions.csv, data/december_chart_inputs.csv ,
         validation_report.csv (CV metrics)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, median_absolute_error

warnings.filterwarnings("ignore")
DATA = Path("data")
SEED = 42
CATEGORICALS = ["equipment", "pickup", "delivery", "lane"]


# ----------------------------------------------------------------- cleaning
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3959 * 2 * np.arcsin(np.sqrt(a))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # ~0.6% of weights are negative with a plausible magnitude -> sign-flip error
    df["weight_was_negative"] = (df["weight"] < 0).astype(int)
    df["weight"] = df["weight"].abs()
    df["weight_missing"] = df["weight"].isna().astype(int)
    df["market_index_missing"] = df["market_index"].isna().astype(int)
    return df


def add_features(df: pd.DataFrame, weight_med: dict, mi_by_date: pd.Series) -> pd.DataFrame:
    df = df.copy()
    # missing weight -> equipment median (from training data only)
    df["weight"] = df["weight"].fillna(df["equipment"].map(weight_med))
    # missing market_index -> mean index of that date across all loads (feature, not target)
    df["market_index"] = df["market_index"].fillna(df["date"].map(mi_by_date))
    df["market_index"] = df["market_index"].fillna(mi_by_date.mean())

    df["log_distance"] = np.log(df["distance"])
    df["inv_distance"] = 1.0 / df["distance"]  # short hauls have much higher $/mile
    df["geo_distance"] = haversine(df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])
    df["circuity"] = df["distance"] / df["geo_distance"].clip(lower=1)
    df["weight_k"] = df["weight"] / 1000
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.dayofweek
    df["doy"] = df["date"].dt.dayofyear
    df["day_of_month"] = df["date"].dt.day
    df["lat_diff"] = df["delivery_lat"] - df["pickup_lat"]
    df["lon_diff"] = df["delivery_lon"] - df["pickup_lon"]
    df["lane"] = df["pickup"] + "->" + df["delivery"]
    for c in CATEGORICALS:
        df[c] = df[c].astype("category")
    return df


FEATURES = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance", "log_distance",
    "inv_distance", "geo_distance", "circuity", "weight_k", "market_index", "quote_signal",
    "dow", "day_of_month", "lat_diff", "lon_diff", "equipment", "weight_was_negative",
    "weight_missing", "market_index_missing",
]

def prep(train_raw, other_raw_list):
    """Fit imputation stats on train only and apply to everything."""
    tr = clean(train_raw)
    weight_med = tr.groupby("equipment")["weight"].median().to_dict()
    cleaned_others = [clean(o) for o in other_raw_list]
    all_dates = pd.concat([tr] + cleaned_others)
    mi_by_date = all_dates.groupby("date")["market_index"].mean()
    return (add_features(tr, weight_med, mi_by_date),
            [add_features(o, weight_med, mi_by_date) for o in cleaned_others])



LGB_PARAMS = dict(
    objective="huber", alpha=0.35,      
    learning_rate=0.03, n_estimators=900, num_leaves=31, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
    random_state=SEED, verbose=-1,
)


def trim_outliers(df, y, z=4.0):
    """Drop training rows whose log-rate is wildly off a simple robust baseline."""
    base = Ridge(alpha=1.0).fit(pd.get_dummies(df[["log_distance", "inv_distance", "market_index", "quote_signal", "equipment"]]), y)
    res = y - base.predict(pd.get_dummies(df[["log_distance", "inv_distance", "market_index", "quote_signal", "equipment"]]))
    mad = np.median(np.abs(res - np.median(res))) * 1.4826
    return (np.abs(res - np.median(res)) < z * mad)


def fit_lgb(df, y, trim=False):
    mask = trim_outliers(df, y) if trim else np.ones(len(df), bool)
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(df.loc[mask, FEATURES], y[mask])
    return m


def fit_baseline(df, y):
    X = pd.get_dummies(df[["log_distance", "inv_distance", "market_index", "quote_signal", "equipment"]], drop_first=True)
    return Ridge(alpha=1.0).fit(X, y), X.columns


def metrics(y_true, y_pred):
    return dict(
        MAE=mean_absolute_error(y_true, y_pred),
        MedAE=median_absolute_error(y_true, y_pred),
        MAPE=mean_absolute_percentage_error(y_true, y_pred) * 100,
        WAPE=np.abs(y_true - y_pred).sum() / y_true.sum() * 100,
    )


# ----------------------------------------------------------------- validation
def time_series_cv(train_raw):
    """Expanding-window, forward-chaining CV. Each fold trains on all earlier months and tests
    on the next month, mimicking the real task (train Jan-Oct -> predict Nov-Dec)."""
    raw = train_raw.copy()
    raw["m"] = pd.to_datetime(raw["date"]).dt.month
    rows = []
    for test_month in [6, 7, 8, 9, 10]:
        tr_raw, te_raw = raw[raw.m < test_month], raw[raw.m == test_month]
        tr, (te,) = prep(tr_raw, [te_raw])
        y_tr, y_te = np.log(tr["posted_rate"].values), te["posted_rate"].values

        base, cols = fit_baseline(tr, y_tr)
        Xte = pd.get_dummies(te[["log_distance", "inv_distance", "market_index", "quote_signal", "equipment"]], drop_first=True).reindex(columns=cols, fill_value=0)
        p_base = np.exp(base.predict(Xte))
        p_raw = np.exp(fit_lgb(tr, y_tr, trim=False).predict(te[FEATURES]))
        p_trim = np.exp(fit_lgb(tr, y_tr, trim=True).predict(te[FEATURES]))
        for name, p in [("ridge_baseline", p_base), ("lgbm_huber", p_raw), ("lgbm_huber_trimmed", p_trim)]:
            rows.append(dict(test_month=test_month, model=name, n=len(te), **metrics(y_te, p)))
    rep = pd.DataFrame(rows)
    print("\nForward-chaining CV (per fold):\n", rep.round(2).to_string(index=False))
    summ = rep.groupby("model")[["MAE", "MedAE", "MAPE", "WAPE"]].mean().round(2)
    print("\nMean over folds:\n", summ.to_string())
    rep.to_csv("validation_report.csv", index=False)
    return summ


# ----------------------------------------------------------------- main
def main():
    train_raw = pd.read_csv(DATA / "train_test.csv")
    val_raw = pd.read_csv(DATA / "validation.csv")
    dec_raw = pd.read_csv(DATA / "december_chart_inputs.csv")
    template = pd.read_csv(DATA / "validation_predictions_template.csv")

    print(f"train rows={len(train_raw)}, validation rows={len(val_raw)}")
    print("missing (train):", train_raw.isna().sum()[lambda s: s > 0].to_dict())
    print("negative weights (train):", int((train_raw.weight < 0).sum()))
    time_series_cv(train_raw)

    
    raw = train_raw.copy(); raw["m"] = pd.to_datetime(raw["date"]).dt.month
    tr, (te,) = prep(raw[raw.m <= 8], [raw[raw.m >= 9]])
    p = np.exp(fit_lgb(tr, np.log(tr.posted_rate.values)).predict(te[FEATURES]))
    print("\n2-month-ahead holdout (train Jan-Aug, test Sep-Oct):", {k: round(v, 2) for k, v in metrics(te.posted_rate.values, p).items()})


    sig = val_raw.assign(date=pd.to_datetime(val_raw["date"])).groupby("date")[["market_index", "quote_signal"]].mean()
    coords = pd.concat([
        train_raw[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        train_raw[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        val_raw[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        val_raw[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
    ]).drop_duplicates("city").set_index("city")
    dec = dec_raw.copy()
    dec["date"] = pd.to_datetime(dec["date"])
    dec["market_index"] = dec["date"].map(sig["market_index"])
    dec["quote_signal"] = dec["date"].map(sig["quote_signal"])
    for side in ["pickup", "delivery"]:
        dec[f"{side}_lat"] = dec[side].map(coords["lat"])
        dec[f"{side}_lon"] = dec[side].map(coords["lon"])
    assert dec[["market_index", "quote_signal", "pickup_lat", "delivery_lat"]].notna().all().all()

    tr, (va, de) = prep(train_raw, [val_raw, dec])
    y = np.log(tr["posted_rate"].values)
    preds_val, preds_dec = [], []
    for s in range(5):
        LGB_PARAMS["random_state"] = SEED + s
        m = fit_lgb(tr, y)
        preds_val.append(m.predict(va[FEATURES])); preds_dec.append(m.predict(de[FEATURES]))
    va_pred = np.exp(np.mean(preds_val, axis=0)); de_pred = np.exp(np.mean(preds_dec, axis=0))

    out = pd.DataFrame({"load_id": val_raw["load_id"], "predicted_rate": va_pred.round(2)})
    out = template[["load_id"]].merge(out, on="load_id", how="left")
    assert out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all() and len(out) == 12000
    out.to_csv("validation_predictions.csv", index=False)

    dec_out = dec_raw.copy()
    dec_out["predicted_rate"] = de_pred.round(2)
    dec_out.to_csv(DATA / "december_chart_inputs.csv", index=False)
    print("\nSaved validation_predictions.csv and filled data/december_chart_inputs.csv")
    print(out.predicted_rate.describe().round(2).to_dict())


if __name__ == "__main__":
    main()

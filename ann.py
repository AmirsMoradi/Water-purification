import warnings
warnings.filterwarnings("ignore")

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
N_SPLITS = 5

OUT1 = Path("outputs_ann_stage1")
OUT2 = Path("outputs_ann_stage2")
for p in [OUT1, OUT2]:
    p.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.figsize"] = (9, 6)
plt.rcParams["font.size"] = 11

def ensure_cols(df: pd.DataFrame, cols: list) -> list:
    return [c for c in cols if c in df.columns]

def clean_df(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    d = df.copy()
    d[cols] = d[cols].replace([np.inf, -np.inf], np.nan)
    d[cols] = d[cols].fillna(method="ffill").fillna(method="bfill")
    return d

def safe_log_transform(y: np.ndarray, name: str):
    nm = name.lower()
    if any(k in nm for k in ["bod","cod","tss"]) and np.all(y >= 0):
        return np.log1p(y), True
    return y, False

def safe_inverse_transform(y: np.ndarray, was_logged: bool):
    return np.expm1(y) if was_logged else y

def unit_of(param: str) -> str:
    p = param.lower()
    if "ph" in p: return "-"
    if "temp" in p: return "°C"
    if "flow" in p: return "m³/d"
    if "rain" in p: return "mm"
    return "mg/L"

def stats_table(df: pd.DataFrame, sample_label: str) -> pd.DataFrame:
    s = pd.DataFrame({
        "کمینه": df.min(),
        "بیشینه": df.max(),
        "میانگین": df.mean(),
        "میانه": df.median(),
        "انحراف معیار": df.std(),
    })
    s["ضریب تغییرات (%)"] = (s["انحراف معیار"] / s["میانگین"]).replace([np.inf, -np.inf], np.nan) * 100
    s = s.round(3).reset_index().rename(columns={"index":"پارامتر"})
    s["واحد"] = s["پارامتر"].apply(unit_of)
    s.insert(0, "نمونه", sample_label)
    return s[["نمونه","پارامتر","کمینه","بیشینه","میانگین","میانه","انحراف معیار","ضریب تغییرات (%)","واحد"]]

def plot_parity(y_true, y_pred, tag, outdir):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6)
    lo, hi = np.min([y_true.min(), y_pred.min()]), np.max([y_true.max(), y_pred.max()])
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(f"Parity - {tag}")
    plt.grid(True); plt.tight_layout()
    plt.savefig(outdir / f"{tag}_parity.png", dpi=140); plt.close()

def plot_residuals(y_true, y_pred, tag, outdir):
    resid = y_true - y_pred
    plt.figure()
    plt.hist(resid, bins=30)
    plt.title(f"Residuals - {tag}")
    plt.xlabel("Residual"); plt.ylabel("Frequency")
    plt.grid(True); plt.tight_layout()
    plt.savefig(outdir / f"{tag}_resid_hist.png", dpi=140); plt.close()

def plot_timeseries(y_true, y_pred, tag, outdir, n=300):
    n = min(n, len(y_true))
    idx = np.arange(n)
    plt.figure()
    plt.plot(idx, y_true[:n], label="Actual")
    plt.plot(idx, y_pred[:n], label="Predicted")
    plt.legend(); plt.grid(True); plt.title(f"Time Series - {tag}")
    plt.tight_layout()
    plt.savefig(outdir / f"{tag}_timeseries.png", dpi=140); plt.close()

def plot_feature_importance(model, X, y, feature_names, tag, outdir):
    result = permutation_importance(model, X, y, n_repeats=20, random_state=RANDOM_STATE)
    sorted_idx = result.importances_mean.argsort()[::-1]
    plt.figure(figsize=(8,5))
    plt.bar(range(len(sorted_idx)), result.importances_mean[sorted_idx], align="center")
    plt.xticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx], rotation=45, ha="right")
    plt.title(f"Permutation Importance - {tag}")
    plt.tight_layout()
    plt.savefig(outdir / f"{tag}_feature_importance.png", dpi=140)
    plt.close()

def ann_cv_fit(X, y, target_name, hidden=(64,32)):
    y_t, was_logged = safe_log_transform(y, target_name)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    r2s, rmses, maes = [], [], []
    oof_preds = np.zeros_like(y_t)

    for tr, va in kf.split(X):
        sx = StandardScaler()
        Xtr, Xva = sx.fit_transform(X[tr]), sx.transform(X[va])
        ytr, yva = y_t[tr], y_t[va]

        model = MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                             solver="adam", random_state=RANDOM_STATE, max_iter=1000)
        model.fit(Xtr, ytr)
        yhat = model.predict(Xva)

        yhat = safe_inverse_transform(yhat, was_logged)
        yva_o = safe_inverse_transform(yva, was_logged)

        oof_preds[va] = yhat
        r2s.append(r2_score(yva_o, yhat))
        rmses.append(np.sqrt(mean_squared_error(yva_o, yhat)))
        maes.append(mean_absolute_error(yva_o, yhat))

    sx_full = StandardScaler()
    Xs = sx_full.fit_transform(X)
    model_full = MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                              solver="adam", random_state=RANDOM_STATE, max_iter=1000)
    model_full.fit(Xs, y_t)

    yhat_full = safe_inverse_transform(model_full.predict(Xs), was_logged)
    ytrue_full = safe_inverse_transform(y_t, was_logged)

    return model_full, sx_full, was_logged, oof_preds, (
        np.mean(r2s), np.mean(rmses), np.mean(maes),
        r2_score(ytrue_full, yhat_full),
        math.sqrt(mean_squared_error(ytrue_full, yhat_full)),
        mean_absolute_error(ytrue_full, yhat_full)
    )

def main():
    hist = pd.read_csv("cleaned_historical.csv")
    fut  = pd.read_csv("cleaned_future.csv")

    climate_cols = ensure_cols(hist, ["temp_air","rainfall","month","year"])
    in_cols  = ensure_cols(hist, ["temp_wastewater_in","bod_in","cod_in","tss_in","ph_in","flowrate_in"])
    out_cols = ensure_cols(hist, ["bod_out","cod_out","tss_out","ph_out","temp_effluent_out"])

    hist = clean_df(hist, climate_cols + in_cols + out_cols)

    X1 = hist[climate_cols].to_numpy(float)
    oof_dict = {}
    models_in = {}
    metrics_stage1 = []

    for target in in_cols:
        y = hist[target].to_numpy(float)
        model, sx, was_logged, oof_pred, (r2_cv, rmse_cv, mae_cv, r2_full, rmse_full, mae_full) = ann_cv_fit(X1, y, target)

        oof_dict[target] = oof_pred
        models_in[target] = (model, sx, was_logged)
        metrics_stage1.append([target, r2_cv, rmse_cv, mae_cv, r2_full, rmse_full, mae_full])

        tag = f"Stage1_{target}"
        plot_parity(y, oof_pred, tag, OUT1)
        plot_residuals(y, oof_pred, tag, OUT1)
        plot_timeseries(y, oof_pred, tag, OUT1)
        plot_feature_importance(model, sx.transform(X1), y, climate_cols, tag, OUT1)

        print(f"[Stage1] {target} | R2_CV={r2_cv:.4f} | Full R2={r2_full:.4f}")

    pd.DataFrame(metrics_stage1, columns=["target","R2_CV","RMSE_CV","MAE_CV","R2_Full","RMSE_Full","MAE_Full"])\
      .to_csv(OUT1/"Stage1_CV_Metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(oof_dict).to_csv(OUT1/"Stage1_OOF_Preds.csv", index=False, encoding="utf-8-sig")

    X2_train = pd.concat([
        hist[climate_cols+in_cols].reset_index(drop=True),
        pd.DataFrame(oof_dict).reset_index(drop=True).add_prefix("pred_")
    ], axis=1)
    X2 = X2_train.to_numpy(float)

    models_out = {}
    metrics_stage2 = []
    for target in out_cols:
        y = hist[target].to_numpy(float)
        model, sx, was_logged, oof_pred, (r2_cv, rmse_cv, mae_cv, r2_full, rmse_full, mae_full) = ann_cv_fit(X2, y, target)

        models_out[target] = (model, sx, was_logged)
        metrics_stage2.append([target, r2_cv, rmse_cv, mae_cv, r2_full, rmse_full, mae_full])

        tag = f"Stage2_{target}"
        plot_parity(y, oof_pred, tag, OUT2)
        plot_residuals(y, oof_pred, tag, OUT2)
        plot_timeseries(y, oof_pred, tag, OUT2)
        plot_feature_importance(model, sx.transform(X2), y, list(X2_train.columns), tag, OUT2)

        print(f"[Stage2] {target} | R2_CV={r2_cv:.4f} | Full R2={r2_full:.4f}")

    pd.DataFrame(metrics_stage2, columns=["target","R2_CV","RMSE_CV","MAE_CV","R2_Full","RMSE_Full","MAE_Full"])\
      .to_csv(OUT2/"Stage2_CV_Metrics.csv", index=False, encoding="utf-8-sig")

    # Final Stats Historical
    stats_in_hist  = stats_table(hist[in_cols],  "نمونه: ورودی (تاریخی)")
    stats_out_hist = stats_table(hist[out_cols], "نمونه: خروجی (تاریخی)")
    stats_cli_hist = stats_table(hist[climate_cols], "نمونه: اقلیم (تاریخی)")
    pd.concat([stats_in_hist, stats_out_hist, stats_cli_hist], ignore_index=True)\
      .to_csv("Final_Stats_Historical_ANN.csv", index=False, encoding="utf-8-sig")

    scenarios = {
        "ssp126": ("temp_air_mean", "rainfall"),
        "ssp245": ("temp_air_mean_1", "rainfall_1"),
        "ssp585": ("temp_air_mean_2", "rainfall_2"),
    }

    for scen, (tcol, rcol) in scenarios.items():
        if tcol not in fut.columns or rcol not in fut.columns:
            print(f"[warn] scenario {scen} skipped (missing {tcol}/{rcol})")
            continue

        fut_df = fut[[tcol, rcol]].rename(columns={tcol: "temp_air", rcol: "rainfall"})
        fut_df["month"] = fut["month"] if "month" in fut.columns else np.nan
        fut_df["year"] = fut["year"] if "year" in fut.columns else np.nan
        fut_df = fut_df[["temp_air", "rainfall", "month", "year"]]

        fut_df = fut_df.fillna(method="ffill").fillna(method="bfill")
        fut_df = fut_df.fillna(0)

        Xs1 = fut_df.to_numpy(float)

        pred_in = {}
        for name, (m, sx, wl) in models_in.items():
            yh = m.predict(sx.transform(Xs1))
            pred_in[name] = safe_inverse_transform(yh, wl)
        df_in_pred = pd.DataFrame(pred_in)

        Xs2_df = pd.concat([
            fut_df.reset_index(drop=True),
            df_in_pred.reset_index(drop=True),
            df_in_pred.add_prefix("pred_").reset_index(drop=True)
        ], axis=1)

        Xs2_df = Xs2_df.fillna(method="ffill").fillna(method="bfill").fillna(0)

        Xs2 = Xs2_df.to_numpy(float)

        pred_out = {}
        for name, (m, sx, wl) in models_out.items():
            yh = m.predict(sx.transform(Xs2))
            pred_out[name] = safe_inverse_transform(yh, wl)

        pd.DataFrame(pred_in).to_csv(OUT1 / f"predicted_IN_{scen}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(pred_out).to_csv(OUT2 / f"predicted_OUT_{scen}.csv", index=False, encoding="utf-8-sig")

        stats_cli_future = stats_table(fut_df[["temp_air", "rainfall"]], f"نمونه: اقلیم ({scen})")
        stats_out_future = stats_table(pd.DataFrame(pred_out), f"نمونه: خروجی ({scen})")
        pd.concat([stats_cli_future, stats_out_future], ignore_index=True) \
            .to_csv(OUT2 / f"Final_Stats_{scen}_ANN.csv", index=False, encoding="utf-8-sig")

    print("\n[done] ANN outputs (historical + future scenarios) saved!")

if __name__ == "__main__":
    main()

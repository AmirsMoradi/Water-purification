import warnings
warnings.filterwarnings("ignore")

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

RANDOM_STATE = 42
N_SPLITS = 5

OUT1 = Path("outputs_stage1")
OUT2 = Path("outputs_stage2")
for p in [OUT1, OUT2]:
    p.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.figsize"] = (9, 6)
plt.rcParams["font.size"] = 11

def ensure_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    ok = [c for c in cols if c in df.columns]
    miss = [c for c in cols if c not in df.columns]
    if miss: print(f"[warn] missing columns skipped: {miss}")
    return ok

def clean_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
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

def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, tag: str, outdir: Path):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6)
    lo, hi = np.min([y_true.min(), y_pred.min()]), np.max([y_true.max(), y_pred.max()])
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(f"Parity - {tag}")
    plt.grid(True); plt.tight_layout()
    plt.savefig(outdir / f"{tag}_parity.png", dpi=140); plt.close()

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, tag: str, outdir: Path):
    resid = y_true - y_pred
    plt.figure()
    plt.hist(resid, bins=30)
    plt.title(f"Residuals - {tag}")
    plt.xlabel("Residual"); plt.ylabel("Frequency")
    plt.grid(True); plt.tight_layout()
    plt.savefig(outdir / f"{tag}_resid_hist.png", dpi=140); plt.close()

def plot_timeseries(y_true: np.ndarray, y_pred: np.ndarray, tag: str, outdir: Path, n: int = 300):
    n = min(n, len(y_true))
    idx = np.arange(n)
    plt.figure()
    plt.plot(idx, y_true[:n], label="Actual")
    plt.plot(idx, y_pred[:n], label="Predicted")
    plt.legend(); plt.grid(True); plt.title(f"Time Series - {tag}")
    plt.tight_layout()
    plt.savefig(outdir / f"{tag}_timeseries.png", dpi=140); plt.close()

# تابع جدید برای پارامترهای اقلیمی
def plot_climate_timeseries(df: pd.DataFrame, cols: List[str], outdir: Path, n: int = 300):
    n = min(n, len(df))
    idx = np.arange(n)
    for c in cols:
        plt.figure()
        plt.plot(idx, df[c].values[:n], label=c, color="tab:blue")
        plt.legend()
        plt.grid(True)
        plt.title(f"Climate Time Series - {c}")
        plt.tight_layout()
        plt.savefig(outdir / f"Stage1_climate_{c}_timeseries.png", dpi=140)
        plt.close()

def xgb_cv_fit(X: np.ndarray, y: np.ndarray, target_name: str, param_grid: List[Dict]):
    y_t, was_logged = safe_log_transform(y, target_name)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    best_r2, best_params = -np.inf, None
    rows = []
    for params in param_grid:
        r2s, rmses, maes = [], [], []
        for tr, va in kf.split(X):
            sx = StandardScaler()
            Xtr, Xva = sx.fit_transform(X[tr]), sx.transform(X[va])
            ytr, yva = y_t[tr], y_t[va]
            model = XGBRegressor(
                n_estimators=params.get("n_estimators", 1000),
                learning_rate=params.get("learning_rate", 0.03),
                max_depth=params.get("max_depth", 8),
                subsample=params.get("subsample", 0.9),
                colsample_bytree=params.get("colsample_bytree", 0.9),
                reg_lambda=params.get("reg_lambda", 1.0),
                reg_alpha=params.get("reg_alpha", 0.0),
                n_jobs=-1,
                tree_method="hist",
                random_state=RANDOM_STATE
            )
            model.fit(Xtr, ytr)
            yhat = model.predict(Xva)
            yhat = safe_inverse_transform(yhat, was_logged)
            yva_o = safe_inverse_transform(yva, was_logged)
            r2s.append(r2_score(yva_o, yhat))
            rmses.append(np.sqrt(mean_squared_error(yva_o, yhat)))
            maes.append(mean_absolute_error(yva_o, yhat))
        row = {
            "target": target_name,
            "params": str(params),
            "R2_CV_mean": float(np.mean(r2s)),
            "R2_CV_std": float(np.std(r2s)),
            "RMSE_CV_mean": float(np.mean(rmses)),
            "MAE_CV_mean": float(np.mean(maes)),
        }
        rows.append(row)
        if row["R2_CV_mean"] > best_r2:
            best_r2, best_params = row["R2_CV_mean"], params
    sx_full = StandardScaler()
    Xs = sx_full.fit_transform(X)
    model_full = XGBRegressor(
        n_estimators=best_params.get("n_estimators", 1000),
        learning_rate=best_params.get("learning_rate", 0.03),
        max_depth=best_params.get("max_depth", 8),
        subsample=best_params.get("subsample", 0.9),
        colsample_bytree=best_params.get("colsample_bytree", 0.9),
        reg_lambda=best_params.get("reg_lambda", 1.0),
        reg_alpha=best_params.get("reg_alpha", 0.0),
        n_jobs=-1,
        tree_method="hist",
        random_state=RANDOM_STATE
    )
    model_full.fit(Xs, y_t)
    yhat_full = safe_inverse_transform(model_full.predict(Xs), was_logged)
    ytrue_full = safe_inverse_transform(y_t, was_logged)
    r2_full = r2_score(ytrue_full, yhat_full)
    rmse_full = math.sqrt(mean_squared_error(ytrue_full, yhat_full))
    mae_full = mean_absolute_error(ytrue_full, yhat_full)
    return model_full, sx_full, was_logged, rows, best_r2, (ytrue_full, yhat_full, r2_full, rmse_full, mae_full)

def stage1_oof_predictions(X_clim: np.ndarray, y: np.ndarray, target_name: str, params: Dict):
    y_t, was_logged = safe_log_transform(y, target_name)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros_like(y_t, dtype=float)
    for tr, va in kf.split(X_clim):
        sx = StandardScaler()
        Xtr, Xva = sx.fit_transform(X_clim[tr]), sx.transform(X_clim[va])
        ytr = y_t[tr]
        model = XGBRegressor(
            n_estimators=params.get("n_estimators", 1000),
            learning_rate=params.get("learning_rate", 0.03),
            max_depth=params.get("max_depth", 8),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample_bytree", 0.9),
            reg_lambda=params.get("reg_lambda", 1.0),
            reg_alpha=params.get("reg_alpha", 0.0),
            n_jobs=-1,
            tree_method="hist",
            random_state=RANDOM_STATE
        )
        model.fit(Xtr, ytr)
        oof[va] = model.predict(Xva)
    return safe_inverse_transform(oof, was_logged)

def main():
    hist = pd.read_csv("cleaned_historical.csv")
    fut  = pd.read_csv("cleaned_future.csv")

    climate_cols = ensure_cols(hist, ["temp_air","rainfall","month","year"])
    in_cols  = ensure_cols(hist, ["temp_wastewater_in","bod_in","cod_in","tss_in","ph_in","flowrate_in"])
    out_cols = ensure_cols(hist, ["bod_out","cod_out","tss_out","ph_out","temp_effluent_out"])

    hist = clean_df(hist, climate_cols + in_cols + out_cols)

    param_grid = [
        {"n_estimators":800, "learning_rate":0.05, "max_depth":6, "subsample":0.9, "colsample_bytree":0.9},
        {"n_estimators":1000,"learning_rate":0.03, "max_depth":8, "subsample":0.9, "colsample_bytree":0.9},
        {"n_estimators":1200,"learning_rate":0.02, "max_depth":10,"subsample":0.85,"colsample_bytree":0.85},
    ]

    X1 = hist[climate_cols].to_numpy(dtype=float)

    cv_rows_stage1 = []
    models_in: Dict[str, Tuple] = {}
    oof_dict: Dict[str, np.ndarray] = {}

    for target in in_cols:
        y = hist[target].to_numpy(dtype=float)
        model, sx, was_logged, rows, best_r2, full = xgb_cv_fit(X1, y, target, param_grid)
        cv_rows_stage1.extend(rows)
        models_in[target] = (model, sx, was_logged)
        best_params = eval([r for r in rows if r["R2_CV_mean"]==max([x["R2_CV_mean"] for x in rows])][0]["params"])
        oof_pred = stage1_oof_predictions(X1, y, target, best_params)
        oof_dict[target] = oof_pred
        y_true, y_hat, r2_full, rmse_full, mae_full = full
        tag = f"Stage1_{target}"
        plot_parity(y_true, y_hat, tag, OUT1)
        plot_residuals(y_true, y_hat, tag, OUT1)
        plot_timeseries(y_true, y_hat, tag, OUT1)
        print(f"[Stage1] {target} | R2_CV={best_r2:.4f} | Full R2={r2_full:.4f} | RMSE={rmse_full:.4f} | MAE={mae_full:.4f}")

    # پلات پارامترهای اقلیمی
    plot_climate_timeseries(hist, climate_cols, OUT1)

    pd.DataFrame(cv_rows_stage1).to_csv(OUT1/"Stage1_CV_Metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(oof_dict).to_csv(OUT1/"Stage1_OOF_Preds.csv", index=False, encoding="utf-8-sig")

    X2_train = pd.concat([
        hist[climate_cols+in_cols].reset_index(drop=True),
        pd.DataFrame(oof_dict).reset_index(drop=True).add_prefix("pred_")
    ], axis=1)
    X2 = X2_train.to_numpy(dtype=float)

    cv_rows_stage2 = []
    models_out: Dict[str, Tuple] = {}

    for target in out_cols:
        y = hist[target].to_numpy(dtype=float)
        model, sx, was_logged, rows, best_r2, full = xgb_cv_fit(X2, y, target, param_grid)
        cv_rows_stage2.extend(rows)
        models_out[target] = (model, sx, was_logged)
        y_true, y_hat, r2_full, rmse_full, mae_full = full
        tag = f"Stage2_{target}"
        plot_parity(y_true, y_hat, tag, OUT2)
        plot_residuals(y_true, y_hat, tag, OUT2)
        plot_timeseries(y_true, y_hat, tag, OUT2)
        print(f"[Stage2] {target} | R2_CV={best_r2:.4f} | Full R2={r2_full:.4f} | RMSE={rmse_full:.4f} | MAE={mae_full:.4f}")

    pd.DataFrame(cv_rows_stage2).to_csv(OUT2/"Stage2_CV_Metrics.csv", index=False, encoding="utf-8-sig")

    stats_in_hist  = stats_table(hist[in_cols],  "نمونه: ورودی (تاریخی)")
    stats_out_hist = stats_table(hist[out_cols], "نمونه: خروجی (تاریخی)")
    stats_cli_hist = stats_table(hist[climate_cols], "نمونه: اقلیم (تاریخی)")
    pd.concat([stats_in_hist, stats_out_hist, stats_cli_hist], ignore_index=True)\
      .to_csv(OUT1/"Final_Stats_Historical.csv", index=False, encoding="utf-8-sig")

    # سناریوهای آینده بدون تغییر

if __name__ == "__main__":
    main()

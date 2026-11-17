import pandas as pd
import numpy as np
import re

TRAIN_END = pd.Timestamp("2024-12-01")
TEST_START = pd.Timestamp("2025-01-01")
SEGMENTS = ["Smooth", "Erratic", "Intermittent"]


def load_data():
    """The modeling panel: lanes x 60 months (Lumpy + Sporadic lanes already removed)."""
    return pd.read_pickle("modeling_panel.pkl")


def load_segments():
    """Segment label (Smooth / Erratic / Intermittent) for each lane in the modeling panel."""
    return pd.read_pickle("lane_segments.pkl").loc[load_data().index]


def to_quarters(monthly):
    """Sum a lanes x months table into lanes x quarters."""
    q = monthly.copy()
    q.columns = pd.PeriodIndex(q.columns, freq="M").asfreq("Q")
    return q.T.groupby(level=0).sum().T


def score_quarterly(pred_monthly, panel, name="model"):
    """pred_monthly: lanes x 12 table of predicted TEUs for the 2025 months."""
    actual_m = panel.loc[:, TEST_START:]
    pred_m = pred_monthly.reindex(index=actual_m.index, columns=actual_m.columns)
    actual_q, pred_q = to_quarters(actual_m), to_quarters(pred_m)
    err = (pred_q - actual_q).abs()
    return {
        "model": name,
        "quarterly_MAE": err.values.mean(),
        "quarterly_WAPE_%": err.values.sum() / actual_q.values.sum() * 100,
        "volume_bias_%": (pred_q.values.sum() / actual_q.values.sum() - 1) * 100,
    }


def naive_last_month(panel):
    """Baseline 1: next month = this month."""
    return panel.shift(1, axis=1).loc[:, TEST_START:]


def naive_seasonal(panel):
    """Baseline 2: next month = same month last year."""
    return panel.shift(12, axis=1).loc[:, TEST_START:]


def record(name, pred_monthly, panel=None, seg=None):
    """Score a model overall AND by lane group, save its predictions, and add it to
    results_all.csv (replacing any earlier row with the same name)."""
    panel = load_data() if panel is None else panel
    seg = load_segments() if seg is None else seg
    row = score_quarterly(pred_monthly, panel, name)
    for s in SEGMENTS:
        lanes = seg.index[seg == s]
        row[f"WAPE_{s}"] = score_quarterly(pred_monthly.loc[lanes], panel.loc[lanes], name)["quarterly_WAPE_%"]
    pred_monthly.to_pickle("pred_" + re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") + ".pkl")
    try:
        res = pd.read_csv("results_all.csv", index_col="model")
    except FileNotFoundError:
        res = pd.DataFrame()
    res.loc[name, list(row.keys())[1:]] = list(row.values())[1:]
    res.index.name = "model"
    res.to_csv("results_all.csv")
    return res


from sklearn.metrics import roc_auc_score, accuracy_score

COLORS = {"Naive: last month": "#52514e", "Naive: same month last year": "#b9b8b2",
          "SARIMA": "#eb6834", "Linear regression": "#1baf7a", "Random Forest": "#eda100",
          "XGBoost": "#2a78d6", "LSTM": "#e87ba4", "N-HiTS": "#4a3aa7"}


def build_features(panel=None, seg=None):
    """One row per lane-month with only PAST information (nothing from the month itself):
    lag1/2/3/12, rolling means of the previous 3/6/12 months, the lane's 2021-24 average
    TEUs (lane_mean), the lane group, and the calendar month. y = this month's TEUs;
    grew = 1 if y > lag1 (only meaningful when lag1 > 0)."""
    panel = load_data() if panel is None else panel
    seg = load_segments() if seg is None else seg
    tr_cols = panel.columns[panel.columns <= TRAIN_END]
    long = panel.reset_index().melt(id_vars=["us_port", "hs4"], var_name="month", value_name="y")
    long["month"] = pd.to_datetime(long["month"])
    long = long.sort_values(["us_port", "hs4", "month"]).reset_index(drop=True)
    g = long.groupby(["us_port", "hs4"], sort=False)["y"]
    for k in (1, 2, 3, 12):
        long[f"lag{k}"] = g.shift(k)
    for k in (3, 6, 12):
        long[f"roll{k}"] = g.transform(lambda s, k=k: s.shift(1).rolling(k).mean())
    lane_mean = panel[tr_cols].mean(axis=1).rename("lane_mean").reset_index()
    long = long.merge(lane_mean, on=["us_port", "hs4"])
    long["segment"] = pd.MultiIndex.from_frame(long[["us_port", "hs4"]]).map(seg)
    long["month_num"] = long["month"].dt.month
    long["grew"] = (long["y"] > long["lag1"]).astype(int)
    return long


def split(long, needed):
    """Time split: train = through Dec 2024, test = 2025. Drops rows missing any column in `needed`."""
    d = long.dropna(subset=needed)
    return d[d["month"] <= TRAIN_END].copy(), d[d["month"] >= TEST_START].copy()


def to_panel(test_rows, pred):
    """Turn predictions for test rows back into a lanes x 12-months table (negatives clipped to 0)."""
    d = test_rows.assign(p=np.clip(np.asarray(pred, dtype=float), 0, None))
    return d.pivot(index=["us_port", "hs4"], columns="month", values="p")


def record_clf(name, test_rows, proba):
    """Score a growth classifier on 2025 lane-months where the lane shipped last month
    (lag1 > 0): accuracy vs always-guess-the-majority, and AUC. Overall + by lane group."""
    d = test_rows[test_rows["lag1"] > 0].copy()
    d["p"] = pd.Series(np.asarray(proba), index=test_rows.index)[d.index]
    rows = {}
    for label, sub in [("All", d)] + [(s, d[d["segment"] == s]) for s in SEGMENTS]:
        base = max(sub["grew"].mean(), 1 - sub["grew"].mean())
        rows[label] = {"accuracy": accuracy_score(sub["grew"], sub["p"] > 0.5),
                       "majority_baseline": base, "AUC": roc_auc_score(sub["grew"], sub["p"]),
                       "n_rows": len(sub)}
    out = pd.DataFrame(rows).T
    out.insert(0, "model", name)
    out.index.name = "group"
    out = out.reset_index()
    try:
        old = pd.read_csv("results_classification.csv")
        out_all = pd.concat([old[old["model"] != name], out])
    except FileNotFoundError:
        out_all = out
    out_all.to_csv("results_classification.csv", index=False)
    return out.set_index("group").drop(columns="model")

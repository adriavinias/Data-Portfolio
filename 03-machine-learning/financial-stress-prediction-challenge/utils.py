import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
TARGET = "liquidity_stress_next_30d"
from IPython.display import HTML, display


def getFeatures(df:pd.DataFrame, ex: list[str]):
    return [c for c in df.columns.tolist() if c not in ex]

def smallEDA(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    num_of_categorical_features = len(df.select_dtypes(include=["category","str","string"]).columns.tolist())
    num_of_numerical_features = len(df.select_dtypes(include=["number"]).columns.tolist())
    # 1. Se muestra una sola vez en la celda
    display(
        HTML(
            f"<b>Dataset Shape:</b> {df.shape} | "
            f"<b>Total Size:</b> {df.size:,} | "
            f"<b>Total missing:</b> {df.isnull().sum().sum()} | "
            f"<b>Total duplicated:</b> {df.duplicated().sum()} | "
            f"<b>Num of categorical features [str, category]:</b> {num_of_categorical_features} | "
            f"<b>Num of numerical features [int, float]:</b> {num_of_numerical_features}<br></br>"
        )
    )

    rows = []
    for col in features:
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        mode_val = s.mode()
        vc = s.value_counts(ascending=True)

        rows.append({
            "feature": col,
            "dtype": str(s.dtype),
            "unique": s.nunique(),
            "missing": s.isnull().sum(),
            "%_missing": round(100 * s.isnull().mean(), 2),
            "zeros": (s == 0).sum() if is_num else "N/A",
            "%_zeros":round(100 * (s == 0).mean(), 4) if is_num else "N/A",
            "mean": round(s.mean(), 4) if is_num else "N/A",
            "median": round(s.median(), 4) if is_num else "N/A",
            "std": round(s.std(), 4) if is_num else "N/A",
            "skw": round(s.skew(), 4) if is_num else "N/A",
            "kurtosis": round(s.kurtosis(), 4) if is_num else "N/A",
            "most_freq": (
                mode_val.iloc[0] if not is_num and not mode_val.empty else "N/A"
            ),
            "less_freq": vc.index[0] if not is_num and not vc.empty else "N/A",
        })
        
    return pd.DataFrame(rows).set_index("feature").T

def correlation_diff_heatmap(df, features, target=TARGET, method="spearman"):
    """
    Compares the feature-feature correlation matrix within each target class.
    A large difference between the two means the relationship between two
    features changes (or breaks) for stressed customers, which a single
    feature-vs-target correlation can't show.
    """
    corr_0 = df[df[target] == 0][features].corr(method=method)
    corr_1 = df[df[target] == 1][features].corr(method=method)
    diff = corr_1 - corr_0

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    sns.heatmap(corr_0, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title(f"{target}=0 (no stress)")
    sns.heatmap(corr_1, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title(f"{target}=1 (stress)")
    sns.heatmap(diff, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[2])
    axes[2].set_title("Difference (target=1 − target=0)")

    plt.tight_layout()
    plt.show()
    return corr_0, corr_1, diff

def collapse_monthly_columns(df, agg=("mean", "slope")):
    """
    Detects every column following the m{1-6}_<base> naming convention (deposit,
    withdraw, daily_avg_bal, etc.), groups them by <base>, and replaces the 6
    monthly columns with 1-2 representative columns: the mean level and the
    trend slope (m6 -> m1). This removes the same-metric, cross-month
    redundancy that dominated the previous SHAP interaction run, without
    hardcoding which feature_group each column belongs to.
    """
    pattern = re.compile(r"^m([1-6])_(.+)$")
    groups = {}
    for col in df.columns:
        m = pattern.match(col)
        if m:
            month, base = int(m.group(1)), m.group(2)
            groups.setdefault(base, {})[month] = col

    new_cols = {}
    for base, month_map in groups.items():
        if len(month_map) < 2:
            continue

        months_sorted = sorted(month_map)
        cols_sorted = [month_map[m] for m in months_sorted]
        values = df[cols_sorted].to_numpy(dtype=float)

        if "mean" in agg:
            new_cols[f"{base}_mean"] = values.mean(axis=1)

        if "slope" in agg:
            x = np.array([7 - m for m in months_sorted], dtype=float)
            x_centered = x - x.mean()
            denom = (x_centered ** 2).sum()
            new_cols[f"{base}_slope"] = (values * x_centered).sum(axis=1) / denom

    monthly_original_cols = {c for month_map in groups.values() for c in month_map.values()}
    non_monthly_cols = [c for c in df.columns if c not in monthly_original_cols]

    reduced = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df[non_monthly_cols], reduced], axis=1)

def target_conditional_correlation(df, x_col, y_col, target=TARGET, method="spearman"):
    """
    Correlation between two features (already one value per customer, e.g.
    the _mean/_slope columns from collapse_monthly_columns), computed
    separately within each target class.
    """
    return {
        t: df.loc[df[target] == t, x_col].corr(df.loc[df[target] == t, y_col], method=method)
        for t in sorted(df[target].unique())
    }

def joint_participation_state(df, x_prefix, y_prefix, month, participation_col="volume"):
    x_col = f"m{month}_{x_prefix}_{participation_col}"
    y_col = f"m{month}_{y_prefix}_{participation_col}"
    x_active = df[x_col] > 0
    y_active = df[y_col] > 0

    return pd.Series(
        np.select(
            [x_active & y_active, x_active & ~y_active, ~x_active & y_active, ~x_active & ~y_active],
            ["both_active", f"{x_prefix}_only", f"{y_prefix}_only", "both_inactive"],
            default="unknown"
        ),
        index=df.index
    )
    

def intensity_correlation_active_only(df, x_prefix, y_prefix, metric="total_value",
                                       target=TARGET, method="spearman"):
    """
    Same-month correlation between x_prefix and y_prefix, computed ONLY among
    customers active in both — isolates the intensity signal from the
    participation signal handled separately above.
    """
    rows = []
    for m in range(1, 7):
        x_col, y_col = f"m{m}_{x_prefix}_{metric}", f"m{m}_{y_prefix}_{metric}"
        active_both = (df[x_col] > 0) & (df[y_col] > 0)
        for t in sorted(df[target].unique()):
            subset = df[(df[target] == t) & active_both]
            r = subset[x_col].corr(subset[y_col], method=method)
            rows.append({"month": m, target: t, "corr": r, "n": len(subset)})
    return pd.DataFrame(rows)

def find_redundant_pairs(df, features=None, threshold=0.9, method="spearman"):
    """
    Computes pairwise correlation among `features` (numeric columns of df by
    default) and returns every pair whose |correlation| >= threshold, sorted
    by strength. Meant to flag near-duplicate columns before Feature
    Engineering (block 11) or clustering (3.8) — returned as a sorted table,
    not a full heatmap, since we already found the 144-cell heatmap
    unintuitive with more than ~15 columns.
    """
    if features is None:
        features = df.select_dtypes(include="number").columns.tolist()

    corr = df[features].corr(method=method)

    pairs = []
    for i, f1 in enumerate(features):
        for f2 in features[i + 1:]:
            r = corr.loc[f1, f2]
            if abs(r) >= threshold:
                pairs.append({"feature_1": f1, "feature_2": f2, "corr": r})

    n_possible = len(features) * (len(features) - 1) // 2
    result = (
        pd.DataFrame(pairs)
        .sort_values("corr", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )

    print(f"{len(result)} redundant pairs found (|{method} corr| >= {threshold}) "
          f"out of {n_possible} possible pairs across {len(features)} features")
    return result

def drop_redundant_mean_metrics(df, keep_metric="total_value", exceptions=None):
    """
    Drops redundant volume/total_value/highest_amount mean columns per group,
    keeping only `keep_metric` -- except for groups listed in `exceptions`,
    where highest_amount is kept alongside it because it showed a materially
    different (not just noisier) correlation with target (checked for
    `withdraw` in 3.6.1.2's validation cells).
    """
    exceptions = exceptions or {}
    pattern = re.compile(r"^(.+)_(volume|total_value|highest_amount)_mean$")
    groups = {}
    for col in df.columns:
        m = pattern.match(col)
        if m:
            action, metric = m.groups()
            groups.setdefault(action, {})[metric] = col

    cols_to_drop = []
    for action, metrics in groups.items():
        if len(metrics) < 2:
            continue
        keep_cols = {metrics.get(keep_metric, next(iter(metrics.values())))}
        if action in exceptions:
            keep_cols.add(metrics.get(exceptions[action]))
        cols_to_drop.extend(col for metric, col in metrics.items() if col not in keep_cols)

    print(f"Dropping {len(cols_to_drop)} redundant mean columns")
    return df.drop(columns=cols_to_drop), cols_to_drop

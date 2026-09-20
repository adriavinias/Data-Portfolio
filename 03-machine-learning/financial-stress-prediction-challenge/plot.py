import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import trim_mean

TARGET = "liquidity_stress_next_30d"
from utils import * 

def plot(df, dic, group):
    """
        Depending on the number of features per group make grid.
    """
    features = dic[dic["feature_group"] == group]["column_name"].values

    width = 9
    height = 8.5
    
    n_columns = 3
    n_rows = int(np.ceil(len(features) / n_columns))
    
    
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns * width, n_rows * height))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        is_cat = isinstance(df[feature].dtype, pd.StringDtype)    
        if is_cat:
            df_counts = df.groupby([feature, TARGET]).agg(counts=(TARGET,"count")).reset_index()
            df_counts['percentage'] = (
                df_counts['counts'] / df_counts.groupby(feature)['counts'].transform('sum')
            ) * 100

            sns.barplot(
                data=df_counts,
                x=feature,
                y='counts',
                hue=TARGET,
                ax=axes[i]
            )

            hue_order = df_counts[TARGET].unique()

            for container, hue_val in zip(axes[i].containers, hue_order):
                # Filtrar porcentajes correspondientes al valor actual de hue
                pct_values = df_counts[df_counts[TARGET] == hue_val]['percentage']
                
                # Formatear como string "XX.X%"
                labels = [f'{val:.1f}%' for val in pct_values]
                
                # Aplicar las etiquetas personalizadas
                axes[i].bar_label(container, labels=labels, padding=3)
            
            
            axes[i].tick_params("x", rotation=45, rotation_mode="xtick")

        else:
            mean_ = df[feature].mean()
            median_ = df[feature].median()

            sns.kdeplot(data=df, x=feature, ax=axes[i], hue=TARGET, fill=True, common_norm=False, bw_adjust=0.5, cut=0)

            mean_line = axes[i].axvline(mean_, color="red", lw=2, ls="--", alpha=0.9, label=f"Mean = {mean_:.2f}")
            median_line = axes[i].axvline(median_, color="blue", lw=2, ls="-.", alpha=0.9, label=f"Median = {median_:.2f}")

            # sns.kdeplot already draws its own hue legend on this axis; keep it
            # and add a second legend for the mean/median lines so both show up.
            hue_legend = axes[i].get_legend()
            if hue_legend is not None:
                axes[i].add_artist(hue_legend)

            axes[i].legend(handles=[mean_line, median_line], fontsize=11, loc="center right", frameon=True)
            axes[i].tick_params("x", rotation=45, rotation_mode="xtick")

    for i in range(len(features), len(axes.flatten())):
        fig.delaxes(axes[i]) 
                   
    plt.tight_layout()
    plt.show()

def plot_kde_grid(df: pd.DataFrame, substr: str, ncols: int = 4, color: str = "green", cut:bool = False):
    """
    Plot a KDE distribution for every column containing `substr`, with a
    dashed vertical line and text annotation marking the feature's mean.
    """
    cols = [c for c in df.columns if substr in c]
    nrows = int(np.ceil(len(cols) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 4.5))
    axes = np.atleast_1d(axes).flatten()

    for i, col in enumerate(cols):
        mean_ = df[df[col] > 0][col].mean() if cut else df[col].mean()
        median_ = df[df[col] > 0][col].median() if cut else df[col].median()
        #sns.kdeplot(data=df, x=col, ax=axes[i], fill=True, color=color)
        if cut:
            sns.histplot(data=df[df[col] > 0], x=col, ax=axes[i], kde=True, color=color)
        else:
            sns.histplot(data=df, x=col, ax=axes[i], kde=True, color=color)
        axes[i].axvline(mean_, linestyle="--", color="red", label=f"Mean for {col}")
        axes[i].axvline(median_, linestyle="--", color="orange", label=f"Median for {col}")
        axes[i].text(
            x=mean_ * 1.15,
            y=0.85,
            s=f"{mean_:.2f}",
            color="black",
            fontsize=12,
            transform=axes[i].get_xaxis_transform(),
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="black",
                linewidth=.75,
                alpha=0.8
            )
        )
        axes[i].text(
            x=median_ * 1.15,
            y=0.425,
            s=f"{median_:.2f}",
            color="black",
            fontsize=12,
            transform=axes[i].get_xaxis_transform(),
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="black",
                linewidth=.75,
                alpha=0.8
            )
        )
        axes[i].legend(fontsize=12, loc="upper right")

    # hide unused axes if len(cols) isn't a multiple of ncols
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_participation_evolution(df: pd.DataFrame, prefix: str, participation_col: str = "volume",
                                  target: str = TARGET, title: str | None = None, ax=None):
    """
    Plot the month-by-month (m6 -> m1) participation rate (% of customers with
    activity, i.e. value > 0) for a feature group, stratified by target.
    Uses a single representative column per month since the sibling columns
    within a group (volume, total_value, highest_amount, ...) are zero at
    exactly the same rows.
    """
    months = range(1, 7)
    cols = [f"m{m}_{prefix}_{participation_col}" for m in months]

    df_part = df[cols + [target]].copy()
    for c in cols:
        df_part[c] = (df_part[c] > 0).astype(int)

    df_melt = pd.melt(df_part, id_vars=target, value_vars=cols,
                       var_name="month_feature", value_name="participates")
    df_melt["month"] = df_melt["month_feature"].str.extract(r"(\d+)").astype(int)

    # mean of a 0/1 indicator, computed within each target group, is exactly
    # the participation rate for that group -> normalizes for class imbalance
    rate = df_melt.groupby(["month", target])["participates"].mean().reset_index()
    rate["participation_rate"] = rate["participates"] * 100
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.lineplot(data=rate, x="month", y="participation_rate", hue=target,
                 marker="o", linewidth=2.5, ax=ax)

    ax.set_title(f"Participation rate evolution — {title or prefix} (m6 a m1)")
    ax.set_xlabel("Months (6 = Older, 1 = Recent)")
    ax.set_ylabel("% of customers with activity (>0)")
    ax.set_xticks(range(1, 7))
    ax.invert_xaxis()
    ax.set_ylim(0, 100)
    return ax
    
def plot_monthly_evolution_v2(df, prefix, options, title, target=TARGET, participation_col="volume"):
    """
    For each option, draws two side-by-side panels sharing the same x-axis (m6 -> m1):
      - Left: evolution across ALL customers (participation not filtered), with the
        participation rate overlaid on a secondary y-axis (dashed, alpha=0.65) so the
        reader can see when a dip/rise in the metric coincides with customers dropping
        out, instead of a genuine change in behaviour among active customers.
      - Right: the same evolution, computed ONLY on customers active that month
        (value > 0) -- the "intensity" view, isolated from participation.
    """
    palette = {0: "#1f77b4", 1: "#d62728"}
    n_options = len(options)
    fig, axes = plt.subplots(n_options, 2, figsize=(16, 4.5 * n_options))
    if n_options == 1:
        axes = axes.reshape(1, 2)

    for i, option in enumerate(options):
        ax_all, ax_active = axes[i, 0], axes[i, 1]
        value_cols = [f"m{m}_{prefix}_{option}" for m in range(1, 7)]
        part_cols = [f"m{m}_{prefix}_{participation_col}" for m in range(1, 7)]

        # --- Left panel: all customers, mean overlaid with participation rate ---
        df_melt = pd.melt(df, id_vars=target, value_vars=value_cols,
                           var_name="month_feature", value_name=option)
        df_melt["month"] = df_melt["month_feature"].str.extract(r"(\d+)").astype(int)

        for t in sorted(df[target].unique()):
            sub = df_melt[df_melt[target] == t]
            agg = sub.groupby("month")[option].apply(lambda s: trim_mean(s, 0.05))
            ax_all.plot(agg.index, agg.values, marker="o", linewidth=2.5,
                        color=palette[t], label=f"{target}={t}")

        ax_all.set_title(f"{title} — {option} (all customers)", fontsize=11)
        ax_all.set_xlabel("Months (6 = Older, 1 = Recent)")
        ax_all.set_ylabel(f"Avg {option}")
        ax_all.set_xticks(range(1, 7))
        ax_all.invert_xaxis()

        part_rows = []
        for m in range(1, 7):
            active = (df[f"m{m}_{prefix}_{participation_col}"] > 0).astype(int)
            for t in sorted(df[target].unique()):
                part_rows.append({"month": m, target: t, "pct": active[df[target] == t].mean() * 100})
        part_df = pd.DataFrame(part_rows)

        ax_twin = ax_all.twinx()
        for t in sorted(df[target].unique()):
            sub = part_df[part_df[target] == t]
            ax_twin.plot(sub["month"], sub["pct"], linestyle="--", linewidth=2,
                         alpha=0.65, color=palette[t], label=f"{target}={t} participation")
        ax_twin.set_ylabel("% active (participation)")
        ax_twin.set_ylim(0, 100)

        lines1, labels1 = ax_all.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax_all.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

        # --- Right panel: active-only, isolates intensity from participation ---
        rows_active = []
        for m in range(1, 7):
            mask = df[f"m{m}_{prefix}_{participation_col}"] > 0
            sub = df.loc[mask, [target]].copy()
            sub[option] = df.loc[mask, f"m{m}_{prefix}_{option}"]
            sub["month"] = m
            rows_active.append(sub)
        df_active = pd.concat(rows_active, ignore_index=True)

        for t in sorted(df[target].unique()):
            sub = df_active[df_active[target] == t]
            agg = sub.groupby("month")[option].apply(lambda s: trim_mean(s, 0.05))
            ax_active.plot(agg.index, agg.values, marker="o", linewidth=2.5,
                           color=palette[t], label=f"{target}={t}")

        ax_active.set_title(f"{title} — {option} (active only)", fontsize=11)
        ax_active.set_xlabel("Months (6 = Older, 1 = Recent)")
        ax_active.set_ylabel(f"Avg {option} (active customers)")
        ax_active.set_xticks(range(1, 7))
        ax_active.invert_xaxis()
        ax_active.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.show()
    
    
def plot_joint_inactivity_evolution(df, x_prefix, y_prefix, participation_col="volume",
                                     target=TARGET, title=None):
    """
    Plots the % of customers inactive in BOTH x_prefix and y_prefix that
    month, stratified by target. Joint inactivity is a distinct signal from
    intensity — captures customers who shut down activity entirely, not just
    customers whose active spending changed.
    """
    rows = []
    for m in range(1, 7):
        state = joint_participation_state(df, x_prefix, y_prefix, m, participation_col)
        tmp = pd.DataFrame({"month": m, target: df[target].values, "state": state.values})
        rows.append(tmp)
    state_df = pd.concat(rows, ignore_index=True)

    pct = (
        state_df.groupby(["month", target, "state"]).size()
        / state_df.groupby(["month", target]).size()
    ).rename("pct").reset_index()

    both_inactive = pct[pct["state"] == "both_inactive"]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=both_inactive, x="month", y="pct", hue=target, marker="o", linewidth=2.5, ax=ax)
    ax.set_xticks(range(1, 7))
    ax.invert_xaxis()
    ax.set_ylabel(f"% customers inactive in both {x_prefix} & {y_prefix}")
    ax.set_title(title or f"Joint inactivity evolution: {x_prefix} & {y_prefix} (m6 -> m1)")
    plt.tight_layout()
    plt.show()
    return state_df

def plot_same_month_correlation(corr_0, corr_1, x_stem, y_stem, months=range(1, 7),
                                 method="spearman", title=None):
    """
    Extracts the same-month correlation (e.g. m1_deposit vs m1_withdraw) from
    two pre-computed correlation matrices, and plots its evolution m6 -> m1.
    """
    rows = []
    for m in months:
        x_col, y_col = f"m{m}_{x_stem}", f"m{m}_{y_stem}"
        rows.append({"month": m, "target": 0, "corr": corr_0.loc[x_col, y_col]})
        rows.append({"month": m, "target": 1, "corr": corr_1.loc[x_col, y_col]})
    corr_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=corr_df, x="month", y="corr", hue="target", marker="o", linewidth=2.5, ax=ax)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(range(1, 7))
    ax.invert_xaxis()
    ax.set_title(title or f"{x_stem} vs {y_stem} correlation, same month (m6 -> m1)")
    ax.set_xlabel("Months (6 = Older, 1 = Recent)")
    ax.set_ylabel(f"{method.title()} correlation")
    plt.tight_layout()
    plt.show()
    return corr_df

def plot_top_interaction_correlations(df, top_pairs_df, target=TARGET, method="spearman", top_n=10):
    """
    For the top-N SHAP-flagged interaction pairs, plots the target-conditional
    correlation as paired bars. This is the "Correlations" leg of 3.6, using
    SHAP's data-driven pair list (3.6.1) instead of only hypothesis-driven pairs.
    A pair can have strong SHAP interaction but weak/similar correlation across
    target classes -- that would mean the model uses a non-linear combination
    Spearman can't capture, which is worth flagging rather than assuming
    correlation always confirms the SHAP result.
    """
    rows = []
    for _, row in top_pairs_df.head(top_n).iterrows():
        x_col, y_col = row["feature_1"], row["feature_2"]
        for t, r in target_conditional_correlation(df, x_col, y_col, target, method).items():
            rows.append({"pair": f"{x_col}\n× {y_col}", target: t, "corr": r})
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.6)))
    sns.barplot(data=plot_df, y="pair", x="corr", hue=target, ax=ax)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel(f"{method.title()} correlation, within target class")
    ax.set_title("SHAP-flagged pairs: correlation by target class")
    plt.tight_layout()
    plt.show()
    return plot_df
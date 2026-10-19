"""Exploratory data analysis: writes figures + a text summary to reports/."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config  # noqa: E402
from src.data import load_clean, load_raw  # noqa: E402


def run():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    df = load_clean()
    out = config.FIGURES_DIR

    # 1. class balance
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.countplot(x=config.TARGET, data=df, hue=config.TARGET, legend=False, ax=ax,
                  palette=["#4C78A8", "#E45756"])
    ax.set_xticks([0, 1], ["No disease", "Disease"])
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    ax.set_title("Class balance")
    fig.tight_layout()
    fig.savefig(out / "class_balance.png", dpi=150)

    # 2. numeric histograms split by target
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, col in zip(axes.ravel(), config.NUMERIC_FEATURES):
        sns.histplot(data=df, x=col, hue=config.TARGET, kde=True, bins=20, ax=ax,
                     palette=["#4C78A8", "#E45756"])
    axes.ravel()[-1].axis("off")
    fig.suptitle("Numeric feature distributions by target (1 = disease)")
    fig.tight_layout()
    fig.savefig(out / "numeric_histograms.png", dpi=150)

    # 3. correlation heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(out / "correlation_heatmap.png", dpi=150)

    # 4. categorical features vs target
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, col in zip(axes.ravel(), config.CATEGORICAL_FEATURES):
        sns.countplot(data=df, x=col, hue=config.TARGET, ax=ax, palette=["#4C78A8", "#E45756"])
        ax.legend(title="target", fontsize=7)
    fig.suptitle("Categorical features vs target")
    fig.tight_layout()
    fig.savefig(out / "categorical_vs_target.png", dpi=150)

    # 5. boxplots for outliers
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.5))
    for ax, col in zip(axes, config.NUMERIC_FEATURES):
        sns.boxplot(data=df, x=config.TARGET, y=col, hue=config.TARGET, legend=False, ax=ax,
                    palette=["#4C78A8", "#E45756"])
    fig.tight_layout()
    fig.savefig(out / "boxplots.png", dpi=150)
    plt.close("all")

    raw = load_raw()
    summary = [
        f"Raw shape: {raw.shape}; clean shape: {df.shape}",
        "Missing values in raw data:", raw.isna().sum()[raw.isna().sum() > 0].to_string(),
        "Class balance (clean):", df[config.TARGET].value_counts().to_string(),
        "Numeric summary:", df[config.NUMERIC_FEATURES].describe().round(2).to_string(),
    ]
    (config.ROOT / "reports" / "eda_summary.txt").write_text("\n".join(summary))
    print("\n".join(summary))


if __name__ == "__main__":
    run()

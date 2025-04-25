import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .stats import predict_infection


def plot_viral_intensity_over_time(
    pos_ids: list[str], list_of_dataframes: list[pd.DataFrame]
):
    """
    Plots the mean and confidence interval of virus_mCherry intensity over time for each position.

    Args:
        pos_ids (list[str]): Position IDs.
        list_of_dataframes (list[pd.DataFrame]): Corresponding stats dataframes.
    """
    combined_df = []
    for pos_id, df in zip(pos_ids, list_of_dataframes, strict=True):
        temp_df = df[["time", "mean_intensity-virus_mCherry"]].copy()
        temp_df["position"] = pos_id
        combined_df.append(temp_df)

    merged_df = pd.concat(combined_df, ignore_index=True)

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=merged_df,
        x="time",
        y="mean_intensity-virus_mCherry",
        hue="position",
        ci="sd",  # Use "sd" for shaded standard deviation; can also use "95" for 95% CI
        marker="o",
    )
    plt.title(
        "Mean Viral Intensity Over Time (mCherry) with Confidence Bounds"
    )
    plt.xlabel("Time")
    plt.ylabel("Mean Intensity (mCherry)")
    plt.legend(title="Position", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cell_count_over_time(
    pos_ids: list[str], list_of_dataframes: list[pd.DataFrame]
):
    plt.figure(figsize=(12, 6))
    for pos_id, df in zip(pos_ids, list_of_dataframes, strict=True):
        counts = df.groupby("time")["label"].nunique()
        plt.plot(counts.index, counts.values, label=pos_id, marker="s")
    plt.title("Cell Count Over Time")
    plt.xlabel("Time")
    plt.ylabel("Cell Count")
    plt.legend(title="Position")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_mean_dapi_vs_virus(
    pos_ids: list[str], list_of_dataframes: list[pd.DataFrame]
):
    plt.figure(figsize=(12, 6))
    for pos_id, df in zip(pos_ids, list_of_dataframes, strict=True):
        plt.scatter(
            df["mean_intensity-nuclei_DAPI"],
            df["mean_intensity-virus_mCherry"],
            label=pos_id,
            alpha=0.6,
        )
    plt.title("Mean DAPI vs mCherry Intensity")
    plt.xlabel("Mean DAPI Intensity")
    plt.ylabel("Mean mCherry Intensity")
    plt.legend(title="Position")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_phase_intensity_over_time(
    pos_ids: list[str], list_of_dataframes: list[pd.DataFrame]
):
    plt.figure(figsize=(12, 6))
    for pos_id, df in zip(pos_ids, list_of_dataframes, strict=True):
        grouped = df.groupby("time")["mean_intensity-Phase3D"].mean()
        plt.plot(grouped.index, grouped.values, label=pos_id, marker="^")
    plt.title("Mean Phase3D Intensity Over Time")
    plt.xlabel("Time")
    plt.ylabel("Mean Phase Intensity")
    plt.legend(title="Position")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_predicted_infection_over_time(
    pos_ids: list[str], dfs: list[pd.DataFrame]
):
    plt.figure(figsize=(12, 6))
    for pos_id, df in zip(pos_ids, dfs, strict=True):
        if "infected" not in df.columns:
            df = predict_infection(df)
        grouped = df.groupby("time")["infected"].mean()
        sns.lineplot(x=grouped.index, y=grouped.values, label=pos_id)

    plt.title("Predicted Infection Rate Over Time")
    plt.xlabel("Time")
    plt.ylabel("Fraction of Cells Predicted Infected")
    plt.grid(True)
    plt.legend(title="Position")
    plt.tight_layout()
    plt.show()


def plot_infection_rate_change_over_time(
    pos_ids: list[str], dfs: list[pd.DataFrame]
):
    plt.figure(figsize=(12, 6))

    for pos_id, df in zip(pos_ids, dfs, strict=True):
        # Ensure infection labels are present
        if "infected" not in df.columns:
            df = predict_infection(df)

        grouped = df.groupby("time")["infected"].mean()
        rate_of_change = grouped.diff().fillna(0)

        sns.lineplot(x=grouped.index, y=rate_of_change.values, label=pos_id)

    plt.title("Change in Predicted Infection Rate Over Time")
    plt.xlabel("Time")
    plt.ylabel("Δ Fraction Infected (Rate of Change)")
    plt.grid(True)
    plt.legend(title="Position")
    plt.tight_layout()
    plt.show()

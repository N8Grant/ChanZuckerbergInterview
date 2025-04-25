from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from iohub import open_ome_zarr
from iohub.reader import Position
from skimage.filters import threshold_otsu
from skimage.measure import regionprops_table
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


def extract_cell_stats(
    dataset_path: str | Path,
    seg_name: str = "Nuclei_Segmentation",
    save_dir: str | Path | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Extracts cell statistics from a segmented 3D time-series OME-Zarr dataset.

    Args:
        dataset_path: Path to the OME-Zarr dataset.
        seg_name: Name of the segmentation array within each position.
        save_dir: Optional path to save extracted DataFrames as CSVs.

    Returns:
        A nested dictionary: {well_id: {pos_id: DataFrame}}.
    """
    dataset_path = Path(dataset_path)
    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    combined_statistics = {}
    with open_ome_zarr(dataset_path, mode="r") as dataset:
        for well_id, well in dataset.wells():
            for pos_id, pos in tqdm(
                well.positions(), desc="Collecting Statistics"
            ):
                pos = cast(Position, pos)
                sample_stats = []
                seg = pos[seg_name]  # shape: (T, Z, Y, X)
                img = pos["0"]  # shape: (T, C, Z, Y, X)

                for t in range(seg.shape[0]):
                    labels_t = seg[t][0]
                    image_t = img[t]

                    # Move channel to last axis: (Z, Y, X, C)
                    intensity_image = np.moveaxis(image_t, 0, -1)

                    props = regionprops_table(
                        labels_t,
                        intensity_image=intensity_image,
                        properties=[
                            "label",
                            "area",
                            "centroid",
                            "mean_intensity",
                            "max_intensity",
                            "min_intensity",
                            "extent",
                        ],
                    )
                    df = pd.DataFrame(props)
                    df["time"] = t

                    # Rename channel index suffixes with actual names
                    df = rename_channel_columns(df, pos.channel_names)
                    sample_stats.append(df)

                well_pos_df = pd.concat(sample_stats, ignore_index=True)
                if well_id not in combined_statistics:
                    combined_statistics[well_id] = {}
                combined_statistics[well_id][pos_id] = well_pos_df

                if save_dir:
                    out_path = Path(save_dir) / well_id
                    out_path.mkdir(
                        parents=True, exist_ok=True
                    )  # Ensure well-specific folder exists
                    file_path = out_path / f"{pos_id}_stats.csv"
                    well_pos_df.to_csv(file_path, index=False)

    return combined_statistics


def rename_channel_columns(
    df: pd.DataFrame, channel_names: list[str]
) -> pd.DataFrame:
    """
    Renames columns like mean_intensity-0 to mean_intensity-DAPI, etc.

    Skips centroid-* columns which are spatial.
    """
    renamed_columns = {}
    for col in df.columns:
        for i, ch in enumerate(channel_names):
            suffix = f"-{i}"
            if col.endswith(suffix) and not col.startswith("centroid"):
                renamed_columns[col] = col.replace(suffix, f"-{ch}")
    return df.rename(columns=renamed_columns)


def predict_infection(df: pd.DataFrame) -> pd.DataFrame:
    # Heuristic infection label
    threshold = threshold_otsu(df["mean_intensity-virus_mCherry"].values)
    df["infected"] = (df["mean_intensity-virus_mCherry"] > threshold).astype(
        int
    )

    features = [
        "area",
        "centroid-0",  # Z position
        "centroid-1",  # Z position
        "centroid-2",  # Z position
        "mean_intensity-virus_mCherry",
        "mean_intensity-nuclei_DAPI",
        "mean_intensity-Phase3D",
    ]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(df[features], df["infected"])
    df["predicted_infected"] = clf.predict(df[features])

    return df


# Example usage for manual runs
if __name__ == "__main__":
    dfs = extract_cell_stats(
        "./data/20241107_infection.zarr",
        seg_name="Nuclei_Segmentation",
        save_dir="./data/stats_output",
    )
    print(
        f"✅ Extracted statistics for {sum(len(p) for p in dfs.values())} positions."
    )

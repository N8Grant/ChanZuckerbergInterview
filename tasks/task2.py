"""
Segment the cells’ nuclei from your channel of choice and store the
segmentations into an OME-Zarr store using iohub as Zarr Arrays.
- Computes at least 5 metrics or algorithms to characterize and detect the infection
dynamics. (e.g. intensity, segmentation, etc). Can you find the infected cells?
- Generates visualizations of image data and the image analysis results (i.e, matplotlib,
napari, etc).
- Save the visualizations (e.g as .png, .mp4,.jpg, etc). You will share this
output with us


This script will showcase the use of the segmentation api and the toolkit that I built to
examine the given cell datasets.
"""

from chanzuck.segment.nuclei_segmentation import segment_and_track_3d_over_time
from chanzuck.spatial.stats import extract_cell_stats
from chanzuck.spatial.visualize import (
    plot_cell_count_over_time,
    plot_infection_rate_change_over_time,
    plot_mean_dapi_vs_virus,
    plot_phase_intensity_over_time,
    plot_predicted_infection_over_time,
    plot_viral_intensity_over_time,
)


def run_segmentation(
    dataset_path: str, nuclei_channel_index: int, save_dir: str
):

    # Run segmentation
    segment_and_track_3d_over_time(
        dataset_path,
        nuclei_channel_index,
        model_type="otsu",
        on_level=0,
    )

    # Extract statistics from the images
    cell_stats_dict = extract_cell_stats(
        dataset_path=dataset_path, save_dir=save_dir
    )

    names = []
    dfs = []
    for well_id, pos_dict in cell_stats_dict.items():
        for pos_id, df in pos_dict.items():
            names.append(f"{well_id}_{pos_id}")
            dfs.append(df)

    # Plot quantitities of interest
    plot_viral_intensity_over_time(names, dfs)
    plot_predicted_infection_over_time(names, dfs)
    plot_infection_rate_change_over_time(names, dfs)
    plot_cell_count_over_time(names, dfs)
    plot_mean_dapi_vs_virus(names, dfs)
    plot_phase_intensity_over_time(names, dfs)


if __name__ == "__main__":
    # 🔁 Run both GPU and CPU benchmarks
    dataset_path = "../data/20241107_infection.zarr"
    save_dir = "../data/statistics_folder"
    nuclei_channel_index = 1  # Set to DAPI channel or whichever you want

    run_segmentation(dataset_path, nuclei_channel_index, save_dir)

import click


@click.command("view")
@click.option(
    "--dataset-path",
    type=click.Path(exists=True, dir_okay=True),
    required=True,
    help="Path to the OME-Zarr dataset.",
)
@click.option(
    "--show-segmentations/--no-show-segmentations",
    default=True,
    show_default=True,
    help="Whether to display segmentation masks if available.",
)
def view(dataset_path, show_segmentations):
    """
    Launch Napari to visualize OME-Zarr datasets.
    """
    from chanzuck.visualize.image_visualizer import view_image

    view_image(dataset_path, show_segmentations)


@click.command("plot-stats")
@click.option(
    "--stats-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Directory containing .csv files for each position.",
)
def plot_stats(stats_dir):
    """
    Plot some interesting features over the desired image statistics
    """
    import os

    import pandas as pd

    from chanzuck.spatial.visualize import (
        plot_cell_count_over_time,
        plot_infection_rate_change_over_time,
        plot_mean_dapi_vs_virus,
        plot_phase_intensity_over_time,
        plot_predicted_infection_over_time,
        plot_viral_intensity_over_time,
    )

    dfs = []
    pos_ids = []
    for fname in os.listdir(stats_dir):
        if fname.endswith(".csv"):
            pos_ids.append(fname.replace(".csv", ""))
            dfs.append(pd.read_csv(os.path.join(stats_dir, fname)))

    plot_viral_intensity_over_time(pos_ids, dfs)
    plot_predicted_infection_over_time(pos_ids, dfs)
    plot_infection_rate_change_over_time(pos_ids, dfs)
    plot_cell_count_over_time(pos_ids, dfs)
    plot_mean_dapi_vs_virus(pos_ids, dfs)
    plot_phase_intensity_over_time(pos_ids, dfs)

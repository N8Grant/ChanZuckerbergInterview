import click


@click.command("generate-stats")
@click.option(
    "--dataset-path",
    type=click.Path(exists=True, dir_okay=True),
    required=True,
    help="Path to the OME-Zarr dataset.",
)
@click.option(
    "--stats-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Directory to put CSVs in according to well and position.",
)
@click.option(
    "--visualize/--no-visualize",
    default=False,
    show_default=True,
    help="Whether to visualize the stats after generating them.",
)
def generate_stats(dataset_path: str, stats_dir: str, visualize: bool):
    """
    Gather features over the segmented image and optionally display plots
    """
    from chanzuck.spatial.stats import extract_cell_stats

    if not visualize:
        _ = extract_cell_stats(dataset_path, save_dir=stats_dir)
        return

    # If the user wants to visualize then import
    from chanzuck.spatial.visualize import (
        plot_cell_count_over_time,
        plot_infection_rate_change_over_time,
        plot_mean_dapi_vs_virus,
        plot_phase_intensity_over_time,
        plot_predicted_infection_over_time,
        plot_viral_intensity_over_time,
    )

    # Flatten the dictionaries to get the well and position id combo
    cell_stats_dict = extract_cell_stats(dataset_path, save_dir=stats_dir)

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

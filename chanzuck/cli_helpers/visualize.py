import click

from chanzuck.visualize.image_visualizer import view_image


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
    view_image(dataset_path, show_segmentations)

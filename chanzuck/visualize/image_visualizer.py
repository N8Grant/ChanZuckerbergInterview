from pathlib import Path
from typing import cast

import dask.array as da
import napari
from iohub import open_ome_zarr
from iohub.reader import Position
from magicgui import magicgui, use_app

from chanzuck.utils.describe import describe_dataset

use_app("qt")

default_colormaps = [
    "magenta",
    "green",
    "blue",
    "yellow",
    "red",
    "cyan",
    "orange",
    "purple",
]


def _safe_axis_labels(array_shape, axis_label_str):
    """
    Ensure axis_labels length matches array dimensions.
    axis_label_str: a comma-separated string from metadata.
    """
    if not isinstance(axis_label_str, str):
        return None
    labels = [label.strip() for label in axis_label_str.split("; ")]
    return labels[-len(array_shape) :]  # trim to match dimensions


def view_image(dataset_path: str | Path, show_segmentations: bool):
    viewer = napari.Viewer()
    dataset_metadata = describe_dataset(dataset_path=dataset_path)
    dataset = open_ome_zarr(dataset_path, mode="r")

    # Build list of all available positions
    all_positions = []
    for well_id in dataset_metadata["Wells"].keys():
        for pos_id in dataset_metadata["Wells"][well_id].keys():
            all_positions.append(f"{well_id}/{pos_id}")

    # Define loader function
    @magicgui(
        auto_call=True,
        pos={"choices": all_positions},
        layout="vertical",
        call_button="Load position",
    )
    def loader(pos: str):
        viewer.layers.clear()
        *well_parts, pos_id = pos.split("/")
        well_id = "/".join(well_parts)

        pos_info = dataset_metadata["Wells"][well_id][pos_id]
        pos_data = dataset[well_id][pos_id]

        if pos_info["multiscale"]:
            # Add multiscale support here if needed later
            image_array = da.from_array(pos_data["0"])
            axis_labels = _safe_axis_labels(
                image_array.shape, pos_info["axes"]
            )
            axis_labels.pop(1)  # Remove channel axis

            for idx, channel in enumerate(pos_info["channels"]):
                multiscale_image = [
                    da.from_array(pos_data[level["level"]][:, idx])
                    for level in pos_info["levels"]
                ]
                colormap = default_colormaps[idx % len(default_colormaps)]

                viewer.add_image(
                    multiscale_image,
                    name=f"{well_id}_{pos_id}_channel_{channel}",
                    channel_axis=None,
                    colormap=colormap,
                    axis_labels=axis_labels,
                    multiscale=True,
                )

        else:
            image_array = da.from_array(pos_data["0"])
            axis_labels = _safe_axis_labels(
                image_array.shape, pos_info["axes"]
            )
            axis_labels.pop(1)  # Remove channel axis

            for idx, channel in enumerate(pos_info["channels"]):
                colormap = default_colormaps[idx % len(default_colormaps)]
                viewer.add_image(
                    image_array[:, idx],
                    name=f"{well_id}_{pos_id}_channel_{channel}",
                    channel_axis=None,
                    colormap=colormap,
                    axis_labels=axis_labels,
                )

        if show_segmentations:
            for array_name, _ in cast(Position, pos_data).images():
                if "Segmentation" in array_name:
                    viewer.add_labels(
                        pos_data[array_name][:, 0],
                        name=f"{well_id}_{pos_id}_segmentation_{array_name}",
                    )

    viewer.window.add_dock_widget(loader, area="right")

    napari.run()


if __name__ == "__main__":
    view_image("./data/20241107_infection.zarr", True)

from pathlib import Path
from typing import cast

import dask.array as da
import napari
from iohub import open_ome_zarr
from iohub.reader import Position

from chanzuck.utils.describe import describe_dataset


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

    for well_id in dataset_metadata["Wells"].keys():
        for pos_id, pos_info in dataset_metadata["Wells"][well_id].items():
            if pos_info["multiscale"]:
                # TODO
                pass

            else:
                image_array = da.from_array(dataset[well_id][pos_id]["0"])
                axis_labels = _safe_axis_labels(
                    image_array.shape, pos_info["axes"]
                )
                axis_labels.pop(1)
                for idx, channel in enumerate(pos_info["channels"]):
                    viewer.add_image(
                        image_array[:, idx],
                        name=f"{well_id}_{pos_id}_channel_{channel}",
                        channel_axis=None,
                        axis_labels=axis_labels,
                    )

            if show_segmentations:
                for array_name, _ in cast(
                    Position, dataset[well_id][pos_id]
                ).images():
                    if "Segmentation" in array_name:
                        viewer.add_labels(
                            dataset[well_id][pos_id][array_name][:, 0],
                            name=f"{well_id}_{pos_id}_segmentation_{array_name}",
                        )

    napari.run()


if __name__ == "__main__":
    view_image("./data/20241107_infection.zarr", True)

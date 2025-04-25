from pathlib import Path

import numpy as np
from cellpose import models
from iohub import open_ome_zarr
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops

from chanzuck.utils.dataloader import CellposeZarrLoader


def segment_and_track_3d_over_time(
    zarr_path: str | Path,
    channel_index: int = 0,
    model_type: str = "nuclei",
    use_gpu: bool = True,
):
    loader = CellposeZarrLoader(zarr_path, channel_indices=[channel_index])
    print(loader.dataset_chunksizes, loader.dataset_shapes)
    # Get the underlying array paths
    with open_ome_zarr(zarr_path, mode="a") as dataset:
        i = 0
        for _, well in dataset.wells():
            for _, pos in well.positions():
                pos.create_zeros(
                    name=f"{model_type}_segmentation",
                    shape=loader.dataset_shapes[i],
                    dtype="uint32",
                    chunks=loader.dataset_chunksizes[i],  # chunk by XY planes
                )
                i += 1

    model = models.Cellpose(gpu=use_gpu, model_type=model_type)

    previous_labels = None

    for i in range(len(loader)):
        sample = loader[i]
        image = sample["image"]  # (C, Z, Y, X)
        time_idx = sample["time"]
        pos_name = sample["position"]
        well_name = sample["well_name"]

        print(f"🧠 Segmenting T={time_idx}, Pos={pos_name}")

        masks, *_ = model.eval(image, channels=[0, None], z_axis=1, do_3D=True)

        # 🧩 Match and relabel with previous labels
        if previous_labels is not None:
            masks = iou_label_match(previous_labels, masks)

        # 🧠 Save to .npy (or swap to .zarr/.tiff later)
        with open_ome_zarr(zarr_path, mode="a") as dataset:
            dataset[well_name][pos_name]["Nuclei_Segmentation"][
                time_idx
            ] = masks

        # 🔁 Carry over
        previous_labels = masks

        # 💡 Optionally: clear variables to free RAM
        del image, sample, masks


def iou_label_match(prev_mask, curr_mask):
    """Returns new mask for curr_mask with labels remapped to match prev_mask where possible."""
    prev_props = regionprops(prev_mask)
    curr_props = regionprops(curr_mask)

    iou_matrix = np.zeros((len(prev_props), len(curr_props)))

    for i, p in enumerate(prev_props):
        for j, c in enumerate(curr_props):
            intersection = np.logical_and(
                prev_mask == p.label, curr_mask == c.label
            ).sum()
            union = np.logical_or(
                prev_mask == p.label, curr_mask == c.label
            ).sum()
            iou_matrix[i, j] = intersection / union if union > 0 else 0

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # maximize IoU
    remap = {
        curr_props[j].label: prev_props[i].label
        for i, j in zip(row_ind, col_ind, strict=True)
        if iou_matrix[i, j] > 0.1
    }

    remapped_mask = np.zeros_like(curr_mask)
    next_label = max([p.label for p in prev_props], default=1) + 1

    for prop in curr_props:
        old_id = prop.label
        new_id = remap.get(old_id, next_label)
        remapped_mask[curr_mask == old_id] = new_id
        if old_id not in remap:
            next_label += 1

    return remapped_mask

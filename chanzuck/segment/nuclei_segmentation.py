from pathlib import Path

import numpy as np
import torch
from cellpose import models
from iohub import open_ome_zarr
from scipy.ndimage import label as ndi_label
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from tqdm import tqdm

from chanzuck.utils.dataloader import CellposeZarrLoader


def segment_and_track_3d_over_time(
    zarr_path: str | Path,
    channel_index: int = 0,
    model_type: str = "cellpose",  # or "otsu"
    use_gpu: bool = False,
    on_level: int = 0,
):
    loader = CellposeZarrLoader(zarr_path, channel_indices=[channel_index])

    if model_type == "cellpose":
        model = models.Cellpose(gpu=use_gpu, model_type="nuclei")
    if use_gpu and not torch.cuda.is_available():
        print("⚠️ GPU requested but not available. Falling back to CPU.")
        use_gpu = False

    # Store scales and prepare output
    dataset_scales = []
    with open_ome_zarr(zarr_path, mode="a") as dataset:
        for i, (_, well) in enumerate(dataset.wells()):
            for _, pos in well.positions():
                pos._overwrite = True
                shape = list(loader.dataset_shapes[i])
                shape[1] = 1  # single-channel for output
                pos.create_zeros(
                    name="Nuclei_Segmentation",
                    shape=tuple(shape),
                    dtype="uint32",
                    chunks=loader.dataset_chunksizes[i],
                )
                dataset_scales.append(pos.scale)

    previous_labels = None

    for i in tqdm(range(len(loader)), desc="Segmenting"):
        sample = loader[i]
        image = sample["image"]  # (C, Z, Y, X)
        time_idx = sample["time"]
        pos_name = sample["position"]
        well_name = sample["well"]

        # Chat gpt
        # --- Inference --- #
        if model_type == "cellpose":
            masks, *_ = model.eval(
                image, channels=[0, None], z_axis=1, do_3D=True
            )
        elif model_type == "otsu":
            img = np.squeeze(image[0])  # (Z, Y, X)
            thresh = threshold_otsu(img)
            masks, _ = ndi_label(img > thresh)
        else:
            raise ValueError(f"Unknown model_type '{model_type}'")

        # --- Tracking --- #
        if previous_labels is not None:
            scale = dataset_scales[0][2:]  # Z, Y, X
            masks = track_labels(previous_labels, masks, scale)

        # --- Save --- #
        with open_ome_zarr(zarr_path, mode="a") as dataset:
            dataset[well_name][pos_name]["Nuclei_Segmentation"][time_idx] = (
                masks[np.newaxis, ...]
            )

        previous_labels = masks
        del image, sample, masks


# Chat gpt
def get_centroids(mask, filter_small: bool = True):
    """Returns centroids and labels for each region > 0 in a 3D mask, with optional area filtering."""
    props = regionprops(mask)

    # Filter out small objects using Otsu thresholding on area
    if filter_small and len(props) > 2:
        areas = np.array([p.area for p in props])
        area_thresh = threshold_otsu(areas)
        props = [p for p in props if p.area >= area_thresh]
    else:
        area_thresh = 0

    centroids = [p.centroid for p in props]
    labels = [p.label for p in props]

    return np.array(centroids), np.array(labels)


# Chat gpt
def track_labels(
    prev_mask,
    curr_mask,
    spatial_scales: tuple[float, float, float],
    max_dist_um=50,
):
    """
    Track and relabel masks based on centroid proximity using scaled distances in microns.

    Args:
        prev_mask: 3D numpy array (Z, Y, X) of previous timepoint labels.
        curr_mask: 3D numpy array (Z, Y, X) of current timepoint labels.
        spatial_scales: (Z_um, Y_um, X_um) spacing in microns.
        max_dist_um: Max distance (in microns) allowed for matching labels.
    """
    prev_centroids, prev_labels = get_centroids(prev_mask)
    curr_centroids, curr_labels = get_centroids(curr_mask)

    if len(prev_centroids) == 0 or len(curr_centroids) == 0:
        return curr_mask

    # Scale centroids by physical spacing
    scale_arr = np.array(spatial_scales)
    prev_scaled = prev_centroids * scale_arr
    curr_scaled = curr_centroids * scale_arr

    # Compute cost matrix with scaled distances
    cost_matrix = cdist(prev_scaled, curr_scaled)
    cost_matrix[cost_matrix > max_dist_um] = np.inf

    if np.all(np.isinf(cost_matrix)):
        return curr_mask

    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ValueError:
        return curr_mask

    relabeled = np.zeros_like(curr_mask, dtype=np.uint32)
    used_labels = set()

    for i, j in zip(row_ind, col_ind, strict=True):
        if cost_matrix[i, j] != np.inf:
            target_label = prev_labels[i]
            current_label = curr_labels[j]
            relabeled[curr_mask == current_label] = target_label
            used_labels.add(target_label)

    new_label = max(used_labels, default=0) + 1
    for label in curr_labels:
        if label == 0:
            continue
        if np.all(relabeled[curr_mask == label] == 0):
            relabeled[curr_mask == label] = new_label
            new_label += 1

    return relabeled

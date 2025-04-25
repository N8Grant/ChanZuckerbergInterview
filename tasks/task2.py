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


def run_segmentation(
    dataset_path: str,
    nuclei_channel_index: int,
    use_gpu: bool = True,
):
    print(f"📁 Dataset: {dataset_path}")
    print(f"🔬 Channels: {nuclei_channel_index} ")

    segment_and_track_3d_over_time(
        dataset_path,
        nuclei_channel_index,
        model_type="otsu",
        use_gpu=use_gpu,
        on_level=0,
    )


# 🔁 Run both GPU and CPU benchmarks
dataset_path = "data/20241107_infection.zarr"
nuclei_channel_index = 1  # Set to DAPI channel or whichever you want

run_segmentation(dataset_path, nuclei_channel_index)

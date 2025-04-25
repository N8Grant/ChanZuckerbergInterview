from pathlib import Path

import dask.array as da
import numpy as np
from iohub import open_ome_zarr


class CellposeZarrLoader:
    def __init__(
        self, zarr_path: str | Path, channel_indices: list[int] | None = None
    ):
        self.zarr_path = Path(zarr_path)
        self.channel_indices = channel_indices or [0]
        self.dataset_shapes = None
        self.dataset_chunksizes = None
        self.entries: list[tuple[str, str, int, da.Array]] = (
            self._gather_timepoint_entries()
        )

    def _gather_timepoint_entries(self):
        """
        Returns a flat list of (well_name, pos_name, time_index, image) tuples.
        Each entry corresponds to one timepoint slice.
        """
        timepoint_entries = []
        self.dataset_shapes = []
        self.dataset_chunksizes = []
        with open_ome_zarr(self.zarr_path, mode="r") as dataset:
            for well_name, well in dataset.wells():
                for pos_name, pos in well.positions():
                    image = pos.data  # shape: (T, C, Z, Y, X)
                    num_timepoints = image.shape[0]
                    self.dataset_shapes.append(image.shape)
                    self.dataset_chunksizes.append(image.chunks)

                    for t in range(num_timepoints):
                        timepoint_entries.append(
                            (well_name, pos_name, t, image[t])
                        )
        return timepoint_entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        well_name, pos_name, t_idx, image_t = self.entries[idx]

        # image_t is (C, Z, Y, X)
        if hasattr(image_t, "as_numpy"):
            image_t = image_t.as_numpy()

        image_t = image_t[self.channel_indices]

        # Normalize each channel independently
        norm = np.array(
            [
                (c - np.min(c)) / (np.max(c) - np.min(c) + 1e-8)
                for c in image_t
            ],
            dtype=np.float32,
        )

        return {
            "image": norm,  # shape: (C, Z, Y, X)
            "well": well_name,
            "position": pos_name,
            "time": t_idx,
            "path": str(image_t.path) if hasattr(image_t, "path") else "N/A",
        }

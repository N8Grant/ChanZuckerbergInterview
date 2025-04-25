from pathlib import Path
from typing import cast

import dask.array as da
import numpy as np
from iohub import open_ome_zarr
from iohub.reader import Position
from tqdm import tqdm


# gpt
def create_downsample_pyramid_for_dataset(
    dataset_path: str | Path, levels: int = 3
):
    """
    Initializes and populates a downsample pyramid for each position in an OME-Zarr dataset.
    Uses Dask for memory-efficient downsampling and tqdm for progress display.

    Args:
        dataset_path (Path or str): Path to OME-Zarr dataset.
        levels (int): Number of pyramid levels to generate. Must be >= 2.
    """
    dataset_path = Path(dataset_path)
    if levels < 2:
        raise ValueError(
            "Pyramid must have at least 2 levels (base + 1 downsample)."
        )

    print(f"📂 Creating downsample pyramid at: {dataset_path}")
    with open_ome_zarr(dataset_path, mode="a") as dataset:
        for well_id, well in tqdm(dataset.wells(), desc="🔹 Wells"):
            for pos_id, pos_node in tqdm(
                well.positions(),
                desc=f"  📍 Positions in {well_id}",
                leave=False,
            ):
                pos = cast(Position, pos_node)

                try:
                    pos._overwrite = True
                    pos.initialize_pyramid(levels)
                except Exception as e:
                    print(
                        f"⚠️ Skipping {well_id}/{pos_id} due to pyramid init failure: {e}"
                    )
                    continue

                base = da.from_array(pos.data)  # shape: (T, C, Z, Y, X)
                n_timepoints = base.shape[0]

                for t in tqdm(
                    range(n_timepoints), desc="    ⏱ Timepoints", leave=False
                ):
                    try:
                        original = base[t]  # shape: (C, Z, Y, X)

                        for level in range(1, levels):
                            factor = 2**level
                            downsampled = da.coarsen(
                                np.mean,
                                original,
                                axes={1: factor, 2: factor, 3: factor},
                                trim_excess=True,
                            )

                            da.store(
                                downsampled[None],  # (1, C, Z, Y, X)
                                pos[str(level)],
                                regions=(
                                    slice(t, t + 1),
                                    slice(0, downsampled.shape[0]),
                                    slice(0, downsampled.shape[1]),
                                    slice(0, downsampled.shape[2]),
                                    slice(0, downsampled.shape[3]),
                                ),
                                compute=True,
                            )

                    except Exception as e:
                        print(
                            f"❌ Failed to downsample {well_id}/{pos_id}/t{t}: {e}"
                        )

    print("🎉 Pyramid creation complete.")


if __name__ == "__main__":
    create_downsample_pyramid_for_dataset("./data/20241107_infection.zarr")

import logging
from pathlib import Path

from iohub import open_ome_zarr

logger = logging.getLogger(__name__)


def describe_dataset(dataset_path: str | Path) -> dict:
    """
    Loads an OME-Zarr dataset and returns structured metadata as a dictionary.

    This function supports Plate format datasets and extracts metadata from each
    position inside each well, including shape, chunk size, dtype, channels, axes, and path.

    Args:
        dataset_path (str | Path): Path to the OME-Zarr dataset.

    Returns:
        dict: Metadata structured by well and position.
    """
    dataset_path = Path(dataset_path)
    metadata: dict = {}

    try:
        # Open dataset using iohub
        dataset = open_ome_zarr(dataset_path, mode="r")

        # Base metadata info
        metadata["dataset_type"] = str(type(dataset))
        metadata["Plate Format"] = hasattr(dataset, "wells")
        metadata["Wells"] = {}

        # Iterate through each well in the plate
        for well_name, well_node in dataset.wells():
            metadata["Wells"][well_name] = {}

            # Iterate through each position in the well
            for pos_name, pos_node in well_node.positions():
                image = (
                    pos_node.data
                )  # The actual image array (not multiscale)
                axes = dataset.axes if hasattr(dataset, "axes") else "N/A"

                # Check if image is multiscale and collect per-scale info
                if hasattr(image, "is_multiscale") and image.is_multiscale:
                    level_info = []
                    for level_idx in range(len(image)):
                        level = image[level_idx]
                        level_info.append(
                            {
                                "level": level_idx,
                                "shape": level.shape,
                                "chunks": level.chunks,
                                "dtype": str(level.dtype),
                            }
                        )
                else:
                    level_info = [
                        {
                            "level": 0,
                            "shape": image.shape,
                            "chunks": image.chunks,
                            "dtype": str(image.dtype),
                        }
                    ]

                metadata["Wells"][well_name][pos_name] = {
                    "multiscale": (
                        image.is_multiscale
                        if hasattr(image, "is_multiscale")
                        else False
                    ),
                    "levels": level_info,
                    "axes": format_axes(axes),
                    "channels": (
                        pos_node.channel_names
                        if hasattr(pos_node, "channel_names")
                        else "N/A"
                    ),
                    "path": str(image.path),
                }

        return metadata

    except Exception as e:
        # Raise error with additional context
        raise ValueError(
            f"Could not parse metadata for dataset {dataset_path}: {e}"
        ) from e


def format_pretty_output(metadata: dict) -> str:
    """
    Formats structured metadata into a human-readable CLI output string.

    Args:
        metadata (dict): The dataset metadata dictionary.

    Returns:
        str: A nicely formatted string for display in the terminal.
    """
    lines: list[str] = []

    lines.append(
        f"📦 Dataset Type: {metadata.get('dataset_type', 'N/A').split('.')[-1].strip()}"
    )
    lines.append(
        f"🧫 Plate Format: {'Yes' if metadata.get('Plate Format') else 'No'}\n"
    )

    wells = metadata.get("Wells", {})
    for well_name, positions in wells.items():
        lines.append(f"🔹 Well: {well_name}")
        for pos_name, pos_data in positions.items():
            lines.append(f"  ├── Position: {pos_name}")
            lines.append(
                f"  │   • Multiscale  : {'Yes' if pos_data.get('multiscale') else 'No'}"
            )
            levels = pos_data.get("levels", [])
            for level_info in levels:
                lvl = level_info.get("level")
                shape = tuple(level_info.get("shape", []))
                chunks = tuple(level_info.get("chunks", []))
                dtype = level_info.get("dtype", "N/A")
                lines.append(f"  │     🔸 Level {lvl}")
                lines.append(f"  │        • Shape  : {shape}")
                lines.append(f"  │        • Chunks : {chunks}")
                lines.append(f"  │        • Dtype  : {dtype}")
            lines.append(
                f"  │   • Channels    : {pos_data.get('channels', 'N/A')}"
            )
            lines.append(
                f"  │   • Axes        : {pos_data.get('axes', 'N/A')}"
            )
            lines.append(
                f"  │   • Path        : {pos_data.get('path', 'N/A')}"
            )

    return "\n".join(lines)


def format_axes(axes_meta: str | list) -> str:
    """
    Formats axis metadata into a readable string.

    If the axes are objects like TimeAxisMeta, ChannelAxisMeta, etc., this will
    extract and join their name/type/unit info.

    Args:
        axes_meta (str | list): Axes metadata from the dataset.

    Returns:
        str: A human-readable summary of axis types.
    """
    if not isinstance(axes_meta, list):
        return str(axes_meta)
    elif axes_meta == "N/A":
        return axes_meta

    parts: list[str] = []
    for axis in axes_meta:
        name = getattr(axis, "name", "?")
        typ = getattr(axis, "type", "")
        unit = getattr(axis, "unit", "")
        if unit:
            parts.append(f"{name} ({typ}, {unit})")
        elif typ:
            parts.append(f"{name} ({typ})")
        else:
            parts.append(name)

    return ", ".join(parts)

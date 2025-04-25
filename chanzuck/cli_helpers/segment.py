import traceback

import click

from chanzuck.utils.describe import describe_dataset


@click.command("segment")
@click.option(
    "--dataset-path",
    type=click.Path(exists=True, dir_okay=True),
    required=True,
    help="Path to the OME-Zarr dataset.",
)
@click.option(
    "--model-type",
    type=click.Choice(["cellpose", "otsu"], case_sensitive=False),
    required=False,
    help=(
        "Model type to use:\n"
        "  'cellpose' - Use Cellpose nuclei model\n"
        "  'otsu'     - Otsu threshold the channel if your signal is clean enough\n"
    ),
)
@click.option(
    "--channel-index",
    type=int,
    required=False,
    help="Index of the channel to segment. If not provided, you will be prompted.",
)
@click.option(
    "--gpu/--no-gpu",
    default=True,
    show_default=True,
    help="Use GPU for segmentation if available (Cellpose only).",
)
def segment(dataset_path, model_type, channel_index, gpu):
    """
    Segment 3D time-lapse OME-Zarr datasets and track cells over time.
    Will prompt for channel and model if not provided.
    """
    from chanzuck.segment.nuclei_segmentation import (
        segment_and_track_3d_over_time,
    )

    try:
        # Get metadata and extract channels
        metadata = describe_dataset(dataset_path)
        example_pos = next(iter(metadata["Wells"].values()))
        example_meta = next(iter(example_pos.values()))
        channel_names = example_meta.get("channels", [])

        if not isinstance(channel_names, list) or not channel_names:
            raise ValueError(
                "Could not extract channel names from dataset metadata."
            )

        # Show available channels and prompt if not provided
        if channel_index is None:
            click.echo("📡 Available channels:")
            for i, name in enumerate(channel_names):
                click.echo(f"  [{i}] {name}")
            channel_index = click.prompt(
                "🔍 Which channel contains nuclei?", type=int, default=0
            )

        if not (0 <= channel_index < len(channel_names)):
            raise click.BadParameter("Invalid channel index selected.")

        # Prompt for model if not provided
        if model_type is None:
            model_type = click.prompt(
                "🧠 Select model type",
                type=click.Choice(["cellpose", "otsu"], case_sensitive=False),
                default="cellpose",
            )

        segment_and_track_3d_over_time(
            zarr_path=dataset_path,
            channel_index=channel_index,
            model_type=model_type,
            use_gpu=gpu,
        )
        click.secho("✅ Segmentation complete!", fg="green")

    except Exception as e:
        traceback.print_exc()
        click.secho(f"❌ Error: {e}", fg="red")

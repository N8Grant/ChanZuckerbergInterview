import json
import traceback
from pathlib import Path

import click

from chanzuck.utils.describe import describe_dataset, format_pretty_output


@click.group()
def cli():
    pass


@cli.command("describe")
@click.option(
    "--dataset-path",
    required=True,
    type=click.Path(exists=True, dir_okay=True),
    help="Path to the dataset.",
)
@click.option(
    "--out-file",
    required=False,
    type=click.Path(dir_okay=False),
    help="Optional path to write metadata.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output in JSON format instead of pretty CLI format.",
)
def describe(dataset_path: str, out_file: str | None, as_json: bool):
    """Describe an OME-Zarr dataset and optionally save metadata to a file."""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists() or not dataset_path.is_dir():
        click.secho(
            f"Error: Dataset path '{dataset_path}' does not exist or is not a directory.",
            fg="red",
        )
        raise click.Abort()

    try:
        metadata = describe_dataset(dataset_path)
        output_str = (
            json.dumps(metadata, indent=2)
            if as_json
            else format_pretty_output(metadata)
        )

        click.echo(output_str)

        if out_file:
            Path(out_file).write_text(output_str)

    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        raise click.Abort() from e


@click.command("segment")
@click.option(
    "--dataset-path",
    type=click.Path(exists=True, dir_okay=True),
    required=True,
    help="Path to the OME-Zarr dataset.",
)
@click.option(
    "--model-type",
    type=click.Choice(
        ["cyto", "nuclei", "cyto2", "cyto3"], case_sensitive=False
    ),
    default="nuclei",
    show_default=True,
    help=(
        "Cellpose model type to use:\n"
        "  'nuclei' - Nucleus model\n"
        "  'cyto'   - Cytoplasm model\n"
        "  'cyto2'  - Cytoplasm model trained with user-provided images\n"
        "  'cyto3'  - Super-generalist model (recommended)\n"
    ),
)
@click.option(
    "--gpu/--no-gpu",
    default=True,
    show_default=True,
    help="Use GPU for segmentation if available.",
)
def segment(dataset_path, model_type, gpu):
    """
    Segment 3D time-lapse OME-Zarr datasets and track cells over time.
    Automatically lists available channels and asks which one to use.
    """
    from chanzuck.segment.nuclei_segmentation import (
        segment_and_track_3d_over_time,
    )

    try:
        # Parse metadata
        metadata = describe_dataset(dataset_path)

        # Pick a position to extract channel names from
        example_pos = next(iter(metadata["Wells"].values()))
        example_meta = next(iter(example_pos.values()))
        channel_names = example_meta.get("channels", [])

        if not isinstance(channel_names, list) or not channel_names:
            raise ValueError(
                "Could not extract channel names from dataset metadata."
            )

        # Ask user which channel to use
        click.echo("📡 Available channels:")
        for i, name in enumerate(channel_names):
            click.echo(f"  [{i}] {name}")

        choice = click.prompt(
            "🔍 Which channel contains nuclei?", type=int, default=0
        )
        if not (0 <= choice < len(channel_names)):
            raise click.BadParameter("Invalid channel index selected.")

        # Run segmentation
        segment_and_track_3d_over_time(
            zarr_path=dataset_path,
            channel_index=choice,
            model_type=model_type,
            use_gpu=gpu,
        )
        click.secho("✅ Segmentation complete!", fg="green")

    except Exception as e:
        traceback.print_exc()
        click.secho(f"❌ Error: {e}", fg="red")


## Add commands
cli.add_command(segment)
cli.add_command(describe)

if __name__ == "__main__":
    cli()

    pass

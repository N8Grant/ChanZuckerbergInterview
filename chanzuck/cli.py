import json
from pathlib import Path

import click

from .utils.describe import describe_dataset, format_pretty_output


@click.group()
def cli():
    pass


@cli.command()
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


if __name__ == "__main__":
    cli()

    pass

import click

from chanzuck.cli_helpers.describe import describe
from chanzuck.cli_helpers.segment import segment
from chanzuck.cli_helpers.stats import generate_stats
from chanzuck.cli_helpers.visualize import plot_stats, view


@click.group()
def cli():
    pass


## Add commands
cli.add_command(segment)
cli.add_command(describe)
cli.add_command(view)
cli.add_command(plot_stats)
cli.add_command(generate_stats)

if __name__ == "__main__":
    cli()

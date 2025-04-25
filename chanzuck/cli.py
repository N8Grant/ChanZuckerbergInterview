import click

from chanzuck.cli_helpers.describe import describe
from chanzuck.cli_helpers.segment import segment
from chanzuck.cli_helpers.visualize import view


@click.group()
def cli():
    pass


## Add commands
cli.add_command(segment)
cli.add_command(describe)
cli.add_command(view)

if __name__ == "__main__":
    cli()

    pass

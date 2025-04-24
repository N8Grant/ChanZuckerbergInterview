from click.testing import CliRunner

from chanzuck.cli import describe


def test_describe_valid_mocked_dataset(mocker):
    """Test CLI describe command using mocked dataset."""
    runner = CliRunner()

    # Mock image + dataset (same as in earlier test)
    mock_image = mocker.MagicMock(
        shape=(14, 3, 10, 800, 1100),
        chunks=(1, 1, 10, 800, 1100),
        dtype="float32",
        path="C/2/000001/0",
        attrs={"axes": ["T", "C", "Z", "Y", "X"]},
    )
    mock_position = mocker.MagicMock(
        data=mock_image, channel_names=["DAPI", "GFP"]
    )
    mock_well = mocker.MagicMock(positions=lambda: [("000001", mock_position)])
    mock_dataset = mocker.MagicMock(
        wells=lambda: [("C/2", mock_well)], axes=["T", "C", "Z", "Y", "X"]
    )

    # Patch the loader
    mocker.patch(
        "chanzuck.utils.describe.open_ome_zarr", return_value=mock_dataset
    )

    # Run the CLI command
    result = runner.invoke(describe, ["--dataset-path", ".", "--json"])

    assert result.exit_code == 0
    assert '"shape"' in result.output
    assert "14" in result.output
    assert "3" in result.output
    assert "10" in result.output
    assert "800" in result.output
    assert "1100" in result.output


def test_describe_invalid_path():
    """Test CLI describe with non-existent dataset path."""
    runner = CliRunner()
    result = runner.invoke(describe, ["--dataset-path", "nonexistent"])

    assert result.exit_code != 0
    assert "Path 'nonexistent' does not exist" in result.output

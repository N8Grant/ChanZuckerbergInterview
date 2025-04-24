from pathlib import Path
from types import SimpleNamespace

import pytest

from chanzuck.utils.describe import (
    describe_dataset,
    format_axes,
    format_pretty_output,
)


class TestDescribeDataset:

    def test_valid_plate_dataset_with_mock(self, mocker):
        """Simulate parsing a valid plate dataset using a mocked loader."""

        # Mock image array
        mock_image = SimpleNamespace(
            shape=(14, 3, 10, 800, 1100),
            chunks=(1, 1, 10, 800, 1100),
            dtype="float32",
            path="A1/000000/0",
            attrs={"axes": ["T", "C", "Z", "Y", "X"]},
        )

        # Mock position
        mock_position = SimpleNamespace(
            data=mock_image, channel_names=["DAPI", "GFP"]
        )

        # Mock well
        mock_well = SimpleNamespace(
            positions=lambda: [("000000", mock_position)]
        )

        # Mock plate-level dataset
        mock_dataset = SimpleNamespace(
            wells=lambda: [("A1", mock_well)], axes=["T", "C", "Z", "Y", "X"]
        )

        # Patch the actual iohub call
        mocker.patch(
            "chanzuck.utils.describe.open_ome_zarr", return_value=mock_dataset
        )

        # Run the describe
        metadata = describe_dataset("fake.zarr")

        assert metadata["Plate Format"]
        assert "A1" in metadata["Wells"]
        assert "000000" in metadata["Wells"]["A1"]

    def test_invalid_path_raises(self):
        """Test that an invalid path raises an error."""
        with pytest.raises(ValueError):
            describe_dataset(Path("this/path/does/not/exist.zarr"))

    def test_non_plate_dataset_raises(self, tmp_path):
        """Test that a non-plate OME-Zarr raises an informative error."""
        non_plate_path = tmp_path / "fake.zarr"
        non_plate_path.mkdir()
        # Fake minimal Zarr structure without 'wells'
        (non_plate_path / ".zattrs").write_text("{}")

        with pytest.raises(ValueError) as excinfo:
            describe_dataset(non_plate_path)

        assert "Could not parse metadata" in str(excinfo.value)

    def test_pretty_output_has_lines(self, mock_plate_metadata):
        """Test pretty output formatting contains expected lines."""
        output = format_pretty_output(mock_plate_metadata)
        assert "📦 Dataset Type:" in output
        assert "Well:" in output
        assert "Position:" in output

    def test_format_axes_list(self):
        """Test format_axes handles structured axis objects."""
        mock_axes = [
            SimpleNamespace(name="T", type="time", unit="second"),
            SimpleNamespace(name="C", type="channel"),
            SimpleNamespace(name="Z", type="space", unit="micrometer"),
        ]
        result = format_axes(mock_axes)
        assert "T (time, second)" in result
        assert "C (channel)" in result
        assert "Z (space, micrometer)" in result

    def test_format_axes_fallbacks(self):
        """Test format_axes fallback when given a string or N/A."""
        assert format_axes("N/A") == "N/A"
        assert format_axes("just_a_string") == "just_a_string"

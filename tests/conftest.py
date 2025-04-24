import pytest


@pytest.fixture
def mock_plate_metadata():
    return {
        "dataset_type": "iohub.ngff.nodes.Plate",
        "Plate Format": True,
        "Wells": {
            "A1": {
                "000000": {
                    "shape": (14, 3, 10, 800, 1100),
                    "chunks": (1, 1, 10, 800, 1100),
                    "dtype": "float32",
                    "channels": ["DAPI", "GFP"],
                    "axes": "T, C, Z, Y, X",
                    "path": "A1/000000/0",
                }
            }
        },
    }


@pytest.fixture
def mock_plate_dataset(tmp_path):
    # Simulate a basic OME-Zarr plate (you can replace this with a real one)
    plate_path = tmp_path / "mock_plate.zarr"
    plate_path.mkdir()
    # Simulate .zattrs or fake open_ome_zarr patch if necessary
    return plate_path

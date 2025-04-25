# Chan Zuckerberg Biohub Coding Assessment
This repo will contain all of the code related to the problems associated with the PDF given.


## Goals
Complete each of the tasks:

### Task 1
- Convert the OME-TIFF to OME-Zarr using iohub ✅
- Inspect the converted OME-Zarr and retrieve key metadata using iohub (i.e, array
shapes, scale, channel names, chunk sizes, etc). Tell us what method you used and
save them in a text file. ✅
- Write a CLI to parse and pretty print metadata information of an OME-Zarr store. ✅
- Implement a PyTorch DataLoader that uses iohub and the OME-Zarr (Dataset 2) to
read data. ✅
- Profile the time it takes to read the data and to run inference with a pre-trained model
(e.g., those provided with torchivision). ✅

### Task 2
- Leverages iohub to handle the OME-Zarr 5D dataset (T, C, Z, Y, X) ✅
- Performs some basic image analysis ✅
- Segment the cells’ nuclei from your channel of choice and store the
segmentations into an OME-Zarr store using iohub as Zarr Arrays. ✅
- Computes at least 5 metrics or algorithms to characterize and detect the infection
dynamics. (e.g. intensity, segmentation, etc). Can you find the infected cells?
- Generates visualizations of image data and the image analysis results (i.e, matplotlib,
napari, etc).
- Save the visualizations (e.g as .png, .mp4,.jpg, etc). You will share this
output with us.
- Write a script that showcases the use of your API.
- Write a README.md that describes this pipeline so we can reproduce it.
- Speed up your functions! Parallelize methods in your image processing library. Which
workloads can benefit from GPU compute?
- Profile the performance of the parallelized implementation. How does it scale with
different hardware resources?

### Task 3
- Implement a command line interface (CLI) on top of this to interact with your image
management and analysis API. ✅
- Create a spec-conformant Python package that can be installed with frontends like pip
and uv. Make sure that the dependencies resolve correctly with a fresh virtual
environment. ✅
- Implement your CLI so that it shows a progress bar for long processing steps. ✅
- Don't forget to add usage instructions in the README.md.
- Create a Docker container that uses your CLI to process data.
- Build your image, test it, and share it with us.
- Don't forget to add instructions in the README.md if you do.


## For Devs

### 1. Clone Repo
```bash
git clone https://github.com/N8Grant/ChanZuckerbergInterview.git
cd ChanZuckerbergInterview
```

### 2. Install Package
```bash
pip install -e .[dev]
pre-commit install
```

### 3. Run Tests
```bash
pytest
```

Now you can work on the package in dev mode as well as have pre-commits all hooked up for when you want to push changes.

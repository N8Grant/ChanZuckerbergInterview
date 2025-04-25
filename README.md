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
dynamics. (e.g. intensity, segmentation, etc). Can you find the infected cells? ✅
- Generates visualizations of image data and the image analysis results (i.e, matplotlib,
napari, etc).✅
- Save the visualizations (e.g as .png, .mp4,.jpg, etc). You will share this
output with us. ✅
- Write a script that showcases the use of your API.
- Write a README.md that describes this pipeline so we can reproduce it.
- Speed up your functions! Parallelize methods in your image processing library. Which
workloads can benefit from GPU compute? ✅
- Profile the performance of the parallelized implementation. How does it scale with
different hardware resources? ✅

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

## Installation

### 1. Clone Repo
```bash
git clone https://github.com/N8Grant/ChanZuckerbergInterview.git
cd ChanZuckerbergInterview
```

### 2. Install Package
```bash
pip install .
```
Now your ready to use the tool so long as your python scripts folder is on your PATH variable.

### 3. Install Package For Devs
To install in developer mode make sure to include both the pip flag as well as the dev tag for the installation of all of the
required precommit libraries as well.

```bash
pip install -e .[dev]
pre-commit install
```

### 4. Run Tests
```bash
pytest
```

Now you can work on the package in dev mode as well as have pre-commits all hooked up for when you want to push changes.


## Examples

### Viewing A Dataset
To view a dataset using napari you can use the command

```bash
chanzuck view --dataset-path "<path_to_zarr>"
```
A napari window should pop up witha all of the wells and posisions displayed for you to look through.

*Note:* This command by default will try to display any existing segmenation labels on top of the image.
If this is not desired however it can be toggled off with the --no-show-segmentations flag.

### Describing a Dataset
To easily check the metadata within a dataset you can use the describe command as follows:

```bash
chanzuck view --dataset-path "<path_to_zarr>"
```

This command also has the option to wtire metadat to a file instead of just printing out in the cli.
You can do this by using the --out-file flag with the desired path following it.
If you would like just a regular json format instead of the well formatted cli output then provide the --json flag.

Options:
  --dataset-path PATH  Path to the dataset.  [required]
  --out-file FILE      Optional path to write metadata.
  --json               Output in JSON format instead of pretty CLI format.
  --help               Show this message and exit.

### Segment a Dataset
Segmenting nuclei out of a dataset is easy with chanzuck. All you have to do is run the command below and it will walk you through
setting up your segmentation routine.

```bash
chanzuck segment --dataset-path "<path_to_zarr>"
```

This will run segmentation and labeling over the given channel and try to temporally connect the labels so the time information is more useful.

#### Additional Parameters
Once ran it will prompt you for the index of the nuclei stain, dont worry you dont have to remember that though, it will
pull the information from the dataset so you can easily select it. However if you do know it ahead of time then you can add
the --channel-index flag to the command with an integer indicating teh index of the desired segmentation channel.

There are also two models to chose from using the --model-type command:
1. "cellpose": Open source cell detection model that can be very slow if you dont include the --gpu flag so be sure to add that in the command as well.
2. "otsu": Quick thresholding if you have a clean staining over the object of interest

#### Results
Use the view command on your dataset to see the results!

Example:
![Cell Segmentation](./images/cells_segmentsion_results.png)

### Segment a Dataset

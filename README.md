# Lung Segmentation and Dataset Preparation

This repository contains the data preparation pipeline for merging the NIH and CheXpert chest X-ray datasets, standardizing them to 384x384 resolution, and extracting lung Region of Interest (ROI) using a U-Net architecture.

## Repository Structure

- `A_prepare_data.py`: Merges raw NIH and CheXpert data, extracts target classes, resizes images to 384x384, and outputs a single master dataset. Configurable for 4 or 5 classes.
- `B_preprocess.py`: Processes the dataset (e.g., from Kaggle), isolates lung ROIs using U-Net, and balances the class distribution dynamically by targeting the maximum class count.
- `unet.py`: The U-Net architecture implementation.
- `notebooks/kaggle_pipeline.ipynb`: An end-to-to Jupyter notebook designed to run on Kaggle, which automatically downloads datasets, weights, and executes the preprocessing pipeline.
- `metadata/`: Contains reference CSV label files.
- `weights/`: Directory where the pre-trained U-Net weights should be placed (or downloaded into).

## Setup & Installation

```bash
pip install -r requirements.txt
```

## How to use

### Local execution (from scratch)

1. Place `raw_nih` and `raw_chexpert` in the root folder.
2. Run data preparation to merge and resize:
   ```bash
   python A_prepare_data.py
   ```
3. Run U-Net ROI extraction (will auto-download U-Net weights if `--download-weights` flag is passed):
   ```bash
   python B_preprocess.py --download-weights
   ```

### Kaggle/Cloud execution

The simplest way to process the dataset on a rented machine (like Vast.ai) or Kaggle is using the provided notebook:

1. Open `notebooks/kaggle_pipeline.ipynb` on Kaggle or a Jupyter environment.
2. The notebook will automatically:
   - Download the pre-merged dataset: `amhyou/thesis-dataset-5classes`
   - Download the U-Net weights from: `nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset`
   - Run `B_preprocess.py` to isolate the ROIs.

## Important Note regarding Git

- **Do NOT commit the `.hdf5` weights or large `.csv` files (except the reference ones in `metadata/`) or images to Git.** They are properly ignored in the `.gitignore`.

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from unet import get_unet
import argparse

# ─── EXPERIMENT CONFIGURATION ─────────────────────────────
NUM_CLASSES = 5   # Set to 4 or 5

if NUM_CLASSES == 5:
    TARGET_CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Normal', 'Pneumonia']
    CSV_NAME = "master_384_5class_labels.csv"
else:
    TARGET_CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Normal']
    CSV_NAME = "master_384_4class_labels.csv"

# Check if running in Kaggle environment
IS_KAGGLE = os.path.exists('/kaggle/input')

if IS_KAGGLE:
    DATA_DIR = f"/kaggle/input/thesis-dataset-5classes"
    RAW_IMAGES = os.path.join(DATA_DIR, "images")
    INPUT_CSV = os.path.join(DATA_DIR, CSV_NAME)
    
    OUTPUT_BASE = "/kaggle/working"
    ROI_OUTPUT = os.path.join(OUTPUT_BASE, "roi")
    METADATA_DIR = os.path.join(OUTPUT_BASE, "metadata")
else:
    # Local fallback
    DATA_DIR = f"data_384_{NUM_CLASSES}class"
    RAW_IMAGES = os.path.join(DATA_DIR, "images")
    INPUT_CSV = os.path.join(DATA_DIR, CSV_NAME)
    if not os.path.exists(INPUT_CSV):
        # Fallback to metadata directory
        INPUT_CSV = os.path.join("metadata", CSV_NAME)
        
    OUTPUT_BASE = "."
    ROI_OUTPUT = os.path.join(OUTPUT_BASE, f"data_roi_{NUM_CLASSES}class")
    METADATA_DIR = os.path.join(OUTPUT_BASE, "metadata")

FINAL_METADATA = os.path.join(METADATA_DIR, f"DATA_ROI_{NUM_CLASSES}CLASS.csv")
WEIGHTS = "weights/cxr_reg_weights.best.hdf5"
IMG_SIZE = 384  # Updated for merged Kaggle dataset

# ──────────────────────────────────────────────────────────

class LungProcessor:
    def __init__(self, weights_path, img_size=384):
        self.img_size = img_size

        # Build the U-Net architecture with input shape (img_size, img_size, 1) for grayscale images
        self.model = get_unet((img_size, img_size, 1))

        # Load the pre-trained weights into the model
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found at {weights_path}. Please download them first.")
        self.model.load_weights(weights_path)

        # Initialize CLAHE (Contrast Limited Adaptive Histogram Equalization) for image enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def process(self, img_path):
        # Step 1: Load and preprocess the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None  # Return None if image loading fails
            
        # Resize just in case, though they should already be 384x384
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4)
            
        enhanced = self.clahe.apply(img)  # Apply CLAHE for contrast enhancement

        # Step 2: Prepare image for model prediction
        img_in = enhanced.astype(np.float32) / 255.0  # Normalize to [0, 1]
        img_in = np.expand_dims(img_in, axis=(0, -1))  # Add batch and channel dimensions

        # Step 3: Predict segmentation mask using the U-Net model
        mask = self.model.predict(img_in, verbose=0)[0]  # Get prediction, remove batch dim
        mask_binary = (mask > 0.5).astype(np.uint8)  # Threshold to create binary mask

        # Step 4: Apply mask to isolate lung regions
        return cv2.bitwise_and(enhanced, enhanced, mask=mask_binary)


def download_weights_if_needed():
    if not os.path.exists(WEIGHTS):
        print("U-Net weights not found locally. Attempting to download via Kaggle API...")
        os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
        # We download from the specific kernel output mentioned
        os.system("kaggle kernels output nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset -p ./weights/")
        
        # In case the file was named slightly differently or placed in a subfolder,
        # we check the weights directory
        files = os.listdir("./weights/")
        hdf5_files = [f for f in files if f.endswith('.hdf5') or f.endswith('.h5')]
        if hdf5_files and hdf5_files[0] != os.path.basename(WEIGHTS):
            os.rename(os.path.join("./weights", hdf5_files[0]), WEIGHTS)
            
        if os.path.exists(WEIGHTS):
            print("Successfully downloaded U-Net weights!")
        else:
            print(f"Warning: Could not download or find {WEIGHTS}")

def run_targeted_pipeline():
    parser = argparse.ArgumentParser(description="Preprocess Chest X-Rays with U-Net")
    parser.add_argument('--download-weights', action='store_true', help="Download U-Net weights from Kaggle")
    args = parser.parse_args()

    if args.download_weights:
        download_weights_if_needed()

    if not os.path.exists(ROI_OUTPUT): os.makedirs(ROI_OUTPUT, exist_ok=True)
    if not os.path.exists(METADATA_DIR): os.makedirs(METADATA_DIR, exist_ok=True)

    print(f"Step 1: Loading {NUM_CLASSES}-class metadata...")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Please run A_prepare_data.py first.")
        return
        
    df = pd.read_csv(INPUT_CSV)
    
    # We already have 0/1 columns from A_prepare_data.py for TARGET_CLASSES
    # Keep only samples that have at least one target class (they all should anyway)
    target_df = df[df[TARGET_CLASSES].sum(axis=1) > 0].copy()

    print("\nStep 2: Calculating dynamic class balancing...")
    # Find the count of each class
    class_counts = {cls: target_df[cls].sum() for cls in TARGET_CLASSES}
    
    # Target maximum samples per class to ensure minority classes are oversampled 
    # to match the majority class (usually Normal or Atelectasis)
    target_samples_per_class = max(class_counts.values())
    
    print(f"Class counts: {class_counts}")
    print(f"Targeting {target_samples_per_class} samples per class (balancing to maximum).")

    balanced_list = []
    unique_images_to_process = set()

    for cls in TARGET_CLASSES:
        subset = target_df[target_df[cls] == 1]
        count = len(subset)
        
        if count == 0:
            print(f" - {cls}: 0 base images. Skipping.")
            continue
            
        multiplier = max(1, target_samples_per_class // count)
        
        print(f" - {cls}: {count} base images. Augmentation factor: {multiplier}x")
        
        for i in range(multiplier):
            temp = subset.copy()
            temp['aug_instance'] = i # Tracking ID for Grad-CAM
            balanced_list.append(temp)
            unique_images_to_process.update(subset['Image_ID'].tolist())
            
        # Add remainder if we want exactly max_samples (optional, simplified here by just using multiplier)

    print(f"\nStep 3: ROI Isolation for {len(unique_images_to_process)} unique images (Skipping existing)...")
    try:
        processor = LungProcessor(WEIGHTS, img_size=IMG_SIZE)
    except FileNotFoundError as e:
        print(e)
        print("Run with --download-weights to fetch them automatically.")
        return
    
    for img_name in tqdm(list(unique_images_to_process)):
        in_path = os.path.join(RAW_IMAGES, img_name)
        out_path = os.path.join(ROI_OUTPUT, img_name)
        
        # Safety Check: skip if file already exists from previous run 
        if os.path.exists(out_path): continue
        
        # Some paths from Chexpert might be deeply nested in raw format, but our A_ script flattens them
        if not os.path.exists(in_path):
            continue
            
        roi_img = processor.process(in_path)
        if roi_img is not None:
            cv2.imwrite(out_path, roi_img)

    final_df = pd.concat(balanced_list)
    final_df.to_csv(FINAL_METADATA, index=False)
    print(f"\nPipeline Complete! Balanced {NUM_CLASSES}-class metadata saved to {FINAL_METADATA}")

if __name__ == "__main__":
    run_targeted_pipeline()
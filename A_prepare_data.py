import os
import cv2
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# ─── EXPERIMENT CONFIGURATION ─────────────────────────────
NUM_CLASSES = 5   # Set to 4 (Atelectasis, Cardiomegaly, Effusion, Normal)
                  # or  5 (+ Pneumonia)

TARGET_SIZE = (384, 384)
NUM_WORKERS = os.cpu_count() or 4

if NUM_CLASSES == 5:
    TARGET_CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Normal', 'Pneumonia']
    OUTPUT_DIR = "data_384_5class"
    MASTER_CSV = "master_384_5class_labels.csv"
else:
    TARGET_CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Normal']
    OUTPUT_DIR = "data_384_4class"
    MASTER_CSV = "master_384_4class_labels.csv"

OUTPUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")

# Paths to the root downloaded folders
RAW_NIH_DIR = "raw_nih"
RAW_CHEXPERT_DIR = "raw_chexpert/extracted"
# ──────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────

def build_image_index(root_dir):
    """Recursively scans a directory and maps filename -> full absolute path."""
    print(f"Scanning {root_dir} to build a robust file index... (This takes a few seconds)")
    index = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                index[f] = os.path.join(dirpath, f)
    print(f"  -> Found {len(index)} total images in {root_dir}.")
    return index

def find_csv(root_dir, expected_name):
    """Recursively searches for a CSV file in the directory tree."""
    for dirpath, _, filenames in os.walk(root_dir):
        if expected_name in filenames:
            return os.path.join(dirpath, expected_name)
    return None

def find_batch_dirs(base_dir):
    """Finds all CheXpert batch directories inside the extracted folder."""
    batch_dirs = []
    if os.path.exists(base_dir):
        for name in os.listdir(base_dir):
            full_path = os.path.join(base_dir, name)
            if os.path.isdir(full_path) and 'CheXpert' in name:
                batch_dirs.append(full_path)
    print(f"  -> Found {len(batch_dirs)} CheXpert batch folder(s): {[os.path.basename(d) for d in batch_dirs]}")
    return batch_dirs

def resolve_chexpert_path(batch_dirs, csv_path_col):
    """
    Strips 'CheXpert-v1.0/train/' prefix from CSV path and searches all batch dirs.
    """
    parts = csv_path_col.replace('\\', '/').split('/')
    relative_path = os.path.join(*parts[2:])
    for batch_dir in batch_dirs:
        candidate = os.path.join(batch_dir, relative_path)
        if os.path.exists(candidate):
            return candidate
    return None

def process_image(task):
    """Resizes a single image to TARGET_SIZE and saves it."""
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
    src_path, dst_path = task
    try:
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, False
        img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
        ext = os.path.splitext(dst_path)[1].lower()
        if ext in ('.jpg', '.jpeg'):
            cv2.imwrite(dst_path, img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            cv2.imwrite(dst_path, img_resized, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        return os.path.basename(dst_path), True
    except Exception:
        return None, False

def execute_tasks(tasks, records, dataset_name):
    print(f"Resizing {len(tasks)} images for {dataset_name} using {NUM_WORKERS} CPU cores...")
    successful_ids = set()
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(process_image, tasks), total=len(tasks)))
    for img_id, success in results:
        if success:
            successful_ids.add(img_id)
    final_records = [r for r in records if r['Image_ID'] in successful_ids]
    print(f"Successfully processed {len(final_records)} images.")
    return pd.DataFrame(final_records)

# ─────────────────────────────────────────────────────────
# NIH PROCESSING
# ─────────────────────────────────────────────────────────

def prepare_nih():
    print(f"\n--- Processing NIH Dataset ({NUM_CLASSES}-class) ---")
    if not os.path.exists(RAW_NIH_DIR):
        print(f"Skipping NIH: {RAW_NIH_DIR} not found.")
        return pd.DataFrame()
        
    csv_path = find_csv(RAW_NIH_DIR, "Data_Entry_2017.csv")
    if not csv_path:
        print("Skipping NIH: Data_Entry_2017.csv not found anywhere inside raw_nih/.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    img_index = build_image_index(RAW_NIH_DIR)

    processed_records = []
    tasks = []

    for _, row in df.iterrows():
        labels_str = row['Finding Labels']
        img_id = row['Image Index']

        labels = labels_str.split('|')

        # Map 'No Finding' -> 'Normal'
        if 'No Finding' in labels:
            labels.remove('No Finding')
            labels.append('Normal')

        # Only keep images with at least one of our target classes
        has_target = any(c in TARGET_CLASSES for c in labels)
        if not has_target:
            continue

        record = {
            'Image_ID': f"nih_{img_id}",
            'Dataset': 'NIH',
            'Atelectasis':  1 if 'Atelectasis'  in labels else 0,
            'Cardiomegaly': 1 if 'Cardiomegaly' in labels else 0,
            'Effusion':     1 if 'Effusion'      in labels else 0,
            'Normal':       1 if 'Normal'        in labels else 0,
        }
        
        if NUM_CLASSES == 5:
            record['Pneumonia'] = 1 if 'Pneumonia' in labels else 0
            
        processed_records.append(record)

        if img_id in img_index:
            src_path = img_index[img_id]
            dst_path = os.path.join(OUTPUT_IMG_DIR, f"nih_{img_id}")
            tasks.append((src_path, dst_path))

    return execute_tasks(tasks, processed_records, "NIH")

# ─────────────────────────────────────────────────────────
# CHEXPERT PROCESSING
# ─────────────────────────────────────────────────────────

def prepare_chexpert():
    print(f"\n--- Processing CheXpert Dataset ({NUM_CLASSES}-class) ---")
    if not os.path.exists(RAW_CHEXPERT_DIR):
        print(f"Skipping CheXpert: {RAW_CHEXPERT_DIR} not found.")
        return pd.DataFrame()
        
    csv_path = find_csv(RAW_CHEXPERT_DIR, "train.csv")
    if not csv_path:
        print("Skipping CheXpert: train.csv not found anywhere inside raw_chexpert/.")
        return pd.DataFrame()

    batch_dirs = find_batch_dirs(RAW_CHEXPERT_DIR)
    if not batch_dirs:
        print("Skipping CheXpert: Cannot find any 'CheXpert' batch folders after extraction.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df = df.fillna(0)

    processed_records = []
    tasks = []
    skipped_missing = 0

    for _, row in df.iterrows():
        raw_at = row.get('Atelectasis', 0)
        raw_cd = row.get('Cardiomegaly', 0)
        raw_ef = row.get('Pleural Effusion', 0)
        raw_nm = row.get('No Finding', 0)

        # STRICT RULE: Drop image if ANY target class is Uncertain (-1)
        if raw_at == -1 or raw_cd == -1 or raw_ef == -1 or raw_nm == -1:
            continue
            
        if NUM_CLASSES == 5:
            raw_pn = row.get('Pneumonia', 0)
            if raw_pn == -1:
                continue

        at = 1 if raw_at == 1 else 0
        cd = 1 if raw_cd == 1 else 0
        ef = 1 if raw_ef == 1 else 0
        nm = 1 if raw_nm == 1 else 0
        
        has_target = (at == 1 or cd == 1 or ef == 1 or nm == 1)
        if NUM_CLASSES == 5:
            pn = 1 if raw_pn == 1 else 0
            has_target = has_target or (pn == 1)

        if has_target:
            # Only keep Frontal views (AP/PA), ignore Lateral — case-insensitive!
            if 'frontal' not in str(row['Path']).lower():
                continue

            # Multi-batch path resolution
            src_path = resolve_chexpert_path(batch_dirs, row['Path'])
            if src_path is None:
                skipped_missing += 1
                continue

            safe_filename = 'chexpert_' + row['Path'].replace('CheXpert-v1.0/', '').replace('/', '_')

            record = {
                'Image_ID': safe_filename,
                'Dataset': 'CheXpert',
                'Atelectasis':  at,
                'Cardiomegaly': cd,
                'Effusion':     ef,
                'Normal':       nm,
            }
            if NUM_CLASSES == 5:
                record['Pneumonia'] = pn
                
            processed_records.append(record)
            dst_path = os.path.join(OUTPUT_IMG_DIR, safe_filename)
            tasks.append((src_path, dst_path))

    if skipped_missing > 0:
        print(f"  [INFO] Skipped {skipped_missing} images not found on disk.")

    return execute_tasks(tasks, processed_records, "CheXpert")

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

    # CheXpert first for quick verification
    chexpert_df = prepare_chexpert()
    nih_df = prepare_nih()

    if chexpert_df.empty and nih_df.empty:
        print("\nNo data processed. Ensure raw data folders exist and are populated.")
        return

    master_df = pd.concat([nih_df, chexpert_df], ignore_index=True)

    master_csv_path = os.path.join(OUTPUT_DIR, MASTER_CSV)
    master_df.to_csv(master_csv_path, index=False)

    print("\n==================================================")
    print(f"✅ {NUM_CLASSES}-Class Data Preparation Complete!")
    print(f"Total Images: {len(master_df)}")
    print(f"Class Counts:")
    for cls in TARGET_CLASSES:
        print(f" - {cls}: {master_df[cls].sum()}")
    print(f"Master CSV saved to: {master_csv_path}")
    print("==================================================")
    print(f"\nNext step: zip -r {OUTPUT_DIR}.zip {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()

from sklearn.preprocessing import StandardScaler
from pandas import read_csv
import numpy as np
import nibabel as nib
import os
def load_lesions_and_behaviors(lesion_folder, csv_file, max_score, do_regress_out_lesion_volume=False):
    """
    Load lesion files, behavioral data, and covariates from CSV, and compute lesion volumes.
    """
    print("Loading behavioral data and lesion files...")
    df = read_csv(csv_file)

    # Check for required columns
    required_columns = ['filename', 'behavior']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"CSV file must contain '{col}' column.")

    lesion_files = [os.path.join(lesion_folder, f) for f in df['filename']]
    behaviors = df['behavior'].values

    print('\nBehavior before:\n', behaviors)

    behaviors = behaviors / max_score

    print('\nBehavior after:\n', behaviors)

    # Compute lesion volumes

    print("\nComputing lesion volumes...")
    lesion_volumes = []
    for file in lesion_files:
        lesion_img = nib.load(file)
        lesion_data = lesion_img.get_fdata()
        voxel_volume = np.prod(lesion_img.header.get_zooms())
        lesion_volumes.append(np.sum(lesion_data > 0) * voxel_volume)
    lesion_volumes = np.array(lesion_volumes).reshape(-1, 1)

    # Load additional covariates
    covariates = df.iloc[:, 2:].values  # Assuming additional covariates start from the 3rd column
    if covariates.shape[1] > 0:
        print(f"Loaded {covariates.shape[1]} additional covariates",end='')
        # Combine lesion volumes with additional covariates
        if do_regress_out_lesion_volume:
            print(" and lesion volume as covariate")
            covariates = np.hstack([lesion_volumes, covariates])
        else:
            print("\nLesion volume not regressed out as covariate")
            covariates = np.hstack([covariates])
        # Z-transform covariates
        scaler = StandardScaler()
        covariates = scaler.fit_transform(covariates)
    else:
        print("No additional covariates found in the CSV.")
        covariates = None

    return lesion_files, behaviors, covariates, lesion_volumes
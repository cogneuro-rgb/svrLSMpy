import nibabel as nib
import numpy as np
from sklearn.preprocessing import normalize
from nilearn import masking

def filter_voxels_by_patient_count(lesion_files, min_patient_count, normalize_vector, output_folder):
    """
    Filter voxels by the number of patients they are involved in.
    """
    no_of_patients = len(lesion_files)
    if not isinstance(min_patient_count, int):
        if isinstance(min_patient_count, str):
            if min_patient_count.endswith('%'):
                min_patient_count = min_patient_count.strip('%')
            min_patient_count = float(min_patient_count)
            print(f"{min_patient_count}% of {no_of_patients} patients = ",end='')
            min_patient_count = no_of_patients*min_patient_count/100
            print(min_patient_count)
            min_patient_count = int(min_patient_count)
            print("Thus,")


    if min_patient_count>0:
        print(f"Filtering voxels by patient count: {min_patient_count}/{no_of_patients}")
    else:
        print("Filtering is not done")

    lesion_imgs = [nib.load(f) for f in lesion_files]
    masker = masking.compute_brain_mask(lesion_imgs[0])
    lesion_data = [masking.apply_mask(img, masker) for img in lesion_imgs]

    # Stack the vectorized lesion data
    lesion_data_stack = np.vstack(lesion_data)

    sum_of_vectors = np.sum(lesion_data_stack, axis=0)
    sum_of_vectors = sum_of_vectors.astype(np.int32)  # Ensure integer type
    sum_of_voxel_mni = masking.unmask(sum_of_vectors, masker)
    sum_of_vectors_path = output_folder / "lesion_overlap.nii.gz"
    nib.save(sum_of_voxel_mni, sum_of_vectors_path)

    # Count the number of patients each voxel is involved in
    voxel_patient_count = np.sum(lesion_data_stack > 0, axis=0)

    # Filter voxels
    lesion_data_prepared = np.copy(lesion_data_stack)
    lesion_data_prepared[:, voxel_patient_count < min_patient_count] = 0

    sum_of_vectors_filtered = np.sum(lesion_data_prepared, axis=0)
    sum_of_vectors_filtered = sum_of_vectors_filtered.astype(np.int32)  # Ensure integer type
    sum_of_voxel_mni_filtered = masking.unmask(sum_of_vectors_filtered, masker)
    sum_of_vectors_filtered_path = output_folder / "lesion_overlap_filtered.nii.gz"
    nib.save(sum_of_voxel_mni_filtered, sum_of_vectors_filtered_path)

    if normalize_vector:
        # Normalize the data to have unit norm
        lesion_data_prepared = normalize(lesion_data_prepared, norm='l2', axis=1)

    return min_patient_count,lesion_data_prepared, masker
import numpy as np
import nibabel as nib


def load_nifti(path):
    """
    Load a NIfTI file and return the image volume and affine matrix.
    """
    nii = nib.load(str(path))
    volume = nii.get_fdata()
    affine = nii.affine
    return volume, affine


def clip_hu(volume, hu_min=-1000, hu_max=400):
    """
    Clip CT values to a useful Hounsfield Unit range.
    """
    return np.clip(volume, hu_min, hu_max)


def normalize_to_01(volume, hu_min=-1000, hu_max=400):
    """
    Normalize CT volume values to the [0, 1] range.
    """
    volume = clip_hu(volume, hu_min, hu_max)
    volume = (volume - hu_min) / (hu_max - hu_min)
    return volume.astype(np.float32)


def get_slice(volume, index, axis=2):
    """
    Extract a 2D slice from a 3D volume.

    axis=0: sagittal
    axis=1: coronal
    axis=2: axial
    """
    if axis == 0:
        return volume[index, :, :]

    if axis == 1:
        return volume[:, index, :]

    if axis == 2:
        return volume[:, :, index]

    raise ValueError("axis must be 0, 1, or 2")


def load_and_preprocess_nifti(path, hu_min=-1000, hu_max=400):
    """
    Load a NIfTI file, clip HU values, and normalize to [0, 1].
    """
    volume, affine = load_nifti(path)
    volume = normalize_to_01(volume, hu_min, hu_max)
    return volume, affine

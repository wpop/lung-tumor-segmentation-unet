from pathlib import Path

import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

from src.config import HU_MAX, HU_MIN


class LungTumorSliceDataset(Dataset):
    """
    PyTorch Dataset for loading matching lung CT image and tumor mask volumes.
    """

    def __init__(self, image_dir: str | Path, mask_dir: str | Path) -> None:
        """
        Find and store matching .nii.gz image and mask file paths.
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        image_paths = sorted(self.image_dir.glob("*.nii.gz"))
        mask_paths = sorted(self.mask_dir.glob("*.nii.gz"))

        image_files = {path.name: path for path in image_paths}
        mask_files = {path.name: path for path in mask_paths}

        # Filenames must match so each image has the correct mask.
        if set(image_files) != set(mask_files):
            missing_masks = sorted(set(image_files) - set(mask_files))
            missing_images = sorted(set(mask_files) - set(image_files))
            raise ValueError(
                "Image and mask filenames do not match. "
                f"Missing masks: {missing_masks}. "
                f"Missing images: {missing_images}."
            )

        self.image_paths = [image_files[name] for name in sorted(image_files)]
        self.mask_paths = [mask_files[name] for name in sorted(mask_files)]

    def __len__(self) -> int:
        """
        Return the number of matched image and mask volumes.
        """
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Load and return one image volume and its matching mask volume.
        """
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        # get_fdata returns the volume data as a NumPy array.
        image_volume = nib.load(image_path).get_fdata()
        mask_volume = nib.load(mask_path).get_fdata()

        return image_volume, mask_volume


class LungTumorSliceDataset2D(Dataset):
    """
    PyTorch Dataset for loading 2D axial slices that contain tumor pixels.
    """

    def __init__(self, image_dir: str | Path, mask_dir: str | Path) -> None:
        """
        Find matching image and mask files, then index slices with positive masks.
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        image_paths = sorted(self.image_dir.glob("*.nii.gz"))
        mask_paths = sorted(self.mask_dir.glob("*.nii.gz"))

        image_files = {path.name: path for path in image_paths}
        mask_files = {path.name: path for path in mask_paths}

        # Filenames must match so each image has the correct mask.
        if set(image_files) != set(mask_files):
            missing_masks = sorted(set(image_files) - set(mask_files))
            missing_images = sorted(set(mask_files) - set(image_files))
            raise ValueError(
                "Image and mask filenames do not match. "
                f"Missing masks: {missing_masks}. "
                f"Missing images: {missing_images}."
            )

        self.slice_index: list[tuple[Path, Path, int]] = []

        for filename in sorted(image_files):
            image_path = image_files[filename]
            mask_path = mask_files[filename]

            # Load the mask once here so we can find slices that contain tumor.
            mask_volume = nib.load(mask_path).get_fdata()

            for index in range(mask_volume.shape[2]):
                mask_slice = mask_volume[:, :, index]

                if np.any(mask_slice > 0):
                    self.slice_index.append((image_path, mask_path, index))

    def __len__(self) -> int:
        """
        Return the number of tumor-containing 2D slices.
        """
        return len(self.slice_index)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int]:
        """
        Load one image and mask volume, then return one normalized axial slice.
        """
        image_path, mask_path, slice_index = self.slice_index[index]

        image_volume = nib.load(image_path).get_fdata()
        mask_volume = nib.load(mask_path).get_fdata()

        # Extract the axial slice from both volumes.
        image_slice = image_volume[:, :, slice_index]
        mask_slice = mask_volume[:, :, slice_index]

        # Clip CT Hounsfield Units to the configured window.
        image_slice = np.clip(image_slice, HU_MIN, HU_MAX)

        # Normalize the clipped CT values to [0, 1].
        image_slice = (image_slice - HU_MIN) / (HU_MAX - HU_MIN)

        image_slice = image_slice.astype(np.float32)
        mask_slice = mask_slice.astype(np.uint8)

        return {
            "image": image_slice,
            "mask": mask_slice,
            "slice_index": slice_index,
        }

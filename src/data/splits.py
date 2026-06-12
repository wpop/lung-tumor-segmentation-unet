from pathlib import Path
import random


def create_patient_split(
    image_dir: str | Path,
    mask_dir: str | Path,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    """
    Create a patient-level train/validation split for MSD Task06 Lung.

    Each patient is represented by one image file and one mask file. Image and
    mask files are matched by filename so the split keeps each image with its
    correct mask.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    if not 0 <= val_fraction <= 1:
        raise ValueError("val_fraction must be between 0 and 1.")

    image_files = {path.name: path for path in image_dir.iterdir() if path.is_file()}
    mask_files = {path.name: path for path in mask_dir.iterdir() if path.is_file()}

    # Filenames must match so every image has the correct mask.
    if set(image_files) != set(mask_files):
        missing_masks = sorted(set(image_files) - set(mask_files))
        missing_images = sorted(set(mask_files) - set(image_files))
        raise ValueError(
            "Image and mask filenames do not match. "
            f"Missing masks: {missing_masks}. "
            f"Missing images: {missing_images}."
        )

    filenames = sorted(image_files)

    # Use a local random generator so this does not affect any other code.
    rng = random.Random(seed)
    rng.shuffle(filenames)

    total_patients = len(filenames)
    val_count = int(total_patients * val_fraction)

    val_filenames = filenames[:val_count]
    train_filenames = filenames[val_count:]

    train_image_paths = [image_files[name] for name in train_filenames]
    train_mask_paths = [mask_files[name] for name in train_filenames]
    val_image_paths = [image_files[name] for name in val_filenames]
    val_mask_paths = [mask_files[name] for name in val_filenames]

    print(f"Total patients: {total_patients}")
    print(f"Train patients: {len(train_filenames)}")
    print(f"Validation patients: {len(val_filenames)}")

    return train_image_paths, train_mask_paths, val_image_paths, val_mask_paths

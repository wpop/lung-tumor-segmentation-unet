from src.config import IMAGES_TR_DIR, LABELS_TR_DIR
from src.data.dataset import LungTumorSliceDataset2D
from src.data.splits import create_patient_split


def main() -> None:
    """
    Create a patient-level split and print basic dataset information.
    """
    (
        train_image_paths,
        train_mask_paths,
        val_image_paths,
        val_mask_paths,
    ) = create_patient_split(IMAGES_TR_DIR, LABELS_TR_DIR)

    print(f"Train patient count: {len(train_image_paths)}")
    print(f"Validation patient count: {len(val_image_paths)}")

    # Build datasets from explicit patient file lists and include some negative slices.
    train_dataset = LungTumorSliceDataset2D(
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        negative_ratio=0.3,
        seed=42,
    )
    val_dataset = LungTumorSliceDataset2D(
        image_paths=val_image_paths,
        mask_paths=val_mask_paths,
        negative_ratio=0.3,
        seed=42,
    )

    print(f"Train slice count with negatives: {len(train_dataset)}")
    print(f"Validation slice count with negatives: {len(val_dataset)}")

    # Print the first filename from each split for a quick sanity check.
    if train_image_paths:
        print(f"First train filename: {train_image_paths[0].name}")

    if val_image_paths:
        print(f"First validation filename: {val_image_paths[0].name}")


if __name__ == "__main__":
    main()

from pathlib import Path

from src.config import IMAGES_TR_DIR


def main():

    image_files = sorted(IMAGES_TR_DIR.glob("*.nii.gz"))

    print(f"Training volumes: {len(image_files)}")

    print("\nFirst 10 files:")

    for path in image_files[:10]:
        print(path.name)


if __name__ == "__main__":
    main()

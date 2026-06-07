from pathlib import Path

import matplotlib.pyplot as plt


def plot_slice(slice_2d, title="CT Slice", cmap="gray"):
    """
    Display one 2D slice.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(slice_2d.T, cmap=cmap, origin="lower")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_middle_slice(volume, axis=2, title="Middle CT Slice"):
    """
    Display the middle slice of a 3D volume along the selected axis.
    """
    index = volume.shape[axis] // 2

    if axis == 0:
        slice_2d = volume[index, :, :]
    elif axis == 1:
        slice_2d = volume[:, index, :]
    elif axis == 2:
        slice_2d = volume[:, :, index]
    else:
        raise ValueError("axis must be 0, 1, or 2")

    plot_slice(slice_2d, title=f"{title}, axis={axis}, index={index}")


def plot_image_and_mask(image_slice, mask_slice, title="Image and Mask"):
    """
    Display a CT slice and its corresponding segmentation mask side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image_slice.T, cmap="gray", origin="lower")
    axes[0].set_title("CT image")
    axes[0].axis("off")

    axes[1].imshow(mask_slice.T, cmap="gray", origin="lower")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    output_path = "outputs/overlays/first_case.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"Saved: {output_path}")

    plt.close()


def plot_overlay(
    image_slice,
    mask_slice,
    title="CT with Tumor Overlay",
    output_path="outputs/overlays/overlay.png",
):
    """
    Save a CT slice with the tumor mask overlaid on top.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Show the CT slice in grayscale.
    ax.imshow(image_slice.T, cmap="gray", origin="lower")

    # Overlay the mask with transparency so the CT remains visible.
    ax.imshow(mask_slice.T, cmap="Reds", origin="lower", alpha=0.4)

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    print(f"Saved: {output_path}")

    plt.close(fig)

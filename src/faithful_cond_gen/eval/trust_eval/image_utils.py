"""
Image path mapping and grid creation utilities.
"""

from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


def condition_to_signature(condition: tuple, condition_keys: List[str]) -> str:
    """
    Convert condition tuple to alphabetically sorted filename signature.

    Example:
        condition = (1, 0, 0, 1)  # (Male, Smiling, Blond_Hair, Eyeglasses)
        condition_keys = ["Male", "Smiling", "Blond_Hair", "Eyeglasses"]
        -> "Blond_Hair0_Eyeglasses1_Male1_Smiling0"
    """
    # Create list of (key, value) pairs
    pairs = list(zip(condition_keys, condition))

    # Sort alphabetically by key
    pairs_sorted = sorted(pairs, key=lambda x: x[0])

    # Format as key0_key1_...
    parts = [f"{k}{v}" for k, v in pairs_sorted]
    return "_".join(parts)


def get_image_path(
    condition: tuple,
    idx: int,
    model_dir: str,
    condition_keys: List[str],
) -> Path:
    """
    Map condition tuple and index to image file path.

    Args:
        condition: Condition values tuple
        idx: Sample index within condition
        model_dir: Model directory name (e.g., "celeba_vanilla_full")
        condition_keys: Attribute names

    Returns:
        Path to image file
    """
    signature = condition_to_signature(condition, condition_keys)
    filename = f"{signature}_{idx}.png"
    return Path(f"outputs/gen/{model_dir}/images/{filename}")


def create_image_grid(
    image_paths: List[Path],
    titles: List[str],
    scores: List[Tuple[float, float]] = None,
) -> Image.Image:
    """
    Create a 2×2 grid of images with titles and optional scores.

    Args:
        image_paths: List of 4 image paths (top-left, top-right, bottom-left, bottom-right)
        titles: List of 4 titles (e.g., ["Good R + Good F", "Good R + Bad F", ...])
        scores: Optional list of 4 (realism_z, faithfulness_z) tuples

    Returns:
        PIL Image with 2×2 grid
    """
    if len(image_paths) != 4 or len(titles) != 4:
        raise ValueError("Need exactly 4 images and 4 titles for 2×2 grid")

    # Load images (or use placeholder if missing)
    images = []
    for path in image_paths:
        if path.exists():
            img = Image.open(path).convert("RGB")
        else:
            # Create placeholder
            img = Image.new("RGB", (256, 256), color=(200, 200, 200))
            draw = ImageDraw.Draw(img)
            draw.text((128, 128), "Missing", fill=(100, 100, 100), anchor="mm")
        images.append(img)

    # Resize to consistent size
    img_size = 256
    images = [
        img.resize((img_size, img_size), Image.Resampling.LANCZOS) for img in images
    ]

    # Create grid canvas (2×2 + margins for text)
    margin = 60
    grid_width = 2 * img_size + 3 * margin
    grid_height = 2 * img_size + 4 * margin
    grid = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))

    # Paste images
    positions = [
        (margin, margin),  # top-left
        (2 * margin + img_size, margin),  # top-right
        (margin, 2 * margin + img_size),  # bottom-left
        (2 * margin + img_size, 2 * margin + img_size),  # bottom-right
    ]

    for img, pos in zip(images, positions):
        grid.paste(img, pos)

    # Add titles and scores
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for i, (title, pos) in enumerate(zip(titles, positions)):
        text_x = pos[0] + img_size // 2
        text_y = pos[1] + img_size + 10

        # Draw title
        draw.text((text_x, text_y), title, fill=(0, 0, 0), anchor="mt", font=font)

        # Draw scores if provided
        if scores is not None:
            r_z, f_z = scores[i]
            score_text = f"R:{r_z:.2f} F:{f_z:.2f}"
            draw.text(
                (text_x, text_y + 20),
                score_text,
                fill=(100, 100, 100),
                anchor="mt",
                font=font,
            )

    return grid

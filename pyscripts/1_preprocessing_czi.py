
import time
import gc
import os
import argparse
from pathlib import Path
from shutil import rmtree

import czifile
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from tqdm.notebook import tqdm


def process_wsi(wsi, path_wsi, save_dir, um_per_px=0.5, tile_size=1536):
    """
    Process a single WSI into tiles.
    """

    # Handle skip list
    skip_list = [
        "02-145-Temporal_4G8.czi",
        "10-419-Parietal_4G8.czi",
        "10-813-Parietal_4G8.czi",
        "11-135-Parietal_4G8.czi",
    ]
    if wsi in skip_list:
        print("Skipping", wsi, "as it always kills the process")
        return False

    # Skip already tiled
    save_dir = Path(save_dir) / wsi
    if save_dir.exists():
        print("Skipping since WSI already tiled:", save_dir)
        return True

    # Read CZI
    readstart = time.time()
    #old code 
    # try:
    #     img = czifile.imread(os.path.join(path_wsi, wsi))
    # except Exception as e:
    #     print("Skipped due to large size or error:", wsi, str(e))
    #     return False

    # print("Time taken to read czi:", time.time() - readstart)
    # print("Shape of image:", img.shape)
    
    # img = img.squeeze()  # remove all size-1 dimensions

    # # Now ensure it's (H, W, C)
    # if img.ndim == 2:
    #     img = np.expand_dims(img, axis=-1)

    # print("Shape after squeeze:", img.shape)
    # h, w, c = img.shape[0], img.shape[1], img.shape[2]
    
    #new code with better memory handling and axis parsing
    try:
        with czifile.CziFile(os.path.join(path_wsi, wsi)) as czi:
            img = czi.asarray()
            axes = czi.axes
    except Exception as e:
        print("Skipped due to large size or error:", wsi, str(e))
        return False

    print("Time taken to read czi:", time.time() - readstart)
    print(f"Axes: {axes}, Shape: {img.shape}")

    # Keep spatial + pixel data axes; collapse everything else to index 0
    keep_axes = {'Y', 'X', 'C', '0', 'A'}
    collapse_axes = set(axes) - keep_axes

    slicing = tuple(
        0 if ax in collapse_axes else slice(None)
        for ax in axes
    )
    img = img[slicing]

    # Squeeze out remaining singleton dims, but protect dims of size > 1
    while img.ndim > 3:
        for i, s in enumerate(img.shape):
            if s == 1:
                img = img.squeeze(axis=i)
                break
        else:
            break

    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    print(f"Final shape (H, W, C): {img.shape}")
    h, w, c = img.shape

    print("Height: {}, Width: {}, Channels: {}".format(h, w, c))

    # Resolution scaling
    scan_to_desired_res = 0.11 / um_per_px
    tile_size_scan = int(tile_size / scan_to_desired_res)

    # Clean save directory
    if save_dir.exists():
        rmtree(save_dir, ignore_errors=True)

    level_dir = save_dir / "0"
    level_dir.mkdir(parents=True)

    def process_row(row_index, y):
        row_dir = level_dir / str(row_index)
        row_dir.mkdir()

        for x_index, x in enumerate(range(0, w, tile_size_scan)):
            # tile = img[0, 0, y:y+tile_size_scan, x:x+tile_size_scan]
            tile = img[y:y+tile_size_scan, x:x+tile_size_scan]

            # Pad edges if needed
            tile_h, tile_w = tile.shape[:2]
            if (tile_h, tile_w) != (tile_size_scan, tile_size_scan):
                tile = cv.copyMakeBorder(
                    tile,
                    0, tile_size_scan - tile_h,
                    0, tile_size_scan - tile_w,
                    cv.BORDER_CONSTANT,
                    value=[255, 255, 255]
                )

            # Rescale to desired tile size
            if tile.shape[:2] != (tile_size, tile_size):
                tile = cv.resize(tile, (tile_size, tile_size))

            # Convert to BGR and save
            tile = cv.cvtColor(tile, cv.COLOR_RGB2BGR)
            cv.imwrite(str(row_dir / "{}.jpg".format(x_index)), tile)
            del tile

        gc.collect()

    # Run tiling
    start = time.time()
    ys = list(range(0, h, tile_size_scan))
    for row_index, y in enumerate(tqdm(ys)):
        process_row(row_index, y)

    print("Process took seconds:", time.time() - start)

    # del img
    gc.collect()
    return True


def main():
    parser = argparse.ArgumentParser(description="Tile WSIs from CZI format.")
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Path to input directory containing WSI (.czi) files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Path to output directory for saving tiles"
    )
    parser.add_argument(
        "--um_per_px", type=float, default=0.5,
        help="Desired resolution in um/px (default: 0.5)"
    )
    parser.add_argument(
        "--tile_size", type=int, default=1536,
        help="Tile size after scaling (default: 1536)"
    )

    args = parser.parse_args()

    couldnt_tile = []
    wsi_names = os.listdir(args.input_dir)
    print("Found WSIs:", wsi_names)

    for wsi in wsi_names:
        if not wsi.endswith(".czi"):
            continue
        print("Now processing", wsi)

        ok = process_wsi(
            wsi,
            path_wsi=args.input_dir,
            save_dir=args.output_dir,
            um_per_px=args.um_per_px,
            tile_size=args.tile_size
        )

        if not ok:
            couldnt_tile.append(wsi)

    print("Could not process these WSIs due to large size/errors:", couldnt_tile)


if __name__ == "__main__":
    main()

"""
python 1_preprocessing_czi.py \
  --input_dir /cache/Shivam/braindata_repo/wsis/ \
  --output_dir /cache/Shivam/braindata_repo/norm_tiles/ \
  --um_per_px 0.5 \
  --tile_size 1536

python 1_preprocessing_czi.py \
  --input_dir /cache/Shivam/Dryad/ \
  --output_dir /cache/Shivam/braindata_repo/norm_tiles/ \
  --um_per_px 0.5 \
  --tile_size 1536
  
"""

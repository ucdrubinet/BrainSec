#!/usr/bin/env python
# coding: utf-8

# ### 1 Preprocessing - Reinhard Normalization and WSI Tiling
# 
# As a first preprocessing step, all slides were color normalized with respect to a reference image selected by an expert neuropathologist. Color normalization was performed using the method described by [Reinhard et. al](https://ieeexplore.ieee.org/document/946629).
# 
# The resulting color normalized whole slide images were tiled using PyVips to generate 1536 x 1536 images patches.

import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyvips as Vips
import pandas as pd
import argparse
import time
import gc
from tqdm import tqdm

from utils import vips_utils, normalize

def tile(WSI_DIR, SAVE_DIR, NORMALIZE):
    wsi = [pathname.split('/')[-1] for pathname in glob.glob(WSI_DIR+"*.svs")]
    imagenames = sorted(wsi)

    normalizer = normalize.Reinhard()
    if NORMALIZE:
        ref_imagename = imagenames[0]
        #print(imagenames, ref_imagename)
        ref_image = Vips.Image.new_from_file(WSI_DIR + ref_imagename, level=0)
        normalizer.fit(ref_image)

    stats_dict = {}
    print("Starting tiling....")
    for imagename in tqdm(imagenames[:]):
        start = time.time()
        vips_img = Vips.Image.new_from_file(WSI_DIR + imagename, level=0)
        print("____________________________________________")
        print("Loaded Image: " + WSI_DIR + imagename)
        print("Width x Height: ", vips_img.width, "x", vips_img.height)
        if NORMALIZE:
            out = normalizer.transform(vips_img)
            vips_utils.save_and_tile(out, SAVE_DIR)
            os.rename(os.path.join(SAVE_DIR, out.filename), os.path.join(SAVE_DIR, os.path.basename(vips_img.filename).split('.svs')[0]))
            stats_dict[imagename] = normalizer.image_stats
            try:
                del out
            except:
                pass
        else:
            out = vips_img
            vips_utils.save_and_tile(vips_img, SAVE_DIR)
            #os.rename(os.path.join(SAVE_DIR, out.filename), os.path.join(SAVE_DIR, os.path.basename(vips_img.filename).split('.svs')[0]))
            #stats_dict[imagename] = normalizer.image_stats
        try:
            del vips_img
            gc.collect()
        except:
            pass
        
        print("processed in ", time.time()-start," seconds")
    
    if NORMALIZE:
        stats = pd.DataFrame(stats_dict)
        stats = stats.transpose()
        stats.columns = 'means', 'stds'
        #print(stats)
        stats.to_csv(SAVE_DIR + "stats.csv")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi_dir", type=str, default='data/wsi/', help="Directory of Whole Slide Images")
    parser.add_argument("--save_dir", type=str, default='data/norm_tiles/', help="Directory to save the patches for heatmaps")
    parser.add_argument("--normalize", type=bool, default=False)

    args = parser.parse_args()

    print(f"WSI Directory: {args.wsi_dir}")
    print(f"Save Directory: {args.save_dir}")
    print(f"Normalize: {args.normalize}")


    WSI_DIR = args.wsi_dir #'data_1_40/wsi/'
    SAVE_DIR = args.save_dir #'data_1_40/norm_tiles/'
    NORMALIZE = args.normalize


    if not os.path.exists(WSI_DIR):
        print("WSI folder does not exist, script should stop now")
    else:
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        print("Found WSI folder... ")
        wsi_files = os.listdir(WSI_DIR)
        filenames = sorted(wsi_files)
        print("All files in wsi_dir: ")
        print(filenames)

    #----------------------------------------------------------

    tile(WSI_DIR, SAVE_DIR, NORMALIZE)
    print("____________________________________________")
    print("WSI tiled for heatmaps")

if __name__ == "__main__":
    main()
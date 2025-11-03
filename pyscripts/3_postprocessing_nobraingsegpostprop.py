#!/usr/bin/env python
# coding: utf-8

# ### 3 Post-processing and Counting Plaques at each Segmentation Area
# 

import csv
import glob, os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import morphology, measure
from skimage.color import hsv2rgb
from scipy import stats, ndimage
from tqdm import tqdm
from PIL import Image
import argparse


def saveBrainSegImage(nums, save_dir) :
    """
    Converts 2D array with {0,1,2} into RGB
     to determine different segmentation areas
     and saves image at given directory
    
    Input:
       nums: 2D-NumPy Array containing classification
       save_dir: string indicating save location
    """ 
    
    nums = np.repeat(nums[:,:, np.newaxis], 3, axis=2)
    
    # nums[:,:,0] = RED, nums[:,:,1] = Green, nums[:,:,2] = Blue
    idx_1 = np.where(nums[:,:,0] == 1)  # Index of label 1 (WM)
    idx_2 = np.where(nums[:,:,0] == 2)  # Index of label 2 (GM)
    WM_count, GM_count =len(idx_1[0]),len(idx_2[0])
    # For label 0, leave as black color
    # For label 1, set to yellow color: R255G255B0 (WM)
    nums[:,:,0].flat[np.ravel_multi_index(idx_1, nums[:,:,0].shape)] = 255
    nums[:,:,1].flat[np.ravel_multi_index(idx_1, nums[:,:,1].shape)] = 255
    nums[:,:,2].flat[np.ravel_multi_index(idx_1, nums[:,:,2].shape)] = 0
    # For label 2, set to cyan color: R0G255B255 (GM)
    nums[:,:,0].flat[np.ravel_multi_index(idx_2, nums[:,:,0].shape)] = 0
    nums[:,:,1].flat[np.ravel_multi_index(idx_2, nums[:,:,1].shape)] = 255
    nums[:,:,2].flat[np.ravel_multi_index(idx_2, nums[:,:,2].shape)] = 255

    nums = nums.astype(np.uint8) # PIL save only accepts uint8 {0,..,255}
    # save_img = Image.fromarray(nums, 'RGB')
    # save_img.save(save_dir)
    # print("Saved at: " + save_dir)
    return WM_count, GM_count


# # Post-Processing BrainSeg - Jeff, Kolin, Wenda
# def method_6(mask_img: "Image", down_factor=4) -> "NDArray[np.uint8]":
#     """Downsample => Area_opening (Remove local maxima) =>
#     Swap index of GM and WM => Area_opening => Swap index back =>
#     Area_closing => Morphological opening => Upsample"""
#     # pylint: disable=invalid-name
#     def swap_GM_WM(arr):
#         """Swap GM and WM in arr (swaps index 1 and index 2)"""
#         arr_1 = (arr == 1)
#         arr[arr == 2] = 1
#         arr[arr_1] = 2
#         del arr_1
#         return arr
#     # pylint: enable=invalid-name

#     mask_img = Image.fromarray(mask_img)
#     width, height = mask_img.width, mask_img.height
#     area_threshold_prop = 0.05
#     area_threshold = int(area_threshold_prop * width * height // down_factor**2)

#     # Downsample the image
#     mask_arr = np.array(
#         mask_img.resize((width // down_factor, height // down_factor), Image.NEAREST))
#     del mask_img
#     print('Finish downsampling')

#     # Apply area_opening to remove local maxima with area < 20000 px
#     mask_arr = morphology.area_opening(mask_arr, area_threshold=3200 // down_factor**2)
#     print('Finish area_opening #1')

#     # Swap index of GM and WM
#     mask_arr = swap_GM_WM(mask_arr)
#     print('Finish swapping index')

#     # Apply area_opening to remove local maxima with area < 20000 px
#     mask_arr = morphology.area_opening(mask_arr, area_threshold=3200 // down_factor**2)
#     print('Finish area_opening #2')

#     # Swap index back
#     mask_arr = swap_GM_WM(mask_arr)
#     print('Finish swapping index back')

#     # Apply area_closing to remove local minima with area < 12500 px
#     mask_arr = morphology.area_closing(mask_arr, area_threshold=2000 // down_factor**2)
#     print('Finish area_closing')

#     # Apply remove_small_objects to remove tissue residue with area < 0.05 * width * height
#     tissue_arr = morphology.remove_small_objects(mask_arr > 0, min_size=area_threshold,
#                                                  connectivity=2)
#     mask_arr[np.invert(tissue_arr)] = 0
#     del tissue_arr
#     print('Finish remove_small_objects')

#     # Apply opening with disk-shaped kernel (r=8) to smooth boundary
#     mask_arr = morphology.opening(mask_arr, footprint=morphology.disk(radius=32 // down_factor))
#     print('Finish morphological opening')

#     # Upsample the output
#     mask_arr = np.array(Image.fromarray(mask_arr).resize((width, height), Image.NEAREST))
#     print('Finish upsampling')

#     return mask_arr


# UserWarning: The argument 'neighbors' is deprecated and will be removed in scikit-image 0.18,
# use 'connectivity' instead. For neighbors=8, use connectivity=2
#   This is separate from the ipykernel package so we can avoid doing imports until

# Post-Processing to count Plaques
def count_blobs(mask,
               threshold=1500):
#     labels = measure.label(mask, neighbors=8, background=0)
    labels = measure.label(mask, connectivity=2, background=0)
    img_mask = np.zeros(mask.shape, dtype='uint8')
    labeled_mask = np.zeros(mask.shape, dtype='uint16')
    sizes = []
    locations = []
    
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels 
        labelMask = np.zeros(mask.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > threshold:
            sizes.append(numPixels)
            img_mask = cv2.add(img_mask, labelMask)
            
            # Save confirmed unique location of plaque
            labeled_mask[labels==label] = label

    return sizes, img_mask, labeled_mask


# from PIL import Image
def saveMask(mask_array, save_dir) :
    
    mask_array = np.repeat(mask_array[:,:, np.newaxis], 3, axis=2)
    
    # mask_array[:,:,0] = RED, mask_array[:,:,1] = Green, mask_array[:,:,2] = Blue
    idx = np.where(mask_array[:,:,0] == 255)  # Index of label 1 (WM)

    # For label 0, leave as black color
    # For label 1, set to cyan color: R0G255B255
    mask_array[:,:,0].flat[np.ravel_multi_index(idx, mask_array[:,:,0].shape)] = 0
    mask_array[:,:,1].flat[np.ravel_multi_index(idx, mask_array[:,:,1].shape)] = 255
    mask_array[:,:,2].flat[np.ravel_multi_index(idx, mask_array[:,:,2].shape)] = 255

    mask_array = mask_array.astype(np.uint8) # PIL save only accepts uint8 {0,..,255}
    save_img = Image.fromarray(mask_array, 'RGB')
    save_img.save(save_dir)
    print("Saved at: " + save_dir)


def saveUniqueMaskImage(maskArray, save_dir) :
    '''
    Plots post-processed detected Plaques
    with the diversity of Colour distingushing
    the density of Plaques
    
    ie. More Diversity of Colour
    == More Plaque Count for that certain Plaque type
    
    Inputs:
        maskArray = Numpy Array containing Unique plaque
        save_dir  = String for Save Directory
    '''
    
    max_val = np.amax(np.unique(maskArray))
#     print("Maximum Value = ", max_val)
    maskArray = np.asarray(maskArray, dtype=np.float64)
    maskArray = np.repeat(maskArray[:,:, np.newaxis], 3, axis=2)

    for label in np.unique(maskArray) :

        # For label 0, leave as black color (BG)
        if label == 0:
            continue

        idx = np.where(maskArray[:,:,0] == label) 

        # For label, create HSV space based on unique labels
        maskArray[:,:,0].flat[np.ravel_multi_index(idx, maskArray[:,:,0].shape)] = label / max_val
        maskArray[:,:,1].flat[np.ravel_multi_index(idx, maskArray[:,:,1].shape)] = label % max_val
        maskArray[:,:,2].flat[np.ravel_multi_index(idx, maskArray[:,:,2].shape)] = 1

    rgb_maskArray = hsv2rgb(maskArray)
    rgb_maskArray = rgb_maskArray * 255
    rgb_maskArray = rgb_maskArray.astype(np.uint8) # PIL save only accepts uint8 {0,..,255}
    
    save_img = Image.fromarray(rgb_maskArray, 'RGB')
    save_img.save(save_dir)
    print("Saved at: " + save_dir)


def classify_blobs(labeled_mask, seg_area) :
    """
    Classifies each certain plaques according to each
    Segmentation Area and gives each count
    
    Input:
        labeled_mask (NumPy Array): 
            contains plaque information 
            Note: See count_blobs()'s 
            labeled_mask output for more info
        
        seg_area (NumPy Array):
            contains segmentation information
            based on BrainSeg's classification
            
    Output:
        count_dict (Dictionary):
            contains number of plaques at each
            segmentaion area
            
        Other Variables:
            - Background Count
            - WM Count
            - GM Count
            - Unclassified Count
    """
    
    # 0: Background, 1: WM, 2: GM
    count_dict = {0: 0, 1: 0, 2: 0, "uncounted": 0}
    # Loop over unique components
    for label in np.unique(labeled_mask) :
        if label == 0:
            continue
            
        plaque_loc = np.where(labeled_mask == label)
        plaque_area = seg_area[plaque_loc]
        indexes, counts = np.unique(plaque_area, return_counts=True)
        class_idx = indexes[np.where(counts == np.amax(counts))]
        
        try:
            class_idx = class_idx.item()
            count_dict[class_idx] += 1
                
        except:
            count_dict["uncounted"] += 1
            
    return count_dict, count_dict[0], count_dict[1], count_dict[2], count_dict["uncounted"]


def get_filenames(BRAINSEG_NP_PRE_DIR):
    filenames = sorted(os.listdir(BRAINSEG_NP_PRE_DIR))
    filenames = [os.path.splitext(file)[0] for file in filenames]
    return filenames


def post_brainseg(BRAINSEG_NP_PRE_DIR, POST_NP_DIR, POST_IMG_DIR, filenames):
    # Post-process BrainSeg
    for filename in tqdm(filenames) :
        fileLoc = BRAINSEG_NP_PRE_DIR + filename + ".npy"
        print("Loading: " + fileLoc)
        seg_pic = np.load(fileLoc)
        processed = method_6(seg_pic)
        np.save(POST_NP_DIR+filename, processed)
        saveBrainSegImage(processed, \
                          POST_IMG_DIR + filename + '.png')


def post_plaque(SAVE_DIR, CSV_FILE, HEATMAP_DIR, POST_NP_DIR, SAVE_IMG_DIR, SAVE_NP_DIR, filenames):
    # To create CSV containing WSI names for
    # plaque counting at different regions
    file = pd.DataFrame({"WSI_ID": filenames})
    file.to_csv(CSV_FILE, index=False)
    print('Index CSV:', CSV_FILE)

    # Using existing CSV
    file = pd.read_csv(CSV_FILE)
    filenames = list(file['WSI_ID'])
    img_class = ['cored', 'diffuse', 'caa']
    
    # two hyperparameters (For Plaque-Counting)
    confidence_thresholds = [0.1, 0.95, 0.9]
    pixel_thresholds = [100, 1, 200]

    new_file = file
    for index in [0,1,2]:
        preds = np.zeros(len(file))
        confidence_threshold = confidence_thresholds[index]
        pixel_threshold = pixel_thresholds[index]
        
        bg = np.zeros(len(file))
        wm = np.zeros(len(file))
        gm = np.zeros(len(file))
        wmpxlcount= np.zeros(len(file))
        gmpxlcount= np.zeros(len(file))
        unknowns = np.zeros(len(file))

        for i, WSIname in enumerate(tqdm(filenames)):
            print("saving things at",SAVE_IMG_DIR,SAVE_NP_DIR)
            try:
                heatmap_path = HEATMAP_DIR+'new_WSI_heatmap_{}.npy'.format(WSIname)
                h = np.load(heatmap_path)

            except:
                heatmap_path = HEATMAP_DIR+'{}.npy'.format(WSIname)
                h = np.load(heatmap_path)
                print(POST_NP_DIR)
                seg_path = POST_NP_DIR+'{}.npy'.format(WSIname)
                seg = np.load(seg_path)

            mask = h[index] > confidence_threshold
            mask = mask.astype(np.float32)
            print("paths",seg_path,heatmap_path)
            # exit()

            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

            # Apply morphological closing, then opening operations 
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

            labels, img_mask, labeled_mask = count_blobs(closing, threshold=pixel_threshold)
            counts, bg[i], wm[i], gm[i], unknowns[i] = classify_blobs(labeled_mask, seg)
        
            save_img = SAVE_IMG_DIR + WSIname \
                        + "_" + img_class[index] + ".png"
            save_np = SAVE_NP_DIR + WSIname \
                        + "_" + img_class[index] + ".npy"
            print("saving things at",save_np,save_img)
            np.save(save_np, labeled_mask)
            saveUniqueMaskImage(labeled_mask, save_img) # To show Colored Result
    #         saveMask(img_mask, save_img)  # To show Classification Result
            
            preds[i] = len(labels)
            
            print(confidence_threshold, pixel_threshold)
            if index==0:
                wmpxlcount[i], gmpxlcount[i] = saveBrainSegImage(seg, "")
                print("WM area pixel count",wmpxlcount[i],"GM area pixel count",gmpxlcount[i])
        new_file['CNN_{}_count'.format(img_class[index])] = preds
        new_file['BG_{}_count'.format(img_class[index])] = bg
        new_file['GM_{}_count'.format(img_class[index])] = gm
        new_file['WM_{}_count'.format(img_class[index])] = wm
        new_file['{}_no-count'.format(img_class[index])] = unknowns
        if index==0:
            new_file['WM_count'] = wmpxlcount
            new_file['GM_count'] = gmpxlcount
        print("saving csv file at",SAVE_DIR)
        new_file.to_csv(SAVE_DIR+'CNN_vs_CERAD.csv', index=False)
        
    new_file.to_csv(SAVE_DIR+'CNN_vs_CERAD.csv', index=False)
    print('CSVs saved at', SAVE_DIR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data/', help="Data Directory")

    args = parser.parse_args()
    print("Data Directory:,",args.data_dir)
    ODR = args.data_dir #'data_1_40'
    
    # Plaque-counting Directories
    SAVE_DIR = ODR + '/outputs/CNNscore_nopostprocess/'
    CSV_FILE = ODR + '/outputs/CNNscore_nopostprocess/WSI_CERAD_AREA.csv'
    HEATMAP_DIR = ODR + '/outputs/heatmaps/'

    # BrainSeg Post-processing Directories
    BRAINSEG_NP_PRE_DIR = ODR + '/brainseg/numpy/'
    POST_IMG_DIR = ODR + '/postprocess/images/'
    POST_NP_DIR = ODR + '/postprocess/numpy/'

    # Counted Plaques Save Directories
    SAVE_IMG_DIR = ODR + '/outputs/masked_plaque_nopostprocess/images/'
    SAVE_NP_DIR = ODR + '/outputs/masked_plaque_nopostprocess/numpy/'


    if not os.path.exists(ODR):
        print("Data folder does not exist, script should stop now")
    elif not os.path.exists(BRAINSEG_NP_PRE_DIR):
        print("Mask folder does not exist, script should stop now")
    elif not os.path.exists(HEATMAP_DIR):
        print("Heatmap folder does not exist, script should stop now")
    else:
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        if not os.path.exists(POST_IMG_DIR):
            os.makedirs(POST_IMG_DIR)
        if not os.path.exists(POST_NP_DIR):
            os.makedirs(POST_NP_DIR)
        if not os.path.exists(SAVE_IMG_DIR):
            os.makedirs(SAVE_IMG_DIR)
        if not os.path.exists(SAVE_NP_DIR):
            os.makedirs(SAVE_NP_DIR)

        POST_IMG_DIR =""# ODR + '/postprocess/images/'
        POST_NP_DIR = ""#ODR + '/postprocess/numpy/'

        print("Found Data folder... ")
        data_files = os.listdir(BRAINSEG_NP_PRE_DIR)
        filenames = sorted(data_files)
        print("All files in data_dir: ")
        print(filenames)

    #----------------------------------------------------------

    filenames = get_filenames(BRAINSEG_NP_PRE_DIR)
    print("____________________________________________")
    print("Post-processing for BrainSeg ...")
    # post_brainseg(BRAINSEG_NP_PRE_DIR, POST_NP_DIR, POST_IMG_DIR, filenames)
    print("Post-processing for BrainSeg finished")
    print("____________________________________________")
    print("Post-processing for Plaques ...")
    # post_plaque(SAVE_DIR, CSV_FILE, HEATMAP_DIR, POST_NP_DIR, SAVE_IMG_DIR, SAVE_NP_DIR, filenames)
    post_plaque(SAVE_DIR, CSV_FILE, HEATMAP_DIR, BRAINSEG_NP_PRE_DIR, SAVE_IMG_DIR, SAVE_NP_DIR, filenames)
    print("Post-processing for Plaques finished")
    print("____________________________________________")
    print("Post-processing finished")

if __name__ == "__main__":
    main()
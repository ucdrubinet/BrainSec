# BrainSec

Automated Grey and White Matter Segmentation in Digitized A*β*
Human Brain Tissue WSI. This is the implementation details for the paper: 
Z. Lai, L. Cerny Oliveira, R. Guo, W. Xu, Z. Hu, K. Mifflin, C. DeCarlie, S-C. Cheung, C-N. Chuah, and B. N. Dugger, "BrainSec: Automated Brain Tissue Segmentation Pipeline for Scalable Neuropathological Analysis," IEEE Access, 2022.

## BrainSec Authors / Contributors

- Runlin Guo
- Wenda Xu
- Zhengfeng Lai

If you have any questions/suggestions or find any bugs,
please submit a GitHub issue.

## Python Pipeline for CZI Compatiblity 

- The Python port is based on the original Brainsec pipeline (with post-processing) as implemented in [Plaque_Quantification.ipynb](Plaque_Quantification.ipynb).

# BrainSec Processing Pipeline Scripts for CZI image inference

This repository includes multiple Python scripts that implement different stages of the BrainSec pipeline for whole slide image (WSI) segmentation. Depending on whether your input data is in CZI or SVS format, different preprocessing and postprocessing scripts are required.

-----
## Environment Setup
Clone this repo and switch to branch "czi_compatibility"
```bash
pip install -r brainsec_requirements.txt
```

And:
```
git clone https://github.com/keiserlab/plaquebox-paper.git

/home/shivam/plaquebox-paper/
├── utils/
│   └── normalization.npy
└── ...
```
Refer to the directory structure below to see where to keep your WSIs

## Run an example
```bash
bash run_example.sh
```
## 1\. Preprocessing

### `1_preprocessing_czi.py`

Preprocessing script for CZI files (new compatibility feature). This script loads `.czi` WSIs, tiles them into smaller image patches, and prepares them for batch inference.

**Example usage:**

```bash
python 1_preprocessing_czi.py --path_wsi /home/shivam/braindata_repo/wsis/ --save_dir /home/shivam/braindata_repo/norm_tiles/ 

```

### `1_preprocessing.py`

Preprocessing script for SVS files (legacy support).

-----

## 2\. Inference

### `2_inference_czi.py`

Runs the BrainSec inference model on preprocessed tiles. Saves segmentation outputs as `.npy` files in automatically created folders.
```bash
python 2_inference_czi.py \
  --img_dir /cache/braindata_repo/norm_tiles/ \
  --model_plaq /path/to/CNN_model_parameters.pkl \
  --model_seg  /path/to/ResNet18_19.pkl \
  --save_plaq_dir /cache/braindata_repo/outputs/heatmaps/ \
  --save_img_dir  /cache/braindata_repo/brainseg/images/ \
  --save_np_dir   /cache/braindata_repo/brainseg/numpy/ \
  --plaquebox_root /cache/plaquebox-paper \
  --normalization  /cache/plaquebox-paper/utils/normalization.npy
```
-----

## 3\. Postprocessing

### `3_postprocessing.py`

Legacy postprocessing script for SVS inputs.

### `3_postprocessing_nobraingsegpostprop.py`

Updated postprocessing script for CZI workflows. Drops the BrainSeg-specific postprocessing step to produce more accurate white matter boundaries. Use this instead of `3_postprocessing.py` for CZI data.

-----

## Recommended Workflow

### For CZI data:

1.  Run `1_preprocessing_czi.py`
2.  Run `2_inference_czi.py`
3.  Run `3_postprocessing_nobraingsegpostprop.py`

### For SVS data (legacy):

1.  Run `1_preprocessing.py`
2.  Run `2_inference.py`
3.  Run `3_postprocessing.py`


The directory structure is like this:

```
/home/shivam/braindata_repo/
├── wsis/                # raw input whole slide images (.czi files)
│   ├── sample1.czi
│   ├── sample2.czi
│   └── ...
│
├── norm_tiles/          # output of 1_preprocessing_czi.py
│   ├── sample1.czi/     # each WSI gets its own folder
│   │   └── 0/           # tiling level directory
│   │       ├── 0/...
│   │       ├── 1/...
│   │       └── ...
│   ├── sample2.czi/
│   │   └── 0/...
│   └── ...
│
├── outputs/
│   └── heatmaps/        # output heatmaps from 2_inference_czi.py
│       ├── sample1.npy or .png
│       └── ...
│
├── brainseg/
│   ├── images/          # segmentation result images from inference
│   │   ├── sample1.png
│   │   └── ...
│   └── numpy/           # segmentation result numpy arrays
│       ├── sample1.npy
│       └── ...
│
└── (used as --data_dir in step 3_postprocessing_nobraingsegpostprop.py)
```

And separately:
```
git clone https://github.com/keiserlab/plaquebox-paper.git

/home/shivam/plaquebox-paper/
├── utils/
│   └── normalization.npy
└── ...
```

So

* **Input**: `/home/shivam/braindata_repo/wsis/` with `.czi` files.
* **Step 1 output**: `/home/shivam/braindata_repo/norm_tiles/` with per-WSI tile folders.
* **Step 2 outputs**: heatmaps (`/outputs/heatmaps/`), brain segmentation images (`/brainseg/images/`), and numpy arrays (`/brainseg/numpy/`).
* **Step 3**: runs over the whole `/home/shivam/braindata_repo/` directory structure to do counting.



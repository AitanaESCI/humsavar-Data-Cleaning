# humsavar-Data-Cleaning

This repository contains code to clean and filter genetic variant data from Humsavar, ensuring quality and consistency for downstream analysis. The primary goal is to preprocess the dataset for accurate Variant Effect Predictor (VEP) usage, allowing pathogenicity predictions on a curated dataset.

### Repository structure

The repository is organized as follows:
```bash
├── README.md
├── data/
│   ├── humsavar_202102.txt
│   ├── humsavar_202501.txt
│   ├── humsavar_20212025.csv
│   ├── humsavar_20212025_v2.csv
│   ├── humsavar_rsIDs.txt
│   ├── 
│   ├── 
│   └──
├── docs/
│   ├── DatasetCollection_humsavar.pptx
│   └── Thresholds_log.xlsx
├── notebooks/
│   └── Humsavar_DataCleaning.ipynb
└── scripts/
    └── VEP.py
```


#### 1. `data/`

This directory contains both the raw and processed Humsavar datasets. The `humsavar_202102.txt` and `humsavar_202501.txt` files represent different versions of the original dataset. The `humsavar_20212025.csv` and `humsavar_20212025_v2.csv` files store progressively cleaned datasets that have undergone filtering and standardization. The `humsavar_rsIDs.txt` file maps variants to their corresponding Reference SNP IDs (rsIDs), aiding in database integration and annotation. These processed datasets are structured for optimal compatibility with the VEP tool.


#### 2. `docs/`

This directory contains documentation and reference materials. The `DatasetCollection_humsavar.pptx` presentation outlines data sources, filtering strategies, and preprocessing workflow applied to Humsavar. The `Thresholds_log.xlsx` file includes literature-based cutoff values for pathogenicity predictors, detailing thresholds derived from research papers to classify variants effectively.


#### 3. `notebooks/`

This directory contains the main Jupyter Notebook (`Humsavar_DataCleaning.ipynb`), where the dataset preprocessing steps are performed. The notebook details data cleaning, variant filtering, normalization, and preparation for VEP input. The filtering process ensures high-quality variant selection by removing duplicates, handling missing values, and structuring the dataset to improve annotation accuracy.


#### 4. `scripts/`

This directory contains the `VEP.py` script, designed to process and analyze output from the VEP tool. After running VEP, this script extracts relevant annotations, standardizes the output format, and prepares the data for downstream pathogenicity prediction models (i.e., filtering of column names, binarization of existing columns, etc). It enables automated extraction of functional effect scores and impact predictions.

# humsavar Data Cleaning

This repository contains code to clean and filter genetic variant data from Humsavar, ensuring quality and consistency for later analysis. The primary goal is to preprocess the dataset for accurate Variant Effect Predictor (VEP) usage, enabling pathogenicity predictions on a curated dataset.


## Repository structure

The repository is organized as follows:
```bash
├── README.md
├── data/
│   ├── humsavar_202102.txt
│   ├── humsavar_202501.txt
│   ├── uniprotkb_reviewed_true_AND_organism_id_2025_04_09.list
│   ├── humsavar_20212025.csv
│   ├── humsavar_20212025_v2.csv
│   ├── humsavar_rsIDs.txt
│   ├── cleaned_Humsavar_dataset_outputVEP.txt
│   ├── cleaned_Humsavar_dataset_parsed.txt
│   ├── cleaned_Humsavar_dataset_with_preds.txt
│   └── humsavar_no_clinvar.csv
└── code/
    └── Humsavar_DataCleaning.ipynb
```


## Directory overview

#### 1. `data/`

This directory contains raw input files, intermediate filtered datasets, and VEP compatible output:
* Original Humsavar exports for different dataset versions.
* Progressively cleaned and merged datasets prepared for filtering and annotation.
* VEP outputs with parsed and formatted results.
* Reference list and mappings: reviewed UniProt entries and rsID match files.

**Note**: due to file size constraints, this folder is not included directly in the repository.
All files can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.15275485) link. See the [Data Availability](#data-availability) section for setup instructions.


#### 2. `code/`

This folder contains the main Jupyter Notebook:
	•	Humsavar_DataCleaning.ipynb implements the full preprocessing pipeline, including cleaning, filtering, deduplication, standardization, and formatting for VEP input.


## Data availability
To access the dataset files:
1. Visit  [![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.15275485.svg)](https://doi.org/10.5281/zenodo.15275485)
2. Download the full `data/` directory and place it inside the repository.
3. Ensure the folder structure matches the one described above.

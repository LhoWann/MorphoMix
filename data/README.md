# Data

`raw/` holds each dataset as downloaded (archives extracted, never edited); `processed/` is built from it by
`python main.py prepare` (scripts/data.py); `processed/splits/` by `python main.py split` and `split --allidb2`
(scripts/data.py). Everything under `raw/` and the image folders of `processed/` are
git-ignored; `processed/splits/` is tracked.

| Folder | Dataset | Laboratory | Role | Licence |
| --- | --- | --- | --- | --- |
| `raw/C-NMC_2019/` | C-NMC 2019 (ISBI challenge), folds 0-2, single-cell crops, `all` / `hem` | SBILab, AIIMS New Delhi | train | TCIA terms |
| `raw/LeukemiaAttri/H_100X_C1/` | LeukemiaAttri / LLD, 100x full fields with WBC boxes (high-cost microscope, camera 1) | Chughtai Labs / ITU, Lahore | val (lymphoblast vs lymphocyte crops) | CC BY-NC-SA 4.0 |
| `raw/MLL23/` | MLL23, 41,621 single cells in 18 classes (labels unused) | Munich Leukemia Laboratory | stain references of MorphoMix C1 and the stain_mixup baseline (style bank, labels unused); as auxiliary training cells it was tried and rejected | CC BY 4.0 |
| `raw/Bodzas2023/` | Bodzas et al. 2023 WBC dataset, 1200 px BMP single-cell crops; downloaded subset: Lymphoblast 2,557, Lymphocyte 2,000, Monocyte 300 (members of the 46 GB zip fetched by HTTP range, CRC-checked, byte-identical to the archive) | VSB Technical University of Ostrava / City Hospital Ostrava (MGG, Olympus BX51, 100x oil) | with `aux_train.enabled`: auxiliary training cells (lymphoblasts as ALL, lymphocytes and monocytes as Normal) | CC0, figshare doi 10.6084/m9.figshare.22680517 |
| `raw/ALL-IDB2/` | ALL-IDB2, 260 single-cell crops | Università degli Studi di Milano | test (single cell) | on request, not redistributable |
| `raw/Aria_B-ALL/Original/` | Aria et al., 3,256 full fields, Benign / Early / Pre / Pro | Taleqani Hospital, Tehran | test (multi-cell, binary) | Kaggle, see source |

`processed/` (all 224 px PNG, `{Normal,ALL}` sub-folders):

| Folder | Built from | Notes |
| --- | --- | --- |
| `train/` | C-NMC, every fold (10,661) | |
| `val/` | LeukemiaAttri lymphoblast / lymphocyte boxes (1,321) | `src/datasets/leukemiaattri.py`: squeeze x0.60 corrected, one fixed window (2.18 x median box side), edge-cut cells, CLL-slide lymphocytes, three boxes on erythrocytes (`MISPLACED_BOXES`), crops showing an opposite-class cell and same-cell repeats dropped, ALL capped at 100 per slide; `manifest.csv` |
| `test_allidb2/` | ALL-IDB2 (260) | ten non-square crops edge-padded to a square; 8 groups of crops of one cell (17 images: Im025/028/031_1, Im134/245, Im135/244, Im138/146, Im171/211, Im222/256, Im223/257 (identical), Im225/259, all `_0`) |
| `test_aria/` | Aria (3,256): Benign = Normal, Early / Pre / Pro = ALL | scale-bar corner painted out (`aria_annotation_box`) |
| `style_bank_mll23.npz` (+ `.json`) | MLL23 (41,621 cells), read straight from `raw/MLL23` | per cell the Lab mean / sd of nucleus and cytoplasm and a Macenko stain matrix (`src/augmentations/style_bank.py`); no image is copied. The JSON holds the provenance `prepare` compares before rebuilding and the content SHA-256 recorded with every run that uses the bank |
| `randstainna_stats.json` | `train/` (C-NMC) | Lab moment distribution of the training images for the randstainna baseline, refitted when `train/` changes |
| `train_aux_bodzas2023/` | Bodzas lymphoblasts (ALL, 2,250) and lymphocytes + monocytes (Normal, 1,049), C-NMC's class ratio | with `aux_train.enabled` (default): `smear_cell` mask (GrabCut from the nucleus, erythrocyte-coloured pixels dropped beyond a 7 px rim) cut onto black, added to the training set of every arm; exploratory (decision log) |
| `train_aux_bodzas2023_cmatch/` | `train_aux_bodzas2023/` | the same cells with their cell pixels Reinhard-matched in Lab to the mean C-NMC template (`randstainna_stats`), so their slide colour no longer separates the classes; control arm g (`aux_train.colour_matched_dir`) |
| `splits/` | `leukemiaattri_same_cell.csv` (cells imaged in two overlapping fields), the Aria field groups (`split`) and `allidb2_groups.csv` (crops of one cell, `split --allidb2`; 260 rows, 251 groups) | tracked |

ALL-IDB1 must never be added: ALL-IDB2 is cropped from it, so it would leak a test cohort. The LeukemiaAttri
subset `L_100X_C1` (low-cost microscope, `raw/LeukemiaAttri/L_100X_C1/`, from the dataset's Google Drive) is the
pre-registered confirmatory cohort (`python main.py confirm`, crops in `processed/confirm_l100x/`; decision log
entry of 2026-10-07). It is never used for training, selection or any method choice.

Sources: [C-NMC](https://doi.org/10.7937/tcia.2019.dc64i46r),
[ALL-IDB](https://homes.di.unimi.it/scotti/all/),
[LeukemiaAttri](https://github.com/intelligentMachines-ITU/Blood-Cancer-Dataset-Lukemia-Attri-MICCAI-2024),
[MLL23](https://doi.org/10.5281/zenodo.14277609),
[Bodzas 2023](https://doi.org/10.6084/m9.figshare.22680517) (paper: https://doi.org/10.1038/s41597-023-02378-7),
[Aria](https://doi.org/10.34740/KAGGLE/DSV/2175623).

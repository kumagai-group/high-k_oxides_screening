# Physics-Based Factorized Machine Learning for Predicting Ionic Dielectric Tensors

This repository contains the **data and source codes** used in the following paper:

> **Physics-Based Factorized Machine Learning for Predicting Ionic Dielectric Tensors**
> Atsushi Takigawa, Shin Kiyohara, and Yu Kumagai  
> Phys. Rev. X (in press)
> https://doi.org/10.1103/28wr-w896

---

## Contents

```text
.
├── common/             # Utility functions and shared modules
├── dataset/            # Training datasets and screening datasets
├── model/              # Machine learning models and training routines
├── phonon/             # Phonon calculations and feature generation
├── run_cv.py           # Script for k-fold cross-validation
├── run_fulltrain.py    # Script for training using the full dataset
├── results_screening   # First-principles calculation results for screened candidate materials
└── README.md
```

---

## Dataset

The `dataset` directory contains:

- **Training dataset:** Data for **928 oxides** used to train the machine learning model.
- **Screening dataset:** Data for **8,171 oxides** collected from the Materials Project database for large-scale screening.

### Training Dataset

- st_pbesol: Crystal structures optimized using the **PBEsol exchange–correlation functional**.
- dielectric_pbesol: Electronic, ionic, and total dielectric tensors calculated using **density functional perturbation theory (DFPT)** with the **PBEsol functional**.
- phonon_pbesol: Phonon eigenfrequencies calculated using **DFPT with PBEsol**.
- bandgap_ddh: Band gap values calculated using the **DDH method**.

### Screening Dataset

- st_mp: Crystal structures collected from the Materials Project database.
- bandgap_mp: Band gap values obtained from the Materials Project database.

---

## Screening Results

The `results_screening` directory contains **first-principles calculation results for 142 candidate oxides** identified through the machine-learning-based screening.

- st_pbesol: Crystal structures optimized using the **PBEsol exchange–correlation functional**.
- dielectric_pbesol: Electronic, ionic, and total dielectric tensors calculated using **DFPT with PBEsol**.
- phonon_pbesol: Phonon eigenfrequencies calculated using **DFPT with PBEsol**.
- bandgap_ddh: Band gap values calculated using the **DDH method**.

If you encounter any issues, please contact the authors through the issue tracker or email yukumagai@tohoku.ac.jp.

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

The dataset was originally constructed in our previous work [1,2].

- **Screening dataset:** Data for **8,171 oxides** collected from the Materials Project [3] database for large-scale screening.

### Training Dataset

- st_pbesol: Crystal structures optimized using the **PBEsol exchange–correlation functional**.
- dielectric_pbesol: Electronic, ionic, and total dielectric tensors calculated using **density functional perturbation theory (DFPT)** with the **PBEsol functional**.
- phonon_pbesol: Phonon eigenfrequencies calculated using **DFPT with PBEsol**.
- bandgap_ddh: Band gap values calculated using the ** dielectric-dependent hybrid (DDH) method**.

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

### References

```
[1]A. Takahashi, Y. Kumagai, J. Miyamoto, Y. Mochizuki, and F. Oba
Machine learning models for predicting the dielectric constants of oxides based on high-throughput first-principles calculations
Phys. Rev. Materials 4, 103801 (2020).
doi:10.1103/PhysRevMaterials.4.103801

[2] Y. Kumagai, N. Tsunoda, A. Takahashi, and F. Oba
Insights into oxygen vacancies from high-throughput first-principles calculations
Phys. Rev. Materials 5, 123803 (2021).
doi:10.1103/PhysRevMaterials.5.123803

[3] A. Jain, S.P. Ong, G. Hautier, W. Chen, W.D. Richards, S. Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, and K.A. Persson
The Materials Project: A materials genome approach to accelerating materials innovation
APL Materials 1, 011002 (2013).
doi:10.1063/1.4812323
```
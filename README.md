# LVNC_MasterRepo_SalvaDeHaro

## Overview

This repository gathers all materials developed by Salvador de Haro throughout his academic trajectory as a student, researcher, and doctoral candidate. The project focuses on the development, evaluation, and documentation of deep learning methodologies for the automatic segmentation of magnetic resonance imaging (MRI) slices, aiming to support computer-assisted diagnosis of **Left Ventricular Non-Compaction (LVNC)**.

The repository is designed to ensure reproducibility, transparency, and structured documentation of experiments, models, and results associated with this research line.

---

## Objectives

The main goals of this project are:

- Development of robust deep learning architectures for medical image segmentation.
- Implementation of reproducible pipelines for MRI preprocessing and annotation.
- Quantitative and qualitative evaluation of model performance.
- Integration of experimental results into scientific documentation and reports prepared in Overleaf/LaTeX environments.
- Facilitation of collaborative research and future extensions of the work.

---

## Methodology

The project follows a structured pipeline composed of:

### 1. Data Acquisition and Preprocessing
- MRI slice normalization.
- Noise filtering and contrast enhancement.
- Data augmentation strategies.

### 2. Model Development
- Implementation of convolutional neural networks tailored for biomedical segmentation.
- Exploration of encoder-decoder architectures.
- Hyperparameter optimization.

### 3. Training and Validation
- Cross-validation and performance benchmarking.
- Monitoring through loss functions and segmentation metrics.

### 4. Evaluation
- Dice coefficient.
- Intersection over Union (IoU).
- Clinical relevance assessment.

### 5. Scientific Documentation
- Automated integration of results into LaTeX documents.
- Reproducible compilation workflows compatible with Overleaf.

---

## Reproducibility

To ensure reproducibility, all experiments are version-controlled and linked to specific configuration files. The repository provides:

- Fully defined environments.
- Deterministic training options.
- Traceable experiment logging.

---

## 📂 Repository Structure

. <br>
├── GACOP_Scripts <br>
├── LVNC_ViTUNet <br>
├── LVNC_datasets <br>
└── README.md <br>

---

## 🧠 GACOP_Scripts

This directory contains the main scripts required to run the full project pipeline.

It includes tools for:

### ✔ Dataset Creation
- Generation of preprocessed datasets.
- Quality filtering.
- Data preparation for training and evaluation.

### ✔ Training
- Scripts for training ViTUNet models.
- Hyperparameter configuration.
- Training pipeline management.

### ✔ Evaluation
- Quantitative evaluation of trained models.
- Performance metrics computation.
- Results and validation generation.

---

## 🏗 LVNC_ViTUNet

This module contains the full implementation of the model and its components.

It includes:

- ViTUNet architecture definition  
- Training modules  
- Evaluation functions  
- Utility functions  
- Dataset handling within the model  
- Model weight loading and saving  
- Experiment configuration  

This directory represents the core of the model development.

---

## 🗂 LVNC_datasets

This directory provides access to the datasets required to reproduce experiments.

It contains links to download:

### 📌 Full Dataset
Includes:
- Complete RWA dataset  
- Original unfiltered slices  

### 📌 Filtered Dataset
Includes:
- Removal of low-quality slices  
- Visual enhancement applied  
- Dataset optimized for model training  

---

## Installation

Clone the repository:

```bash
git clone https://github.com/salvadeharo10/LVNC_MasterRepo_SalvaDeHaro.git
cd LVNC_MasterRepo_SalvaDeHaro

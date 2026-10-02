# PCM-BTMS ML Analysis

Machine learning-based thermal performance analysis of a Phase Change Material
Battery Thermal Management System (PCM-BTMS).

## Project Overview

This project investigates the relationship between PCM properties, battery
discharge conditions, heat-generation-related parameters, and thermal
performance using machine learning and statistical analysis.

## Input Parameters

- C_rate – Battery discharge rate
- Tm – PCM melting temperature
- rho – PCM density
- cp – Specific heat capacity
- k – Thermal conductivity
- Lf – Latent heat
- Aavg – Average volumetric heat-generation-related input

## Thermal Outputs

- Tmax – Maximum temperature
- dTmax – Maximum temperature difference
- phi_avg – Average liquid fraction

## Methods

The project includes:

- Data preprocessing and normalization
- Kendall rank correlation analysis
- Scatter plot analysis
- TOPSIS-based multi-criteria analysis
- Clustering analysis
- Back Propagation Neural Network (BPNN) modeling
- Model performance evaluation using R², RMSE and MAE

## BPNN Performance

The current model achieved:

| Output | R² | RMSE | MAE |
|---|---:|---:|---:|
| Tmax | 0.9121 | 0.8476 | 0.6892 |
| dTmax | 0.7923 | 0.5227 | 0.4460 |
| phi_avg | 0.9616 | 0.0222 | 0.0176 |

## Project Structure

```text
PCM-BTMS-ML-Analysis/
│
├── data/
├── prepare_data.py
├── kendall.py
├── scatter.py
├── topsis.py
├── clustering.py
├── bpnn_model.py
├── bpnn_results.py
├── dtmax.py
│
├── requirements.txt
├── README.md
│
└── results/
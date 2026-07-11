{\rtf1\ansi\ansicpg1252\cocoartf2868
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 ArialMT;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\paperw11900\paperh16840\margl1440\margr1440\vieww38200\viewh21600\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs28 \cf0 # PM2.5 Forecasting using Gaussian Process Regression\
\
**Author:** Rachadawan Dalai  \
**Affiliation:** School of Mathematics, Statistics and Physics, Newcastle University\
**Date:** 2026-04-02\
\
---\
\
## Overview\
This repository contains the dataset, code, and instructions for forecasting PM2.5 concentrations in Newcastle city centre, Thailand using Gaussian Process Regression (GPR). The project aims to provide reproducible results for air pollution forecasting and data analysis.\
\
---\
\
\pard\pardeftab720\partightenfactor0
\cf0 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 ## Files\
- `pm25_data.csv` \'96 The raw PM2.5 measurements and related features. \
- `gp_forecast.ipynb` \'96 Jupyter Notebook implementing the batch Gaussian Process forecasting pipeline. \
- `results_forecast.csv` \'96 Optional summary CSV of forecasts and evaluation metrics (if exported from the notebook). \
- `gp_results_update/` \'96 Folder containing:\
- `.npz` files with GP predictions (train/test mean and variance)\
- `.pkl` files with kernel parameters, valid/skipped days, and model metadata\
- diagnostic plots (if generated in notebook)\
- `README.md` \'96 This file with project description and usage instructions.\
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0
\cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 \
\
---\
\
## Usage\
1. Load the CSV dataset in Python using pandas.\
2. Open `gp_forecast.ipynb` to explore the code for training and forecasting PM2.5.\
3. Modify parameters or time windows as needed.\
4. Run the notebook step-by-step for reproducible forecasts.\
\
```python\
import pandas as pd\
df = pd.read_csv('pm25_data.csv')\
\
**Note:** This project requires Python 3.x and a Jupyter Notebook environment to run `gp_forecast.ipynb`.}
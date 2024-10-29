# Electricity-Market-Price-Forecasting
This repository is for the electricity forecasting via deep learning methods
## Overview
This project aims to develop a robust forecasting model for electricity market prices using advanced machine learning techniques. By leveraging historical price data, the model predicts future electricity prices, assisting stakeholders in making informed decisions.
## Table of Contents
- Insatllation
- Usage
- Data
- Model Architecture
- Results
## Installation
To set up the project, clone the repository and install the required dependencies:
```bash
git clone https://github.com/hejinhua1/Electricity-Market-Price-Forecasting.git
cd Electricity-Market-Price-Forecasting
pip install -r requirements.txt
```
## Usage
To train the model, run the following command:
```bash
python diffusion_sde.py
```
To evaluate the model, run the following command:
```bash
python sde_sampling.py
```
## Data
The dataset used in this project is the real-world electricity market price data from GanShu Province, China. The dataset contains 15-min electricity prices in 2024.
The label column is the electricity price, namely 'dayahead_clearing_price', and the columns 'wind_power_forecast', 'photovoltaic_power_forecast' and 'realtime_clearing_price' 
are not used.
## Model Architecture
TODO: Add model architecture diagram
## Results
TODO: Add results and visualizations
```

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
To set up the project, install python=3.8, clone the repository and install the required dependencies:
```bash
git clone https://github.com/hejinhua1/Electricity-Market-Price-Forecasting.git
cd Electricity-Market-Price-Forecasting
pip install -r requirements.txt
```
## Usage
We have provided scripts for training and evaluating the models for selected. As for now, we have implemented the following models:
- TimesNet
- TimeXer
- Informer
- Autoformer
- Nonstationary_Transformer

To train the model respectively, run the following command:
```bash
python diffusion_sde.py
```
To evaluate the model, run the following command:
```bash
bash ./scripts/long_term_forecast/ElcPrice_script/TimesNet_ElcPrice.sh
bash ./scripts/long_term_forecast/ElcPrice_script/TimeXer_ElcPrice.sh
bash ./scripts/long_term_forecast/ElcPrice_script/Informer_ElcPrice.sh
bash ./scripts/long_term_forecast/ElcPrice_script/Autoformer_ElcPrice.sh
bash ./scripts/long_term_forecast/ElcPrice_script/Nonstationary_Transformer_ElcPrice.sh
```
## Data
The dataset used in this project is the real-world electricity market price data from GanShu Province, China. The dataset contains 15-min electricity prices in 2024.
The label column is the electricity price, namely 'dayahead_clearing_price', and the columns 'wind_power_forecast', 'photovoltaic_power_forecast' and 'realtime_clearing_price' 
are not used.
For the data preprocessing, we have implemented the following steps:
- Data in the full_data.feather file is cleaned and preprocessed.
- In the model, we have added the temporal features, such as the hour of the day, day of the week, and month of the year.

For the model parameters, you should modify the following parameters in the config file:
- data_path: the path to the data file
- model_name: the name of the model
- seq_len: historical sequence length 
- pred_len: prediction length
- enc_in: the raw number of input features, default is 8
- dec_in: the raw number of output features, default is 8
- c_out: the number of output features, default is 8, but you can set up the 'feature' parameter in the config file to select the feature you want to predict
- feature: the feature you want to predict, default is 'MS', meaning just predict the electricity price
## Model Architecture
TODO: Add model architecture diagram
## Results
TODO: Add results and visualizations
```

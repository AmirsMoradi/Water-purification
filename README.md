Project Introduction

This project has been developed with the aim of predicting the qualitative and quantitative indicators of a wastewater treatment plant.
The models attempt to predict the future outputs of the treatment plant (climate scenarios) based on historical input data + climatic parameters (temperature, precipitation).


Main objectives:

Modeling input parameters (Stage 1: COD_in, BOD_in, TSS_in, …)

Modeling output parameters of the treatment plant (Stage 2: COD_out, BOD_out, TSS_out, …)

Investigating the effect of climate (rainfall and temperature) on the performance of the treatment plant

Forecasting based on future climate scenarios (SSP1-2.6, SSP2-4.5, SSP5-8.5


Implementation Steps
Step 1: Historical Data

The cleaned_historical.csv file contains:

Year (year), Month (month)

Climate parameters (temp_air, rainfall)

WWTP inputs (temp_wastewater_in, bod_in, cod_in, …)

WWTP outputs (bod_out, cod_out, tss_out, …)

First, the data is cleaned:

Remove/replace NaN, inf

Normalize inputs (StandardScaler)





Stage 2: Modeling Stage 1 (Inputs)

Input: Climate only (temperature and precipitation)

Output: WWTP input parameters (BOD_in, COD_in, …)

Algorithms:

XGBoost (stable default)

MLP (ANN) for comparison

Output:

Stage1_CV_Metrics.csv → Cross Validation results (R², RMSE, MAE)

Charts: parity plots, residual histograms, time-series


Stage 3: Stage 2 Modeling (Outputs)

Input:

Climate Parameters

Actual Plant Inputs

Stage 1 Predictions

Output: Plant Output Parameters (BOD_out, COD_out, TSS_out, …)

Same Output as Stage 1:

Stage2_CV_Metrics.csv

Different Graphs

Step 4: Descriptive Stats

Calculate minimum, maximum, mean, median, standard deviation and coefficient of variation (%)

For all input, output and climate parameters

Save to file: Final_Stats_Historical.csv

Step 5: Climate Scenarios

Using the cleaned_future.csv file for three climate scenarios:

SSP1-2.6 (low risk)

SSP2-4.5 (moderate)

SSP5-8.5 (high pressure)

Outputs:

Final_Stats_ssp126.csv, Final_Stats_ssp245.csv, Final_Stats_ssp585.csv

Predicted input parameters (predicted_IN_...csv)

Predicted output parameters (predicted_OUT_...csv)


Charts

Parity plots: Compare actual and predicted values

Residual histograms: Check errors

Time-series plots: Show the trend of predictions

Feature importance (XGBoost): Check the importance of parameters

Permutation importance (ANN): The importance of features

Requirements
python 3.9+
pandas
numpy
matplotlib
scikit-learn
xgboost


Conclusion

XGBoost models performed more consistently and accurately than ANN.

Climate scenarios play an important role in changing water quality, but accurate prediction requires more time (lag/rolling) features.

This project provides a complete framework for predicting and analyzing climate impacts on wastewater treatment plants.






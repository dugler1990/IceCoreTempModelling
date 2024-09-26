#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 02:49:05 2024

@author: fresh
"""

## Icecore data ref:
#https://doi.pangaea.de/10.1594/PANGAEA.824894

## Oxygen isotopes ref
#https://doi.pangaea.de/10.1594/PANGAEA.934094

## Sulpher dataset:
#https://doi.pangaea.de/10.1594/PANGAEA.933271?format=html#download

## Vulcanic reconstruction
# https://doi.pangaea.de/10.1594/PANGAEA.601894?format=html#download

# this mainly just looks at basic diagnostics and timeseries analysis of geo temp
# also looks at milankovic cycles in similar vein
# final analysis is meant to prove relationship between milankovic cycles and volatility

# Basic conclusions
## 1. Show current "stability" of the holoscene is not undeniable ( many previous periods
#  of similar or more stability over 11.7k years)
## 2. GARCH type models show importance of far past on current volatility
## 3. process to show milancovik cycles also affect volatility ( all expect omega from this anlysis )

import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv( "EPICA-DOME-C-ice-core_dD_d180.tab", delimiter="\t" )
depth_age = pd.read_csv("depth_age.tab", delimiter = "\t")


# Resampling age to get a consistent time frame

data.reset_index()

milankovitch_1 = pd.read_csv("Milankovic1991_1.tab", delim_whitespace=True, nrows = 500)
milankovitch_2 = pd.read_csv("Milankovic1991_2.tab", delim_whitespace=True, nrows = 500)
milankovitch_3 = pd.read_csv("Milankovic1991_3.tab", delim_whitespace=True, nrows = 500)
milankovitch_2 = milankovitch_2.reset_index()
milankovitch_3 = milankovitch_3.reset_index()

# co2_ice = pd.read_csv("Composite_CO2_AICC2012_chron.tab", delimiter = "\t")
# ch4_ice = pd.read_csv("Vostok_CH4_AICC2012_chron.tab", delimiter = "\t")
# so2_ice = pd.read_csv("EPICA_sulfur_isotops_eruptions.tab", delimiter="\t")


# TODO: i need to this adjustment tbh : 
# TF:
# This could stand for "Thinning Factor," which is used to account for the compression of deeper ice layers over time.
# LIDIE [m]:
# This likely stands for "Lock-In Depth in Ice Equivalent." It represents the depth at which air bubbles become trapped in the ice, converted to its ice-equivalent depth.




#### annoying debug, 

milankovitch = pd.concat([milankovitch_1,milankovitch_2,milankovitch_3],axis = 1)

milankovitch["AGE"] = milankovitch["AGE"] * -1 
#milankovitch[]
# Create an interpolation functions
# depth to age interpolation
f = interpolate.interp1d(depth_age['Depth ice/snow [m]'], depth_age['Age [ka BP]'], 
                         kind='linear', fill_value='extrapolate')

data['Estimated Age [ka BP]'] = f(data['Depth ice/snow [m]'])

# interpolation for all Milan values to age :  ECC       OMEGA   OBL       PREC 
milan_params = ['ECC', 'OMEGA', 'OBL', 'PREC',"90SJune","90SDec"]
for i in milan_params:
    
    f = interpolate.interp1d( milankovitch["AGE"], milankovitch[i] )
    
    data[f"Milan_{i}"] = f(data['Estimated Age [ka BP]'])





data["delta-iso18o-Base at 6.6m"] = data["δ18O H2O [‰ SMOW]"] - data["δ18O H2O [‰ SMOW]"][0:10].mean()
data["0.6*iso18o-Base at 6.6m"] = data["delta-iso18o-Base at 6.6m"] * 0.6
data["0.7*iso18o-Base at 6.6m"] = data["delta-iso18o-Base at 6.6m"] * 0.7
data["0.4*iso18o-Base at 6.6m"] = data["delta-iso18o-Base at 6.6m"] * 0.4
#data["0.4*iso18o-Base at 6.6m"].plot(x = data["Estimated Age [ka BP]"], y = data["0.4*iso18o-Base at 6.6m"] )
#data["0.7*iso18o-Base at 6.6m"].plot(x = data["Estimated Age [ka BP]"], y = data["0.7*iso18o-Base at 6.6m"] )


# Function to calculate rolling variance
def rolling_variance(x, window):
    return x.rolling(window=window).var()

# Define window size (in data points, adjust based on your data resolution)
window_size = 10  # Adjust this value based on your data resolution

# Calculate rolling variance of temperature
data['Temp_Variance_10'] = rolling_variance(data["0.7*iso18o-Base at 6.6m"], window_size)
# Shift the variance to align with the correct time points
data['Temp_Variance_10'] = data['Temp_Variance_10'].shift(-(window_size - 1))

# Define window size (in data points, adjust based on your data resolution)
window_size = 20  # Adjust this value based on your data resolution

# Calculate rolling variance of temperature
data['Temp_Variance_20'] = rolling_variance(data["0.7*iso18o-Base at 6.6m"], window_size)
# Shift the variance to align with the correct time points
data['Temp_Variance_20'] = data['Temp_Variance_20'].shift(-(window_size - 1))


# Define window size (in data points, adjust based on your data resolution)
window_size = 40  # Adjust this value based on your data resolution

# Calculate rolling variance of temperature
data['Temp_Variance_40'] = rolling_variance(data["0.7*iso18o-Base at 6.6m"], window_size)
# Shift the variance to align with the correct time points
data['Temp_Variance_40'] = data['Temp_Variance_40'].shift(-(window_size - 1))


# Define window size (in data points, adjust based on your data resolution)
window_size = 80  # Adjust this value based on your data resolution

# Calculate rolling variance of temperature
data['Temp_Variance_80'] = rolling_variance(data["0.7*iso18o-Base at 6.6m"], window_size)
# Shift the variance to align with the correct time points
data['Temp_Variance_80'] = data['Temp_Variance_80'].shift(-(window_size - 1))


### Chunk variance


# Define the chunk size (Holocene length)
chunk_size = 1.17  # ka

# Function to calculate variance for a chunk
def chunk_variance(group):
    return group["0.7*iso18o-Base at 6.6m"].var()

# Create a new column for chunk number
data['Chunk'] = (data['Estimated Age [ka BP]'] // chunk_size).astype(int)

# Calculate variance for each chunk
chunk_variances = data.groupby('Chunk').apply(chunk_variance).reset_index()
chunk_variances.columns = ['Chunk', 'Chunk_Variance']

# Merge chunk variances back into the original dataframe
data = data.merge(chunk_variances, on='Chunk', how='left')



plt.figure(figsize=(10, 6))
plt.plot(data["Estimated Age [ka BP]"],  data["0.7*iso18o-Base at 6.6m"] , label='0.7*iso18o-Base at 6.6m', color='blue')

# Set finer-grain ticks on x-axis
min_age = data["Estimated Age [ka BP]"].min()
max_age = data["Estimated Age [ka BP]"].max()
plt.xticks(np.arange(min_age, max_age, 20))  # Adjust the step as needed

plt.show()

### Multiplot attempt : 
    
    

fig, axs = plt.subplots(12, 1, figsize=(15, 20), sharex=True)

# Plot temperature estimate
axs[0].plot(data["Estimated Age [ka BP]"], data["0.7*iso18o-Base at 6.6m"], label='Temperature Estimate', color='blue')
axs[0].set_ylabel('Temperature Change [°C]')
axs[0].legend()
axs[0].set_title('EPICA Dome C Temperature Reconstruction and Milankovitch Cycles')

# Plot Milankovitch parameters

colors = ['blue', 'green', 'purple','orange','red','green']

for i, param in enumerate(milan_params):
    
    axs[i+1].plot(data["Estimated Age [ka BP]"], data[f"Milan_{param}"], label=param, color=colors[i])
    axs[i+1].set_ylabel(param)
    axs[i+1].legend()

axs[i+2].plot(data["Estimated Age [ka BP]"], data[f"Temp_Variance_10"], label="Temp_Variance_10", color='black')
axs[i+2].set_ylabel("Temp_Variance_10")
axs[i+2].legend()

axs[i+3].plot(data["Estimated Age [ka BP]"], data[f"Temp_Variance_20"], label="Temp_Variance_20", color='black')
axs[i+3].set_ylabel("Temp_Variance_20")
axs[i+3].legend()

axs[i+4].plot(data["Estimated Age [ka BP]"], data[f"Temp_Variance_40"], label="Temp_Variance_30", color='black')
axs[i+4].set_ylabel("Temp_Variance_40")
axs[i+4].legend()

axs[i+5].plot(data["Estimated Age [ka BP]"], data[f"Temp_Variance_80"], label="Temp_Variance_80", color='black')
axs[i+5].set_ylabel("Temp_Variance_80")
axs[i+5].legend()

axs[i+6].plot(data["Estimated Age [ka BP]"], data[f"Chunk_Variance"], label="Chunk_Variance", color='black')
# Add a horizontal line at the mean value
# Calculate the mean of the Chunk_Variance data
mean_chunk_variance = data[f"Chunk_Variance"].mean()
axs[i+6].axhline(y=mean_chunk_variance, color='red', linestyle='--', label='Mean Line')
axs[i+6].set_ylabel("Chunk_Variance")
axs[i+6].legend()

# Set x-axis label and ticks
axs[-1].set_xlabel('Age [ka BP]')
min_age = data["Estimated Age [ka BP]"].min()
max_age = data["Estimated Age [ka BP]"].max()
axs[-1].set_xticks(np.arange(min_age, max_age, 20))  # Adjust the step as needed
axs[-1].invert_xaxis()  # Invert x-axis to have older ages on the right

# Adjust layout and display
plt.tight_layout()
plt.show()


correlations = data.corr()



# δD to Temperature Conversion
# For the EPICA Dome C site, the relationship between δD and temperature (T) can be approximated by:
# �
# =
# 0.16289
# ×
# �
# �
# +
# 55.298
# T=0.16289×δD+55.298
# Where:
# T is the temperature in Kelvin (K)
# δD is the deuterium isotope ratio in ‰ SMOW
# δ18O to Temperature Conversion
# Similarly, for δ18O, we can use:
# �
# =
# 1.2815
# ×
# �
# 18
# �
# +
# 54.759
# T=1.2815×δ 
# 18
#  O+54.759
# Converting to Celsius: -9.316 - 273.15 = -282.47 °C




# ΔT is the temperature change
# Δδ is the change in isotopic ratio (either δD or δ18O)
# α is a calibration coefficient (often around 0.6-0.7 for δ18O in Antarctica)

# Based on your request, I'll focus on the best sources for isotope-temperature relationships and how to use relative temperature changes in your analysis. Here are some key sources and approaches:
# Modern Observational Studies:
# The Global Network of Isotopes in Precipitation (GNIP) database, maintained by the International Atomic Energy Agency (IAEA), is an excellent source for modern isotope-temperature relationships. It provides data on isotopic composition of precipitation along with corresponding temperature measurements from various locations worldwide.
# Source: IAEA/WMO (2023). Global Network of Isotopes in Precipitation. The GNIP Database. Accessible at: https://nucleus.iaea.org/wiser
# Antarctic-specific Studies:
# For Antarctic ice cores, the following studies provide valuable insights:
# Masson-Delmotte, V., et al. (2008). A Review of Antarctic Surface Snow Isotopic Composition: Observations, Atmospheric Circulation, and Isotopic Modeling. Journal of Climate, 21(13), 3359-3387.
# This paper reviews the relationship between isotopic composition and temperature in Antarctic snow, which is directly relevant to ice core interpretations.
# Controlled Experiments:
# Merlivat, L., & Jouzel, J. (1979). Global climatic interpretation of the deuterium‐oxygen 18 relationship for precipitation. Journal of Geophysical Research: Oceans, 84(C8), 5029-5033.




# looking at the correlations, something seems off, climate forcing or watvr has
# negative correlation to the temperature.....

# Eccentricity (ECC): should only affect variance (more elipse, more variance in distance)
#                                                  i guess we are only talking forcing in jun so far.
# OBL again only affects seasonally (tilt) and only at poles
# again only variability, not absolute, 







# Full data here : https://www.ncei.noaa.gov/pub/data/paleo/climate_forcing/orbital_variations/insolation/
## Milankovic data readme: 1971 , not used ( milankovic details only at 100k granularity)
    
#     Format for the BEIN files, each containing 100 kyr of data:

# Line 1:          Age in Kyears (0=1950 A.D.),  Eccentricity, Longitude of
#                  Perihelion, Obliquity, Precession.

# Lines 2 to 20:   Age, Latitude, Mid-month insolations for January to December.


#                  - Age is in thousands of years, negative in the past.  
#                  - Latitude is in degrees, positive in the northern hemisphere.
#                  - Insolations are in langleys/day (cal/cm2/day). Multiply
#                    by .4843 to convert to Watts/m2
#                  - Solar constant taken as 1.95 cal/cm2/min for 1978 solution,
#                    and 1360 W/m2 for the 1991 solution.
#                  - For other definitions, see papers listed in file CITATION. 

# 1991 file description : 
    
# Contents of 1991 files:

# 1. File ORBIT91: 0-5 Myr B.P.
#     . first column: time in kyr (negative for the past; origin (0) 
#                                  is 1950 A.D.)
#     . second column: eccentricity, ECC 
#     . third col: longitude of perihelion from moving vernal equinox 
#       in degrees, OMEGA
#     . fourth column: obliquity in degree and decimals, OBL 
#     . fifth column: climatic precession, ECC . SIN(OMEGA) 
#     . sixth column: mid-month insolation 65N for July in W/m**2 
#     . seventh column: mid-month insolation 65S for January in W/m**2 
#     . eighth column: mid-month insolation 15N for July in W/m**2
#     . ninth column: mid-month insolation 15S for January in W/m**2 
#  2. File INSOL91.DEC: 0-1 Myr B.P.
#     . first column: time in kyr (negative for the past; origin (0) 
#       is 1950 A.D.)
#     . second to eighth: Dec mid-month insolation 90N, 60N, 30N, 0, 
#       30S, 60S, 90S, W/m2
#  3. File INSOL91.JUN: 0-1 Myr B.P.
#     . first column: time in kyr (negative for the past; origin (0) 
#       is 1950 A.D.)
#     . second to eighth: June mid-month insolation 90N, 60N, 30N, 0, 
#       30S, 60S, 90S, W/m2





#### I need to check relayshe of both jun and jan in order to get the effect,
#    effect of increase in cycle means less hot Summer it seems, im guessing more
#    hot winter  ?


########### TODO: 
    #1.USE THE CORRECT LATITUDE ! 90 ? 
    #2.Check rolling variability
    #3. get mean and variance of glacial to interglacial cycle, 
    # try to push the idea that the cycles get longer, explaining the 
      # holoscene as just longer interglacial.




# Volatility analysis 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame with temperature data

# Calculate temperature changes
data['Temp_Change'] = data["0.7*iso18o-Base at 6.6m"].diff()

# Calculate rolling volatility (standard deviation)
window = 10  # Adjust as needed
data['Rolling_Volatility'] = data["0.7*iso18o-Base at 6.6m"].rolling(window=window).std()

# Plot temperature and rolling volatility
plt.figure(figsize=(15, 10))
plt.plot(data['Estimated Age [ka BP]'], data["0.7*iso18o-Base at 6.6m"], label='Temperature', alpha=0.7)
plt.plot(data['Estimated Age [ka BP]'], data['Rolling_Volatility'], label='Rolling Volatility', linewidth=2)
plt.xlabel('Age [ka BP]')
plt.ylabel('Temperature / Volatility')
plt.title('Temperature and Rolling Volatility Over Time')
plt.legend()
plt.gca().invert_xaxis()
plt.grid(True)
plt.show()

#
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Autocorrelation and Partial Autocorrelation of Temperature
plt.figure(figsize=(15, 5))
plot_acf(data["0.7*iso18o-Base at 6.6m"].dropna(), lags=40, title='Autocorrelation of Temperature')
plt.show()

plt.figure(figsize=(15, 5))
plot_pacf(data["0.7*iso18o-Base at 6.6m"].dropna(), lags=40, title='Partial Autocorrelation of Temperature')
plt.show()

# Autocorrelation and Partial Autocorrelation of Volatility
plt.figure(figsize=(15, 5))
plot_acf(data['Rolling_Volatility'].dropna(), lags=40, title='Autocorrelation of Temperature Volatility')
plt.show()

plt.figure(figsize=(15, 5))
plot_pacf(data['Rolling_Volatility'].dropna(), lags=40, title='Partial Autocorrelation of Temperature Volatility')
plt.show()

# Arch Garch

from arch import arch_model

# Assuming 'data' is your DataFrame with temperature data

# Calculate temperature changes (returns)
data['Temp_Change'] = data["0.7*iso18o-Base at 6.6m"].diff().dropna()

# Drop NaN values resulting from the differencing
data = data.dropna(subset=['Temp_Change'])

# Fit an ARCH model
arch_model_fit = arch_model(data['Temp_Change'], vol='ARCH', p=1).fit(disp='off')
print("ARCH Model Summary:")
print(arch_model_fit.summary())

# Fit a GARCH model
garch_model_fit = arch_model(data['Temp_Change'], vol='Garch', p=1, q=1).fit(disp='off')
print("\nGARCH Model Summary:")
print(garch_model_fit.summary())

# Plot the conditional volatility from the GARCH model
plt.figure(figsize=(15, 10))
plt.plot(data['Estimated Age [ka BP]'], garch_model_fit.conditional_volatility, label='Conditional Volatility (GARCH)', linewidth=2)
plt.xlabel('Age [ka BP]')
plt.ylabel('Conditional Volatility')
plt.title('Conditional Volatility Over Time (GARCH Model)')
plt.legend()
plt.gca().invert_xaxis()
plt.grid(True)
plt.show()


# ACF PACF analysis

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot ACF and PACF of standardized residuals from GARCH model
plt.figure(figsize=(15, 5))
plot_acf(garch_model_fit.std_resid, lags=40, title='ACF of Standardized Residuals (GARCH Model)')
plt.show()

plt.figure(figsize=(15, 5))
plot_pacf(garch_model_fit.std_resid, lags=40, title='PACF of Standardized Residuals (GARCH Model)')
plt.show()


# Spectral analysis

from scipy.signal import periodogram

# Spectral analysis of Temperature
frequencies, power_spectrum = periodogram(data["0.7*iso18o-Base at 6.6m"].dropna(), fs=1.0)

plt.figure(figsize=(15, 5))
plt.plot(frequencies, power_spectrum)
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.title('Spectral Analysis of Temperature')
plt.grid(True)
plt.show()

# Spectral analysis of Volatility
frequencies_vol, power_spectrum_vol = periodogram(data['Rolling_Volatility'].dropna(), fs=1.0)

plt.figure(figsize=(15, 5))
plt.plot(frequencies_vol, power_spectrum_vol)
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.title('Spectral Analysis of Temperature Volatility')
plt.grid(True)
plt.show()


# Supposid analysis to prove stability of holoscene

import pandas as pd
import numpy as np

# Assuming 'data' is your DataFrame with temperature proxy and age data
holocene_data = data[data['Estimated Age [ka BP]'] <= 11.7]
pleistocene_data = data[data['Estimated Age [ka BP]'] > 11.7]

holocene_std = holocene_data["0.7*iso18o-Base at 6.6m"].std()
pleistocene_std = pleistocene_data["0.7*iso18o-Base at 6.6m"].std()

print(f"Holocene temperature variability: {holocene_std}")
print(f"Pleistocene temperature variability: {pleistocene_std}")
print(f"Ratio (Pleistocene/Holocene): {pleistocene_std / holocene_std}")

# Long-term trend analysis
window = 100  # Adjust based on your data resolution
data['Long_term_trend'] = data["0.7*iso18o-Base at 6.6m"].rolling(window=window, center=True).mean()

# Plot long-term trend
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
plt.plot(data['Estimated Age [ka BP]'], data["0.7*iso18o-Base at 6.6m"], label='Temperature Proxy', alpha=0.5)
plt.plot(data['Estimated Age [ka BP]'], data['Long_term_trend'], label='Long-term Trend', linewidth=2)
plt.axvline(x=11.7, color='r', linestyle='--', label='Start of Holocene')
plt.xlabel('Age [ka BP]')
plt.ylabel('Temperature Proxy')
plt.title('Long-term Temperature Trend')
plt.legend()
plt.gca().invert_xaxis()
plt.show()


## Further attempted AI debunking : 

# Assuming 'data' is your DataFrame with temperature proxy and age data

# Define the chunk size (Holocene length)
chunk_size = 11.7  # ka

# Create a new column for chunk number
data['Chunk'] = (data['Estimated Age [ka BP]'] // chunk_size).astype(int)

# Calculate variance for each chunk
chunk_variances = data.groupby('Chunk').apply(lambda group: group["0.7*iso18o-Base at 6.6m"].var()).reset_index()
chunk_variances.columns = ['Chunk', 'Chunk_Variance']

# Separate Holocene chunk
holocene_variance = chunk_variances[chunk_variances['Chunk'] == 0]['Chunk_Variance'].values[0]

# Plot chunk variances
plt.figure(figsize=(15, 10))
plt.plot(chunk_variances['Chunk'], chunk_variances['Chunk_Variance'], label='Pleistocene Chunks')
plt.axhline(y=holocene_variance, color='r', linestyle='--', label='Holocene Variance')
plt.xlabel('Chunk Number')
plt.ylabel('Temperature Variance')
plt.title('Temperature Variance in 11.7k-Year Chunks')
plt.legend()
plt.grid(True)
plt.show()

# Print chunk variances
print("Chunk Variances:")
print(chunk_variances)

# Compare Holocene variance with earlier periods
earlier_variance_mean = chunk_variances[chunk_variances['Chunk'] != 0]['Chunk_Variance'].mean()

print(f"\nHolocene Variance: {holocene_variance}")
print(f"Mean Variance of Earlier Periods: {earlier_variance_mean}")
print(f"Ratio (Earlier/Holocene): {earlier_variance_mean / holocene_variance}")

# Changepoint detection 
import ruptures as rpt

# Assuming 'data' is your DataFrame with temperature proxy and age data

# Detect change points in the temperature data using Binary Segmentation
algo = rpt.Binseg(model="l2").fit(data["0.7*iso18o-Base at 6.6m"].values)
result = algo.predict(n_bkps=10)  # Adjust number of breakpoints as needed
result.pop()# for some reason last value not working
# Plot temperature data with detected change points
plt.figure(figsize=(15, 10))
plt.plot(data['Estimated Age [ka BP]'], data["0.7*iso18o-Base at 6.6m"], label='Temperature')
for cp in result:
    plt.axvline(x=data['Estimated Age [ka BP]'].iloc[cp], color='r', linestyle='--')
plt.xlabel('Age [ka BP]')
plt.ylabel('Temperature Proxy')
plt.title('Detected Change Points in Temperature Data')
plt.legend()
plt.gca().invert_xaxis()
plt.grid(True)
plt.show()

#Integrate Milankovitch Cycles into Time Series Models

import statsmodels.api as sm

# Prepare the data for regression
X = data[['Milan_ECC', 'Milan_OBL', 'Milan_PREC']]
X = sm.add_constant(X)  # Add a constant term for the intercept
y = data["0.7*iso18o-Base at 6.6m"]

# Fit the regression model
model = sm.OLS(y, X).fit()
print(model.summary())

# Plot fitted values
data['Fitted_Temp'] = model.fittedvalues

plt.figure(figsize=(15, 10))
plt.plot(data['Estimated Age [ka BP]'], data["0.7*iso18o-Base at 6.6m"], label='Observed Temperature')
plt.plot(data['Estimated Age [ka BP]'], data['Fitted_Temp'], label='Fitted Temperature', linestyle='--')
plt.xlabel('Age [ka BP]')
plt.ylabel('Temperature Proxy')
plt.title('Observed vs. Fitted Temperature')
plt.legend()
plt.gca().invert_xaxis()
plt.grid(True)
plt.show()




### Revised arch model with t distribution and no constant mean assumption : 
    
from arch import arch_model

# ARMA(1,1)-GARCH(1,1) model with Student's t distribution
model = arch_model(data['Temp_Change'], mean='AR', lags=1, vol='GARCH', p=1, q=1, dist='t')
results = model.fit(disp='off')

print(results.summary())

# Plot the conditional volatility
plt.figure(figsize=(12, 6))
plt.plot(data.index, results.conditional_volatility)
plt.title('Conditional Volatility')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.show()

# Plot the standardized residuals
standardized_residuals = results.resid / results.conditional_volatility
plt.figure(figsize=(12, 6))
plt.plot(data.index, standardized_residuals)
plt.title('Standardized Residuals')
plt.xlabel('Time')
plt.ylabel('Standardized Residuals')
plt.show()

# Check for remaining ARCH effects
from statsmodels.stats.diagnostic import acorr_ljungbox
lbvalue, pvalue = acorr_ljungbox(standardized_residuals**2)
print(f"Ljung-Box test p-value: {pvalue[-1]}")



# GARCH p and q parameter evaluation.

import itertools
from arch import arch_model

def garch_order_select(data, p_max=2, q_max=5):
    best_aic = np.inf
    best_order = None
    
    for p, q in itertools.product(range(1, p_max+1), range(1, q_max+1)):
        try:
            model = arch_model(data, vol='Garch', p=p, q=q)
            results = model.fit(disp='off')
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = (p, q)
        except:
            continue
    
    return best_order

# Find the best order
best_p, best_q = garch_order_select(data['Temp_Change'])
print(f"Best GARCH order: p={best_p}, q={best_q}")

# Fit the best model
best_model = arch_model(data['Temp_Change'], vol='Garch', p=best_p, q=best_q).fit(disp='off')
print(best_model.summary())


# out of sample performance for p and q estimation : 
    
from arch import arch_model
from sklearn.model_selection import TimeSeriesSplit

def garch_order_select_oos(data, p_max=1, q_max=11, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for p, q in itertools.product(range(1, p_max+1), range(1, q_max+1)):
        mse_scores = []
        for train_index, test_index in tscv.split(data):
            train, test = data.iloc[train_index], data.iloc[test_index]
            try:
                model = arch_model(train, vol='Garch', p=p, q=q)
                res = model.fit(disp='off')
                forecasts = res.forecast(horizon=len(test))
                mse = np.mean((test - forecasts.variance.values[-1])**2)
                mse_scores.append(mse)
            except:
                mse_scores.append(np.inf)
        
        avg_mse = np.mean(mse_scores)
        results.append({'p': p, 'q': q, 'MSE': avg_mse})
    
    results_df = pd.DataFrame(results)
    best_model = results_df.loc[results_df['MSE'].idxmin()]
    return best_model['p'], best_model['q'], results_df

# Find the best order using out-of-sample performance
best_p, best_q, all_results = garch_order_select_oos(data['Temp_Change'])
print(f"Best GARCH order based on out-of-sample performance: p={best_p}, q={best_q}")

# Display all results sorted by MSE
print("\nAll model results sorted by MSE:")
print(all_results.sort_values('MSE'))

# Fit the best model
best_model = arch_model(data['Temp_Change'], vol='Garch', p=int(best_p), q=int(best_q)).fit(disp='off')
print("\nBest model summary:")
print(best_model.summary())

# Plot MSE for different p and q values
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(all_results['p'], all_results['q'], all_results['MSE'])
ax.set_xlabel('p')
ax.set_ylabel('q')
ax.set_zlabel('MSE')
ax.set_title('MSE for different GARCH(p,q) models')
plt.show()




best_model = arch_model(data['Temp_Change'], vol='Garch', p=3, q=4).fit(disp='off')
print("model summary:")
print(best_model.summary())


#analysis of increasing q and their value


# for i in range(7):
#     best_model = arch_model(data['Temp_Change'], vol='Garch', p=1, q=i).fit(disp='off')
#     print("model summary:")
#     print(best_model.summary())

# checking remaining resuduals for patters ( shold be iid )


from statsmodels.stats.diagnostic import acorr_ljungbox

# Assuming best_model is the fitted GARCH model from your previous analysis
# If not, fit the best model again
# best_model = arch_model(data['Temp_Change'], vol='Garch', p=best_p, q=best_q).fit(disp='off')

# Extract standardized residuals
standardized_residuals = best_model.resid / best_model.conditional_volatility

# Plot standardized residuals
plt.figure(figsize=(12, 6))
plt.plot(data.index, standardized_residuals)
plt.title('Standardized Residuals')
plt.xlabel('Time')
plt.ylabel('Standardized Residuals')
plt.show()

# Plot squared standardized residuals
plt.figure(figsize=(12, 6))
plt.plot(data.index, standardized_residuals**2)
plt.title('Squared Standardized Residuals')
plt.xlabel('Time')
plt.ylabel('Squared Standardized Residuals')
plt.show()

# Perform Ljung-Box test on residuals
ljung_box_resid = acorr_ljungbox(standardized_residuals, lags=[10, 20, 30], return_df=True)
print("Ljung-Box test on residuals:")
print(ljung_box_resid)

# Perform Ljung-Box test on squared residuals
ljung_box_squared_resid = acorr_ljungbox(standardized_residuals**2, lags=[10, 20, 30], return_df=True)
print("\nLjung-Box test on squared residuals:")
print(ljung_box_squared_resid)



# Egarch model ( allows for assymetric affects of positive and negative volatility )
# GJR Garch does the same with a new parameter.
# for i, j in itertools.product([1,2],[1,2,3,4,5]):
#     egarch_model = arch_model(data['Temp_Change'], vol='EGARCH', p=i, q=j).fit(disp='off')
#     print("EGARCH Model Summary:")
#     print(egarch_model.summary())
    
    # # Fit a GJR-GARCH model
    # gjr_garch_model = arch_model(data['Temp_Change'], vol='GARCH', p=i, o=1, q=j).fit(disp='off')
    # print("\nGJR-GARCH Model Summary:")
    # print(gjr_garch_model.summary())
    

#  Non-linear Dependence
# For non-linear time series models, you might consider models like the Threshold Autoregressive (TAR) model or the Smooth Transition Autoregressive (STAR) model. These models capture non-linear dependencies that GARCH models might miss.
# Here's a basic example of how to fit a non-linear model using the statsmodels library:

import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# Fit a Markov Switching model
ms_model = MarkovRegression(data['Temp_Change'], k_regimes=2, trend='c', switching_variance=True).fit()
print("\nMarkov Switching Model Summary:")
print(ms_model.summary())



# FIGARCH (Fractionally Integrated GARCH) models are used to capture long memory in volatility.
# Here's how to fit a FIGARCH model using the arch library:


# Fit a FIGARCH model

best_model = arch_model(data['Temp_Change'], vol='FIGARCH', p=1, q=1).fit(disp='off')
print("\nFIGARCH Model Summary:")
print(best_model.summary())


# TODO:  Could use the ruptures structural breaks idea to model inter glacial glacial and then
#   transition ? periods hmmm

# Time-Varying Parameters
# Models with time-varying parameters can capture changing dynamics over time. One approach is to use state-space models.
# Here's an example using the statsmodels library:

import statsmodels.api as sm

# Fit a state-space model with time-varying parameters
mod = sm.tsa.UnobservedComponents(data['Temp_Change'], level='local level', stochastic_level=True)
res = mod.fit()
print("\nState-Space Model Summary:")
print(res.summary())

# Plot the results
res.plot_components()
plt.show()


## playing with figarch powers


# Fit a standard FIGARCH model (power=2.0)
figarch_model_2 = arch_model(data['Temp_Change'], vol='FIGARCH', p=1, q=1, power=2.0).fit(disp='off')
print("\nStandard FIGARCH Model (Power=2.0) Summary:")
print(figarch_model_2.summary())

# Fit a FIAVARCH model (power=1.0)
fiavarch_model = arch_model(data['Temp_Change'], vol='FIGARCH', p=1, q=1, power=1.0).fit(disp='off')
print("\nFIAVARCH Model (Power=1.0) Summary:")
print(fiavarch_model.summary())

# Fit a FIGARCH model with a higher power (e.g., power=10.0)
figarch_model_10 = arch_model(data['Temp_Change'], vol='FIGARCH', p=1, q=1, power=10).fit(disp='off')
print("\nFIGARCH Model (Power=10.0) Summary:")
print(figarch_model_10.summary())








### Fit GARCH with exogonous variables:
    
    # Assuming 'milankovitch_cycles' is a DataFrame with the external variables
exog = data.filter(['Milan_ECC', 'Milan_OMEGA', 'Milan_OBL','Milan_PREC'])

# Fit a GARCH model with exogenous variables
best_model = arch_model(data['Temp_Change'],mean='Constant', vol='Garch', p=1, q=1, x=exog).fit(disp='off')
print("\nGARCH Model with Exogenous Variables Summary:")
print(best_model.summary())




### Attempt at xgarch


from statsmodels.regression.linear_model import OLS

# Define the exogenous variables
exog = data[['Milan_ECC', 'Milan_OMEGA', 'Milan_OBL', 'Milan_PREC']]

# Add a constant term to the exogenous variables
exog_with_const = sm.add_constant(exog)

# Step 1: Regress temperature changes on Milankovitch cycles
model = sm.OLS(data['Temp_Change'], exog_with_const)
results = model.fit()
print("Regression Results:")
print(results.summary())

# Get residuals
residuals = results.resid

# Step 2: Fit GARCH model to residuals
garch_model = arch_model(residuals, vol='GARCH', p=1, q=1).fit(disp='off')
print("\nGARCH Model on Residuals Summary:")
print(garch_model.summary())

# Analyze the relationship between volatility and Milankovitch cycles
conditional_volatility = garch_model.conditional_volatility

# Regress conditional volatility on Milankovitch cycles
volatility_model = OLS(conditional_volatility, exog_with_const)
volatility_results = volatility_model.fit()
print("\nVolatility Regression Results:")
print(volatility_results.summary())

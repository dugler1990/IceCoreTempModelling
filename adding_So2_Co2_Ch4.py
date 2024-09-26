#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 12:05:47 2024

@author: fresh
"""


import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv( "EPICA-DOME-C-ice-core_dD_d180.tab", delimiter="\t" )
depth_age = pd.read_csv("depth_age.tab", delimiter = "\t")
milankovitch_1 = pd.read_csv("Milankovic1991_1.tab", delim_whitespace=True, nrows = 500)
milankovitch_2 = pd.read_csv("Milankovic1991_2.tab", delim_whitespace=True, nrows = 500)
milankovitch_3 = pd.read_csv("Milankovic1991_3.tab", delim_whitespace=True, nrows = 500)
milankovitch_2 = milankovitch_2.reset_index()
milankovitch_3 = milankovitch_3.reset_index()

co2_ice = pd.read_csv("Composite_CO2_AICC2012_chron.tab", delimiter = "\t")
ch4_ice = pd.read_csv("Vostok_CH4_AICC2012_chron.tab", delimiter = "\t")
so2_ice = pd.read_csv("EPICA_sulfur_isotops_eruptions.tab", delimiter="\t")

volcano = pd.read_csv("volcano.csv")
volcano.drop(["Volcano (OR GEOLOGIC AGE)","VEI", 'Deposit (OR GEOLOGIC EVENT)'], axis = 1)
age_cols = ['Oldest Age - Years BP', 'Ave Age -Years BP', 'Young Age - Years BP']
for col in age_cols:
    volcano[col] = volcano[col].astype(float,errors='ignore')
    
volcano = volcano.loc[ volcano['Ave Age -Years BP'] < 400000 ]
volcano = volcano.loc[ ~volcano["Volume – km3"].isna() ]

# Remove values that contain a question mark.

volcano = volcano.loc[ ~volcano["Volume – km3"].astype(str).str.contains('\?', na=False) ]

def remove_invalid_characters(value):
    value = str(value).replace("<","")
    value = str(value).replace(">","")
    
    value = value.strip()
    return value
    
def remove_ranges(value):
    value = str(value).split("-")[0] # go for minimum , conservative
    value = str(value).split("to")[0]
    return value
    
volcano["Volume – km3"] = volcano["Volume – km3"].apply(remove_invalid_characters)  
volcano["Volume – km3"] = volcano["Volume – km3"].apply(remove_ranges) 
volcano["Volume – km3"] = volcano["Volume – km3"].astype(float)
volcano = volcano.loc[ volcano["Volume – km3"] > 250 ]
volcano['Ave Age -Years BP'] = volcano['Ave Age -Years BP']/1000
volcano.reset_index()


# depth to age interpolation
f = interpolate.interp1d(depth_age['Depth ice/snow [m]'], depth_age['Age [ka BP]'], 
                         kind='linear', fill_value='extrapolate')

data['Estimated Age [ka BP]'] = f(data['Depth ice/snow [m]'])


def find_nearest_age(age, age_array):
    idx = (np.abs(age_array - age)).argmin()
    return age_array[idx]


# Step 3: Apply the function to map volcano ages to nearest temperature ages
volcano['nearest_temp_age'] = volcano['Ave Age -Years BP'].apply(lambda x: find_nearest_age(x, data['Estimated Age [ka BP]'].values))

data = pd.merge( data, volcano.filter(["nearest_temp_age","Volume – km3" ],axis=1), left_on="Estimated Age [ka BP]", right_on = "nearest_temp_age", how = 'left')
data["Volume – km3"] = data["Volume – km3"].fillna(0)


# interpolate co2 and ch4

import scipy

f = scipy.interpolate.interp1d(   co2_ice["Gas age [ka BP]"], co2_ice["CO2 [ppmv]"],
                                  bounds_error=False,  # Don't raise an error for out-of-bounds values
                                  fill_value=np.nan)
data['CO2'] = f(data['Estimated Age [ka BP]'])
 
f = scipy.interpolate.interp1d(   ch4_ice["Gas age [ka BP]"], ch4_ice["CH4 [ppbv]"],
                                  bounds_error=False,  # Don't raise an error for out-of-bounds values
                                  fill_value=np.nan)
data['CH4'] = f(data['Estimated Age [ka BP]'])

## Bizarrly, we do not have 0 depth for wither of these values, they only exist at
#  0.d and 3.6 respectively, 3.6 ! 3600 years ago starting point for ch4 wtf.

# temp change from 18o isotope proportion 
data["delta-iso18o-Base at 6.6m"] = data["δ18O H2O [‰ SMOW]"] - data["δ18O H2O [‰ SMOW]"][0:10].mean()
data["0.6*iso18o-Base at 6.6m"] = data["delta-iso18o-Base at 6.6m"] * 0.6
data["0.7*iso18o-Base at 6.6m"] = data["delta-iso18o-Base at 6.6m"] * 0.7


data['Temp_Change'] = data["0.7*iso18o-Base at 6.6m"].diff()


#### Adapting this analysis to see how vulcanic activity correlates with temperature
#    also doing x garch process on volcanic activity residuals
#    also doing straight ols on co2 ch4 volcanic (should add milans)

data= data.iloc[1:data.shape[0],:]
#data = data.loc[~data.Temp_Change.isna()] # dno why na's would be produced, check above
data = data.loc[~data["0.7*iso18o-Base at 6.6m"].isna()] # dno why na's would be produced, check above

### Attempt at xgarch


from statsmodels.regression.linear_model import OLS
from arch import arch_model
import statsmodels.api as sm
# Define the exogenous variables
exog = data[['Volume – km3']]

# Add a constant term to the exogenous variables
exog_with_const = sm.add_constant(exog)

# Step 1: Regress temperature changes on Milankovitch cycles
model = sm.OLS(data['0.7*iso18o-Base at 6.6m'], exog_with_const)
results = model.fit()
print("Regression Results:")
print(results.summary())

# Get residuals
residuals = results.resid

# Step 2: Fit GARCH model to residuals
garch_model = arch_model(residuals, vol='GARCH', p=1, q=1).fit(disp='off')
print("\nGARCH Model on Residuals Summary:")
print(garch_model.summary())

# Analyze the relationship between volatility and volcanic activity
conditional_volatility = garch_model.conditional_volatility

# Regress conditional volatility on Milankovitch cycles
volatility_model = OLS(conditional_volatility, exog_with_const)
volatility_results = volatility_model.fit()
print("\nVolatility Regression Results:")
print(volatility_results.summary())


temp = data.filter(["CO2","CH4","0.7*iso18o-Base at 6.6m"]).dropna()
exog = temp[["CO2","CH4"]].dropna()

# Add a constant term to the exogenous variables
exog_with_const = sm.add_constant(exog)

# Step 1: Regress temperature changes on Milankovitch cycles
model = sm.OLS(temp['0.7*iso18o-Base at 6.6m'], exog_with_const)
results = model.fit()
print("Regression Results:")
print(results.summary())




## Try addincorrelating vulcanic with co2 ch4
## try using rolling mean temp and temp vol

##







# will consider all greater than 300, for this reason, i will convert all ">" to just the value
# not ideal but i will largely just be using 0,1 volcano bigger than 300 happened

# too recent, only 2000 years.
# major volcanic_reconstruction = pd.read_csv( "EDML-B32-EDC96-link.tab", delim_whitespace=True )

# Further vulcanic activity : 
    # https://doi.pangaea.de/10.1594/PANGAEA.602127 # also recent
    
# table of volcanic events:
# https://web.archive.org/web/20100120021410/http://www.tetontectonics.org/Climate/Table_S1.pdf

# potential plans: aggregate total recorded volcanic activity over the 100 year period
#                  use only very large erruptions and see their impact

# Threshold i will use : 
    # To ensure a higher likelihood of detection regardless of the time of eruption within the
    # last 400,000 years, consider setting a threshold of around 200-300 km³. This corresponds
    # to larger VEI 7 eruptions and would help mitigate the bias towards more recent records.
    
    # Not from the reference but one valid resource may be this  : 
        # For example, the study by Wolff et al. (2023) discusses the challenges in
        # detecting older eruptions and the need to combine records from both poles to 
        # improve detection.


## Analysis so far has shown that 

# 1. current stability is not hugely significant given last 400k years ( need to quantify )
# 2. volatility is explained quite well ( need to quantify ) by past volatility.
#    TODO : -> do on chunks
# 3. Milancovik cycles would increase stability in this period (still need to quantify)
# 4. Milankovic cycles do correlate to volatility,  r2 0.26?
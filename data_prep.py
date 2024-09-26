
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import math

# epica c 18o proportions to proxy temperature
data = pd.read_csv( "EPICA-DOME-C-ice-core_dD_d180.tab", delimiter="\t" )

# Depth values from some other epica c dataset (not sure)
depth_age = pd.read_csv("depth_age.tab", delimiter = "\t")


# Extrapolate depth to age calc of above dataset to epica set we use

f = interpolate.interp1d(depth_age['Depth ice/snow [m]'], depth_age['Age [ka BP]'], 
                         kind='linear', fill_value='extrapolate')

data['Estimated Age [ka BP]'] = f(data['Depth ice/snow [m]'])



# append milankovic cycles


milankovitch_1 = pd.read_csv("Milankovic1991_1.tab", delim_whitespace=True, nrows = 500)
milankovitch_2 = pd.read_csv("Milankovic1991_2.tab", delim_whitespace=True, nrows = 500)
milankovitch_3 = pd.read_csv("Milankovic1991_3.tab", delim_whitespace=True, nrows = 500)
milankovitch_2 = milankovitch_2.reset_index()
milankovitch_3 = milankovitch_3.reset_index()


milankovitch = pd.concat([milankovitch_1,milankovitch_2,milankovitch_3],axis = 1)

milankovitch["AGE"] = milankovitch["AGE"] * -1 

milan_params = ['ECC', 'OMEGA', 'OBL', 'PREC',"90SJune","90SDec"]
for i in milan_params:
    
    f = interpolate.interp1d( milankovitch["AGE"], milankovitch[i] )
    
    data[f"Milan_{i}"] = f(data['Estimated Age [ka BP]'])

data["delta-iso18o-Base at 6.6m"] = data["δ18O H2O [‰ SMOW]"] - data["δ18O H2O [‰ SMOW]"][0:10].mean()
data["0.6*iso18o-Base at 6.6m"] = data["delta-iso18o-Base at 6.6m"] * 0.6



## Add large known volvanic activity:
    
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



def find_nearest_age(age, age_array):
    idx = (np.abs(age_array - age)).argmin()
    return age_array[idx]


# Step 3: Apply the function to map volcano ages to nearest temperature ages
volcano['nearest_temp_age'] = volcano['Ave Age -Years BP'].apply(lambda x: find_nearest_age(x, data['Estimated Age [ka BP]'].values))

data = pd.merge( data, volcano.filter(["nearest_temp_age","Volume – km3" ],axis=1), left_on="Estimated Age [ka BP]", right_on = "nearest_temp_age", how = 'left')
data["Volume – km3"] = data["Volume – km3"].fillna(0)




# Add co2 and ch4

co2_ice = pd.read_csv("Composite_CO2_AICC2012_chron.tab", delimiter = "\t")
ch4_ice = pd.read_csv("Vostok_CH4_AICC2012_chron.tab", delimiter = "\t")


# interpolate co2 and ch4


f = interpolate.interp1d(   co2_ice["Gas age [ka BP]"], co2_ice["CO2 [ppmv]"],
                                  bounds_error=False,  # Don't raise an error for out-of-bounds values
                                  fill_value=np.nan)
data['CO2'] = f(data['Estimated Age [ka BP]'])
 
f = interpolate.interp1d(   ch4_ice["Gas age [ka BP]"], ch4_ice["CH4 [ppbv]"],
                                  bounds_error=False,  # Don't raise an error for out-of-bounds values
                                  fill_value=np.nan)
data['CH4'] = f(data['Estimated Age [ka BP]'])



# scale values to regular timescale

interval = math.ceil(data['Estimated Age [ka BP]'].max())/ data.shape[0]



new_index = np.arange(math.floor(data['Estimated Age [ka BP]'].min()),
                  math.ceil(data['Estimated Age [ka BP]'].max()),
                  0.35)# every 350 years

interval_index = pd.IntervalIndex.from_breaks( new_index )
data['interval'] = pd.cut( data["Estimated Age [ka BP]"], bins = interval_index )

timeseries = data.groupby('interval').agg({'δD H2O [‰ SMOW]': 'mean',
                              'CO2':'mean',
                              "CH4":"mean",
                              'Milan_ECC':"mean",
                              'Milan_OMEGA':"mean",
                              'Milan_OBL':"mean",
                             'Milan_PREC':"mean",
                             'Milan_90SDec':"mean",
                             'Volume – km3':"mean",
                             'delta-iso18o-Base at 6.6m':"mean",
                             '0.6*iso18o-Base at 6.6m':"mean"})



time_series = timeseries.dropna()
time_series["interval_midpoint"] = time_series.index.map(lambda x: x.mid)
time_series.set_index('interval_midpoint', inplace=True)
time_series.to_csv("time_series_all.csv")






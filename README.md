# Temperature Modelling Using Icecore Data

## Overview

This repository contains a Streamlit application developed for modeling temperature using ice core data, with the capability to handle other tabular datasets as well. The application is designed to test various models and analyze their performance in relation to historical temperature data.

### Streamlit App

You can access the application here: [Icecore Temperature Modelling App](https://icecoretempmodelling-su4mds3nvzenpfcbycjzjw.streamlit.app)

## Background

The application utilizes data related to Milankovitch cycles, which are believed to be significant drivers of temperature and CO2 variations during glacial and interglacial periods. The analysis focuses on data from the Antarctic region, incorporating large volcanic activity datasets as well.

### Data Preparation

For details on data preparation and processing, please refer to the `data_prep` directory.

## Data Sources

- **Icecore Data Reference**: [EPICA Dome C Dataset](https://doi.pangaea.de/10.1594/PANGAEA.824894)
- **Oxygen Isotopes Reference**: [Oxygen Isotope Data](https://doi.pangaea.de/10.1594/PANGAEA.934094)
- **Sulphur Dataset**: [Sulphur Data](https://doi.pangaea.de/10.1594/PANGAEA.933271?format=html#download)
- **Volcanic Reconstruction**: [Volcanic Activity Data](https://doi.pangaea.de/10.1594/PANGAEA.601894?format=html#download)
-  **Volcanic Reconstruction**: [Volcanic Activity Data long history](https://www.whyclimatechanges.com/pdf/Publications/Ward2009SulfurDioxideTableS1.pdf)
  - missing milancokic estimation and forcing source < br >

## Usage:

Still a bit wonky, in development, will fill this in asap as i fix certain issues.

You must upload a dataset for anything to work <br />
choose atleast one independant variable <br />

for plots you can choose multiple but they must have the same x axis ( expected to be time - coded as intermediate_value currently)
there are ways to add squared and cubed values to models <br />
you can add lagged variables <br />
in the timeseries section you can choose the mean and volatility model used for the model <br />
some configs will prompt for more configs like ARX will ask you to define exogonous variables <br />
There is a way to call the trained model with new inputs at the bottom of some models <br />


## Isues

estrapolation from O18 isotope concentration and temperture is done in a very rudementary way (source required) <br />
some other adjustments to data may well be necessary, i am not a climatologist


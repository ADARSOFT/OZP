# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 17:31:49 2020

@author: Damir
"""
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder

# 1. Read data from file (Bank marketing)
bank_marketing = pd.read_csv("../Data/master-data/bank-additional.csv", sep = ";")

# 2. Show descriptive statistics for dataset (datatypes, values range, missing values)
bank_marketing_desc = bank_marketing.describe()

label = bank_marketing["y"]
label.describe()

# Separate categoric and numeric data
categoric_columns = ["job", "maritial", "education", "default", "housing", "loan", "contract", "month", "day_of_week", "poutcome"]
numeric_columns = ["age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]

categoric_data = bank_marketing.loc[:, categoric_columns]

numeric_data = bank_marketing.loc[:, numeric_columns]
numeric_data_description = numeric_data.describe()

# Check missing data
categoric_missing_columns = categoric_data.isnull().any()
numeric_missing_columns = numeric_data.isnull().any()

# 3. Prepare dataset for predictive modeling


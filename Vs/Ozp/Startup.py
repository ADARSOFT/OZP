import numpy as np
import pandas as pd 

# 1. Read data from file (Bank marketing)
bank_marketing = pd.read_csv('../../Data/master-data/bank-additional.csv')

# 2. Show descriptive statistics for dataset (datatypes, values range, missing values)
bank_marketing.values.head()

# 3. Prepare dataset for predictive modeling


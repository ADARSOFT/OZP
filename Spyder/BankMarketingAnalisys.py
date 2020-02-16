import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder

#%% 1. Read source data from file, and separate as categorical and numerical data
bank_marketing = pd.read_csv("../Data/master-data/bank-additional.csv", sep = ";")

categoric_columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]
# don't select duration column, because of high corellation
numeric_columns = ["age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
label_column = "y"
label_codes = {'yes':1, 'no':0}

# Map from yes/no to 1/0
bank_marketing[label_column] = bank_marketing[label_column].map(label_codes)

# Separate source data
label = bank_marketing[label_column]
categoric_data = bank_marketing.loc[:, categoric_columns].copy()
numeric_data = bank_marketing.loc[:, numeric_columns].copy()

# Replace 999 with -1 for column pdays, for experiment purpose i will try with -1 and 999 both combinations
numeric_data['pdays'] = numeric_data['pdays'].replace(999, -1) 

#%% 2. Descriptive statistics for dataset (datatypes, values range, missing values)

# How meny unique values exists in features
numeric_data.nunique()
categoric_data.nunique()
label.nunique()

numeric_data['pdays'].value_counts()

# Show all possible values for categoric data 
pd.value_counts(categoric_data.values.flatten())

# Datatypes
numeric_data.info()
categoric_data.info()

# Statistical measurements
numeric_data.describe()
label.describe()

# Output variable balance
label.value_counts() / len(label)

# Missing values
categoric_missing_columns = categoric_data.isnull().any()
numeric_missing_columns = numeric_data.isnull().any()

## Missing values CATEGORICAL data

''' Possible tactics for handling missing categorical data

0. Drop rows where all columns are NaN
1. Ignore missing data (delete)
2. Develop model to predict missing values 
3. Treat missing data as just another category
4. Replace data by most frequent value (mode)

1. Random versus selective loss of data 
2. Missing by design, not asked, or not applicable
'''

# Option 1
categoric_data = categoric_data.dropna(axis='columns', how='all')
numeric_data = numeric_data.dropna(axis='columns', how='all')

# Check percentage of unglonow categorical nad 
categorical_data_missing_values = ['marital', 'education', 'default', 'housing', 'loan']
data_len = len(categoric_data)

unknowns_categorical_data_percentages = pd.DataFrame([], columns = ['UnknownPercentage'], index = categorical_data_missing_values)

print ('Column name: Unknown values percentage')

for column in categorical_data_missing_values:
	percentage_unknown = categoric_data[column].value_counts()['unknown'] / data_len
	unknowns_categorical_data_percentages.loc[column,:] = {'UnknownPercentage':percentage_unknown*100}
	
unknowns_categorical_data_percentages.head()

print ('Column name: Null values percentage')

for column in categoric_columns:
	percentage_nan = categoric_data['marital'].isnull().value_counts() / data_len
	print('{}: {}%'.format(column, percentage_nan * 100))

# Column MARITAL (small amount of unknown data, strategy : replace with most fraquent value)

col_marital_mode = categoric_data['marital'].mode()
categoric_data['marital'] = categoric_data['marital'].replace('unknown', col_marital_mode.values[0])
categoric_data['marital'].value_counts()

# Column DEFAULT (thread unknown data as another category, because of high unknown values percentage). Action DO NOTHING, cause values already have code value 'unknown'

# Columns HOUSING and LOAD have high corellated UNKNOWN values and the same number. Conclusion this columns are MAR (Missing At Random). Decision is to use predictive model

# Column Education (threat unknown data as another category, because some people don't want to give informations about education, and that can be some pattern latter) MNAR (Missing Not At Random)

## Missing values numeric data



numeric_data['pdays'].value_counts() / len(numeric_data)

#%% 3. Prepare dataset for predictive modeling (dummy variables)
# Delete column 




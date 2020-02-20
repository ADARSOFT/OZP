import numpy as np
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

#%% 1. Read source data from file, and separate as categorical and numerical data
bank_marketing = pd.read_csv("../Data/master-data/bank-additional.csv", sep = ";")

categoric_columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]
# don't select duration column, because of high corellation
numeric_columns = ["age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
label_column = "y"
label_codes = {'yes':1, 'no':0}
label_codes_inverse = {1:'yes', 0:'no'}

# Map from yes/no to 1/0
bank_marketing[label_column] = bank_marketing[label_column].map(label_codes)

# Separate source data
label = bank_marketing[label_column]
categoric_data = bank_marketing.loc[:, categoric_columns].copy()
numeric_data = bank_marketing.loc[:, numeric_columns].copy()

# Replace 999 with -1 for column pdays, for experiment purpose i will try with -1 and 999 both combinations
numeric_data['pdays'] = numeric_data['pdays'].replace(999, -1) 

#%% 2. Descriptive statistics for dataset (datatypes, values range, missing values)

# How many unique values exists in features
numeric_data.nunique()
categoric_data.nunique() # Check if categoric data has to many levels (that can slow calculation)
label.nunique()

numeric_data['pdays'].value_counts()
numeric_data['nr.employed'].value_counts()

nr_employeed_mode = numeric_data['nr.employed'].mode()

numeric_data['nr.employed'] = numeric_data['nr.employed'].replace('no', float(nr_employeed_mode[0])) 
numeric_data[numeric_data['nr.employed'] == 'no'].index.values
# Menjam tip podataka za kolonu nr.employed
numeric_data = numeric_data.astype({'nr.employed': 'float64'})
numeric_data.dtypes

# Show all possible values for categoric data 
pd.value_counts(categoric_data.values.flatten())

# Datatypes
numeric_data.info()
	# NOTE: column nr.employed has wrond data type
categoric_data.info()

# Statistical measurements
desc = numeric_data.describe()

label.describe()

# Output variable balance
label.value_counts() / len(label)

# Missing values
categoric_missing_columns = categoric_data.isnull().any()
numeric_missing_columns = numeric_data.isnull().any()

# Missing values for output class
total_label_missing_rows = label.isnull().sum()
label = label.fillna(label.mode()[0])

## Missing values numeric data
total_missing_for_numeric_columns = numeric_data.isnull().sum()
numeric_missing_data_rows = numeric_data[numeric_data.isnull().any(axis=1)]

# fill numeric missing data by mean of column 
numeric_data = numeric_data.fillna(numeric_data.mean())
# check one more time for missing values NaN in numerical data
numeric_data.isnull().sum()

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

# Columns HOUSING and LOAN have high corellated UNKNOWN values and the same number. Conclusion this columns are MAR (Missing At Random). Decision is to use predictive model

	# get known data for training
categoric_data['housing'].value_counts() # possible values yes/no/unknown
categoric_data['loan'].value_counts() # possible values yes/no/unknown


indexes_not_missing_housing_values = categoric_data[categoric_data['housing'] != 'unknown'].index.values 
x_housing_train = numeric_data[numeric_data.index.isin(indexes_not_missing_housing_values)]
x_housing_test = numeric_data[~numeric_data.index.isin(indexes_not_missing_housing_values)]
Y_housing_train = categoric_data[categoric_data.index.isin(indexes_not_missing_housing_values)]['housing'].map(label_codes)

# Fit housing imputation 
alg_housing_imputation = LogisticRegression(random_state=0).fit(x_housing_train, Y_housing_train)
# Predict housing imputation
housing_predicted_imputation = alg_housing_imputation.predict(x_housing_test)
index_missing_housing = categoric_data[categoric_data['housing'] == 'unknown'].index
imputed_housing_missing_values = pd.DataFrame(data = housing_predicted_imputation, index = index_missing_housing, columns = ['housing'])
imputed_housing_missing_values['housing'] = imputed_housing_missing_values['housing'].map(label_codes_inverse)

# impute housing data
categoric_data.loc[imputed_housing_missing_values.index.values, 'housing'] = imputed_housing_missing_values.loc[:,'housing']

# Column Education (threat unknown data as another category, because some people don't want to give informations about education, and that can be some pattern latter) MNAR (Missing Not At Random)


numeric_data['pdays'].value_counts() / len(numeric_data)

#%% 3. Prepare dataset for predictive modeling (dummy variables)
# Delete column 




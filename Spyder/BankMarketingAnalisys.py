import numpy as np
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn import metrics

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
numeric_data['nr.employed'].value_counts() # Value no in row 760 is categoric and beacause of that whole column isn't float64 data type. We need to change that.

nr_employeed_mode = numeric_data['nr.employed'].mode()

numeric_data[numeric_data['nr.employed'] == 'no'].index.values # row 780
numeric_data['nr.employed'] = numeric_data['nr.employed'].replace('no', float(nr_employeed_mode[0])) 
# just to check
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
numeric_total_missing_data_rows = numeric_data.isnull().sum()

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

# Housing imputation section
indexes_not_missing_housing_values = categoric_data[categoric_data['housing'] != 'unknown'].index.values # get training data indexes
x_housing_train = numeric_data[numeric_data.index.isin(indexes_not_missing_housing_values)] # get features for training
x_housing_test = numeric_data[~numeric_data.index.isin(indexes_not_missing_housing_values)] # get features for prediction
Y_housing_train = categoric_data[categoric_data.index.isin(indexes_not_missing_housing_values)]['housing'].map(label_codes) # get labels for training (housing column)

# Fit housing imputation 
alg_housing_imputation = LogisticRegression(random_state=0).fit(x_housing_train, Y_housing_train)
# Predict housing imputation
housing_predicted_imputation = alg_housing_imputation.predict(x_housing_test)

# Create data frame with predicted values
index_missing_housing = categoric_data[categoric_data['housing'] == 'unknown'].index # get indexes of missing categorical rows
imputed_housing_missing_values = pd.DataFrame(data = housing_predicted_imputation, index = index_missing_housing, columns = ['housing']) # create data frame for new imputed values 
imputed_housing_missing_values['housing'] = imputed_housing_missing_values['housing'].map(label_codes_inverse) # map inverse to original data format yes/no

# Impute predicted data into original dataset
categoric_data.loc[imputed_housing_missing_values.index.values, 'housing'] = imputed_housing_missing_values.loc[:,'housing'] # impute to original data set

# Check imputation results
categoric_data['housing'].value_counts()

# Loan imputation section
indexes_not_missing_loan_values = categoric_data[categoric_data['loan'] != 'unknown'].index.values
x_loan_train = numeric_data[numeric_data.index.isin(indexes_not_missing_loan_values)]
x_loan_test = numeric_data[~numeric_data.index.isin(indexes_not_missing_loan_values)]
Y_loan_train = categoric_data[categoric_data.index.isin(indexes_not_missing_loan_values)]['loan'].map(label_codes)

# Fit loan imputation
alg_loan_imputation = LogisticRegression(random_state=0).fit(x_loan_train, Y_loan_train)

# Predict loan imputation
loan_predicted_imputation = alg_loan_imputation.predict(x_loan_test);

# Create new data frame with predicted values
index_missing_loan = categoric_data[categoric_data['loan'] == 'unknown'].index 
imputed_loan_missing_values = pd.DataFrame(data = loan_predicted_imputation, index = index_missing_loan, columns = ['loan'])
imputed_loan_missing_values['loan'] = imputed_loan_missing_values['loan'].map(label_codes_inverse)

# Impute predicted data into original dataset
categoric_data.loc[imputed_loan_missing_values.index.values, 'loan'] = imputed_loan_missing_values.loc[:,'loan']

# Check imputation results 
categoric_data['loan'].value_counts()

# Column Education (threat unknown data as another category, because some people don't want to give informations about education, and that can be some pattern latter) MNAR (Missing Not At Random)

numeric_data['pdays'].value_counts() / len(numeric_data)

#%% 3. Prepare dataset for predictive modeling 
# Deal with categorical variables and prepare for predictive modeling

'''
1. Convert to number (some ML libraries do not take categorical variables as input)
	1.1 Label encoder 
	1.2 Convert numeric bins to numbers 

2. Combine levels (to avoid redundant levels in a categorical variable and to deal with 
rare levels, we can simply combine the different levels)

	2.1 Using business logic
	2.2 Using frequency or response rate 

3. Dummy Coding  "One hot encoder" -(commonly used method for converting a categorical input variable into continuous variable.
It increase number of features by levels of categorical variables)	

OneHotEncoder vs Label encoder:

- The problem here is since there are different numbers in the same column, 
the model will misunderstand the data to be in some kind of order, 0 < 1 <2.

- The model may derive a correlation like as the country number increases the population 
increases but this clearly may not 
be the scenario in some other data or the prediction set. 
To overcome this problem, we use One Hot Encoder.

'''

# OneHotEncoder for categorical data
categoric_data = pd.get_dummies(categoric_data)

# Dealing with outliers -- USE COMMON SENSE
'''
'age', --> min 18 max 88 (this is ok)
'duration', min 0 (ok) and max value 
'campaign', min 1 max(ok) 35 (ok)
'pdays', min -1 (ok) max 21 (ok)
'previous', min 0 (ok) max 6 (ok)
'emp.var.rate', --> varijacija stope nezaposlenosti min -3.4 max 1.4, OK
'cons.price.idx', --> index potrosackih cena - min 92.201 max 94.767
'cons.conf.idx', https://www.nbs.rs/internet/latinica/glossary.html?id_letter=10&jezik=1 (one outlier more than 7 std) row 780
'euribor3m', Euro Interbank Offered Rate --> https://www.nlb.me/me/stanovnistvo/savjeti/sta-je-euribor row 780 (64 std)
'nr.employed' no rows onver 3std
'''
numeric_data.boxplot(column = numeric_data.columns.values.tolist())
numeric_data.boxplot(column = ['euribor3m'])

column_name = 'euribor3m'
z = np.abs(stats.zscore(numeric_data[column_name]))

threshold = 3
input_array = np.array(np.where(z > threshold))

numeric_data[column_name].ix[numeric_data.index.isin(input_array[0])]

df_outliers = pd.DataFrame([], columns = ['OutliersCount', 'ColumnName'], index = None )
#df_outliers.set_index('ColumnName', inplace=True)



for column in numeric_data.columns:
	zsc = np.abs(stats.zscore(numeric_data[column]))
	input_array_zsc = np.array(np.where(zsc > threshold))
	outliers = numeric_data[column].ix[numeric_data.index.isin(input_array_zsc[0])]
	df_outliers = df_outliers.append({'OutliersCount' : len(outliers) , 'ColumnName' : column} , ignore_index=True)

# Drop row 780
numeric_data = numeric_data.drop(780)
categoric_data = categoric_data.drop(780)
label = label.drop(780)

# Check corellation for data (Pearson coefficient)

def displayCorrelationMatrix(numeric_data_p):
	
	corr1 = numeric_data_p.corr()
	
	plt.subplots(figsize=(15,11))
	sns.heatmap(corr1, 
	            xticklabels=corr1.columns.values,
	            yticklabels=corr1.columns.values,
				cmap='RdBu_r',
				annot=True,
				linewidth=2)

displayCorrelationMatrix(numeric_data)

# Display BAR diagram for outliers description
def displayBarChartForOutliers(df_with_outliers, y_label_name, title_name, y_axis_column_name, x_axis_column_name):
	
	y_axis = df_with_outliers[y_axis_column_name].values
	y_pos = np.arange(len(y_axis))
	x_axis = df_with_outliers[x_axis_column_name].values

	plt.bar(y_pos, x_axis, align='center', alpha=0.5)
	plt.xticks(y_pos, y_axis, rotation=75)
	plt.ylabel(y_label_name)
	plt.title(title_name)
	
	plt.show()

displayBarChartForOutliers(df_outliers, 'Outliers count', 'Outliers by columns', 'ColumnName', 'OutliersCount')

# Concatenate data 
data = pd.concat([numeric_data, categoric_data], axis=1, sort=False)

from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(1- 0.98))
new_data = sel.fit_transform(data).shape

#%% Normalization standard scaler (x - mean) / std

scaler = StandardScaler()
# koristim scaler objekat da bih kasnije mogao da radim inverse transform
standard_scaler_transformed = scaler.fit_transform(data)
# scaler.inverse_transform(standard_scaler_transformed)

data_scaled = pd.DataFrame(data = standard_scaler_transformed, columns = data.columns.values)
inverse_scaled = scaler.inverse_transform(data_scaled)
# dokaz formule, ako su brojevi ispod proseka bice u minusu, i zavisi od standardne devijacije kolone koja ce vrednost biti. 
# sto je standardna devijacija veca (delilac), to ce skalirana vrednost biti manja


std_of_age = data.iloc[:,0:1].std()[0]
mean_of_age = data.iloc[:, 0:1].mean()[0]
x_age_row_0 = data.iloc[0:1, 0:1].values[0][0]

age_scaled_row_0 = (x_age_row_0 - mean_of_age) / std_of_age
age_scaled_row_0_scaled = data_scaled.iloc[0:1, 0:1].values[0][0]

#%%  Split data 

X_train, X_test, Y_train, Y_test = train_test_split(data_scaled, label, test_size=0.3)

#%% PCA analysis

def displayPCAVariationPlot(p):
	plt.figure()
	plt.plot(np.cumsum(p.explained_variance_ratio_))
	plt.xlabel('Number of Components')
	plt.ylabel('Variance (%)') #for each component
	plt.title('Dataset Explained Variance')
	plt.show()

# Data before reduction with all possible components
pca_all = PCA().fit(X_train)

displayPCAVariationPlot(pca_all)

print("Total variance percentage: {}%".format(pca_all.explained_variance_ratio_.sum()))

print("Total number of components: {}".format(len(pca_all.explained_variance_ratio_)))

# Data with 98% percent of components variation
pca_98_percent = PCA(0.98).fit(X_train)

displayPCAVariationPlot(pca_98_percent)

print("Total variance percentage: {}%".format(pca_98_percent.explained_variance_ratio_.sum()))

components_number_98_per_variance = len(pca_98_percent.explained_variance_ratio_)

print("Total number of components: {}".format(components_number_98_per_variance))

# Transform data with PCA (reduce dimensionality)

pca = PCA(n_components = components_number_98_per_variance)

pca_transformed_variance_ratio = pd.DataFrame(pca.fit(X_train).explained_variance_ratio_*100, columns = ['Variance_ratio_%'])
print(pca_transformed_variance_ratio)
print('Total variance percentage: {} %'.format(pca_transformed_variance_ratio.sum()[0]))

train_data_pca_transformed = pca.fit_transform(X_train).copy()

#%% vezbanje PCA
train_data_pca_transformed_test = pca.fit(X_train)
train_data_pca_transformed_test = pca.transform(X_train)
original_data_test = pca.inverse_transform(train_data_pca_transformed_test)

train_data_pca_transformed.shape
train_data_pca_transformed_test.shape
original_data_test.shape

my_dataframe = pd.DataFrame(train_data_pca_transformed_test)
my_datafeame2 = pd.DataFrame(original_data_test)

#%% Kreirajte klaster model, i odredite klastere svake instance. Karakterisite dobijene klastere. 

# trazim optimalan broj klastera
def calculateSilhoueteIndexForClusters(max_clusters):	
	
	silhouette_idex_array = np.array([])
	
	for i in range(2,max_clusters):
		kmeans_model = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(train_data_pca_transformed)
		labels = kmeans_model.labels_
		sil_idx =  metrics.silhouette_score(train_data_pca_transformed, labels, metric='euclidean')
		silhouette_idex_array = np.append(silhouette_idex_array, sil_idx)

		print('Iteration: {} , score: {}'.format(i,sil_idx))
	
	plt.figure()
	plt.plot(silhouette_idex_array)
	plt.xlabel('Cluster number')
	plt.ylabel('Silhouette index') #for each component
	plt.title('Optimal clusters - silhouette index by cluster')
	plt.show()


calculateSilhoueteIndexForClusters(60)

# But for simplicity we will choose small number of clusters
cluster_n = 4

# Fit
cluster_model = KMeans(n_clusters = cluster_n, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0, n_jobs =-1).fit(data_scaled)

clusters_ratio = np.unique(cluster_model.labels_, return_counts = True) 

centers_df = pd.DataFrame(cluster_model.cluster_centers_, columns = X_train.columns)
centers_df.head(cluster_n)

def displayCentersCountPlotBarChart(centers_counts, centers_index):
	fig, ax = plt.subplots()
	plt.bar(centers_index, centers_counts)
	plt.xlabel('Klaster')
	plt.ylabel('Ukupan broj instanci')
	plt.title('Odnos ukupnog broja instanci po klasteru')
	plt.show()
	
displayCentersCountPlotBarChart(clusters_ratio[1], clusters_ratio[0])

# Spojiti dataset sa labelama klastera
labels_df = pd.DataFrame(cluster_model.labels_, columns = ['Cluster'])

data_scaled_labeled = pd.concat([data_scaled, labels_df], axis=1, sort=False)



# Prikazati statistiku po klasteru

def returnInverseTransformedData(data_scaled_labeled_p):
	features = data_scaled_labeled_p.iloc[:,:-1]
	lbl = pd.DataFrame(data_scaled_labeled_p.iloc[:,-1], columns = ['Cluster']) 
	
	inverse_transformed_data = pd.DataFrame(scaler.inverse_transform(features), columns = features.columns)
	inverse_transformed_data['Cluster'] = lbl.iloc[:,-1]
	return inverse_transformed_data

inverse_transfomed_data = returnInverseTransformedData(data_scaled_labeled)

def getDescriptionForClusters(data_scaled_labeled_p, cluster_number, cluster_centers):
	
	clusters_description = pd.DataFrame([])
	
	for i in range(cluster_number):
		
		desc_internal = data_scaled_labeled_p[data_scaled_labeled_p['Cluster'] == i].describe()
		desc_internal['Cluster'] = i
		center = centers_df.iloc[i:i+1,:]
		center['Cluster'] = i
		desc_internal = desc_internal.append(center)
		clusters_description = clusters_description.append(desc_internal)
		
	return clusters_description

desc_for_cluster = getDescriptionForClusters(inverse_transfomed_data, 4, centers_df)

def showScatterPlotForSomeColumnsMax4Clusters(labels, features, first_column_name, second_column_name):
	
	cluster_map = {0:'g', 1:'b', 2:'r', 3:'y'}
	cluster_color = [cluster_map[i] for i in labels]
	plt.close()
	plt.xlabel(first_column_name)
	plt.ylabel(second_column_name)
	plt.title('Cluster distributions by colors')
	plt.scatter(features.loc[:,first_column_name], features.loc[:,second_column_name], c = cluster_color, label = labels)
	

# Call Scatter plots for different column combinations
showScatterPlotForSomeColumnsMax4Clusters(inverse_transfomed_data.iloc[:,-1].values, inverse_transfomed_data.iloc[:,:-1], 'age', 'campaign')
showScatterPlotForSomeColumnsMax4Clusters(inverse_transfomed_data.iloc[:,-1].values, inverse_transfomed_data.iloc[:,:-1], 'age', 'emp.var.rate')
showScatterPlotForSomeColumnsMax4Clusters(inverse_transfomed_data.iloc[:,-1].values, inverse_transfomed_data.iloc[:,:-1], 'age', 'nr.employed')
showScatterPlotForSomeColumnsMax4Clusters(inverse_transfomed_data.iloc[:,-1].values, inverse_transfomed_data.iloc[:,:-1], 'age', 'previous')

#%% Kreirati minimalno 3 prediktivna modela  (sa default parametrima) uporeditw ih kros validacijom i ocenite gresku na test setu 
# Minimum 2 mere evaluacije. Koristiti Pipeline

'''
Logistic Regression
Decision Tree
Naive Bayes
'''
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

pipeline_steps_lr = [('minMax', MinMaxScaler()), ('lr', LogisticRegression())] 
pipe_lr = Pipeline(steps = pipeline_steps_lr)
pipe_lr.set_params(lr__random_state=0)
pipe_lr.set_params(lr__class_weight='balanced')



rf_alg = RandomForestClassifier(random_state=0, class_weight='balanced')
rf_model = rf_alg.fit(X_train,Y_train)

lr_alg = LogisticRegression(random_state=0, class_weight='balanced')
lr_model = lr_alg.fit(X_train,Y_train)

nb_alg = GaussianNB()
nb_model = nb_alg.fit(X_train,Y_train)

crossValidResult_randomForest = pd.DataFrame(cross_validate(rf_alg, X_train, Y_train, cv=10, n_jobs=-1, return_train_score=True, scoring = ['precision', 'recall','accuracy','f1']))
crossValidResult_logisticRegression = pd.DataFrame(cross_validate(lr_alg, X_train, Y_train, cv=10, n_jobs=-1, return_train_score=True, scoring = ['precision', 'recall','accuracy','f1'])) 
crossValidResult_gaussianNB = pd.DataFrame(cross_validate(nb_alg, X_train, Y_train, cv=10, n_jobs=-1, return_train_score=True, scoring = ['precision', 'recall','accuracy','f1'])) 

# Cross validate with pipeline
crossValidResult_lr_pipe = pd.DataFrame(cross_validate(pipe_lr, X_train, Y_train, cv=10, n_jobs=-1, return_train_score=True, scoring = ['precision', 'recall','accuracy','f1'])) 





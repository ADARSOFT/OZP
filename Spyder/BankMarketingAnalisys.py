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
pd.DataFrame(pd.value_counts(categoric_data.values.flatten())).T

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
categoric_data = categoric_data.dropna(axis='index', how='all')
numeric_data = numeric_data.dropna(axis='index', how='all')

# Check percentage of unknown categorical data 
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
	percentage_nan = categoric_data[column].isnull().value_counts() / data_len
	print('{}: {}%'.format(column, percentage_nan * 100))

# Column MARITAL (small amount of unknown data, strategy : replace with most fraquent value)

col_marital_mode = categoric_data['marital'].mode()
categoric_data['marital'] = categoric_data['marital'].replace('unknown', col_marital_mode.values[0])
categoric_data['marital'].value_counts()

# Column DEFAULT (thread unknown data as another category, because of high unknown values percentage). Action DO NOTHING, cause values already have code value 'unknown'

#%% Columns HOUSING and LOAN have high corellated UNKNOWN values and the same number. Conclusion this columns are MAR (Missing At Random). Decision is to use predictive model

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

#%%
# 'age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'

sns.distplot(numeric_data['age']);
sns.distplot(numeric_data['duration']);
sns.distplot(numeric_data['campaign']);
sns.distplot(numeric_data['emp.var.rate'])
sns.distplot(numeric_data['previous'])
sns.distplot(numeric_data['pdays'])

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

# Display pairplot
sns.pairplot(numeric_data);

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

# numeric_data.boxplot(column = numeric_data.columns.values.tolist())
numeric_data.boxplot(column = ['age'])
numeric_data.boxplot(column = ['duration'])
numeric_data.boxplot(column = ['nr.employed'])
numeric_data.boxplot(column = ['cons.conf.idx'])


threshold = 3

df_outliers = pd.DataFrame([], columns = ['OutliersCount', 'ColumnName'], index = None )

for column in numeric_data.columns:
	zsc = np.abs(stats.zscore(numeric_data[column]))
	input_array_zsc = np.array(np.where(zsc > threshold))
	outliers = numeric_data[column].ix[numeric_data.index.isin(input_array_zsc[0])]
	df_outliers = df_outliers.append({'OutliersCount' : len(outliers) , 'ColumnName' : column} , ignore_index=True)

# Drop row 780
numeric_data = numeric_data.drop(780)
categoric_data = categoric_data.drop(780)
label = label.drop(780)


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


# Concatenate data 
data = pd.concat([numeric_data, categoric_data], axis=1, sort=False)

#%% Normalization standard scaler (x - mean) / std
scaler = StandardScaler()
# koristim scaler objekat da bih kasnije mogao da radim inverse transform
standard_scaler_transformed = scaler.fit_transform(data)
# scaler.inverse_transform(standard_scaler_transformed)

data_scaled = pd.DataFrame(data = standard_scaler_transformed, columns = data.columns.values)

#%% PCA analysis

def displayPCAVariationPlot(p):
	plt.figure()
	plt.plot(np.cumsum(p.explained_variance_ratio_))
	plt.xlabel('Number of Components')
	plt.ylabel('Variance (%)')
	plt.title('Dataset Explained Variance')
	plt.show()

# Data before reduction with all possible components
pca_all = PCA().fit(data_scaled)

displayPCAVariationPlot(pca_all)

print("Total variance percentage: {}%".format(pca_all.explained_variance_ratio_.sum()))

print("Total number of components: {}".format(len(pca_all.explained_variance_ratio_)))

# Data with 98% percent of components variation
pca_98_percent = PCA(0.98).fit(data_scaled)

displayPCAVariationPlot(pca_98_percent)

print("Total variance percentage: {}%".format(pca_98_percent.explained_variance_ratio_.sum()))

components_number_98_per_variance = len(pca_98_percent.explained_variance_ratio_)

print("Total number of components: {}".format(components_number_98_per_variance))

# Transform data with PCA (reduce dimensionality)

pca = PCA(n_components = components_number_98_per_variance)

pca_transformed_variance_ratio = pd.DataFrame(pca.fit(data_scaled).explained_variance_ratio_*100, columns = ['Variance_ratio_%'])
print(pca_transformed_variance_ratio)
print('Total variance percentage: {} %'.format(pca_transformed_variance_ratio.sum()[0]))

train_data_pca_transformed = pca.fit_transform(data_scaled).copy()

#%% vezbanje PCA
train_data_pca_transformed_test = pca.fit(data_scaled)
train_data_pca_transformed_test = pca.transform(data_scaled)
original_data_test = pca.inverse_transform(train_data_pca_transformed_test)

train_data_pca_transformed.shape
train_data_pca_transformed_test.shape
original_data_test.shape

my_dataframe = pd.DataFrame(train_data_pca_transformed_test)
my_datafeame2 = pd.DataFrame(original_data_test)

#%%  Split data 
data_min_max_scaled = MinMaxScaler().fit_transform(data)

x_data, x_forget_data, y_data, y_forget_data =  train_test_split(data, label, test_size=0.1)

X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.2)

#%% Kreirajte klaster model, i odredite klastere svake instance. Karakterisite dobijene klastere. 

# trazim optimalan broj klastera
def calculateSilhoueteIndexForClusters(max_clusters):	
	
	silhouette_idex_array = np.array([])
	
	for i in range(2,max_clusters):
		kmeans_model = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(data_min_max_scaled)
		labels = kmeans_model.labels_
		sil_idx =  metrics.silhouette_score(data_min_max_scaled, labels, metric='euclidean')
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
print(desc_for_cluster.head(36))

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

#%% 7. Kreirati minimalno 3 prediktivna modela  (sa default parametrima) uporeditw ih kros validacijom i ocenite gresku na test setu 
# Minimum 2 mere evaluacije. Koristiti Pipeline

from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def doCrossValidation(alg_):
	return cross_validate(alg_, X_train, Y_train, cv=10, n_jobs=-1, return_train_score=True, scoring = ['precision', 'recall','accuracy','f1'])

# Primer pipeline-a za logisticku regresiju
pipeline_steps_lr = [('minMax', MinMaxScaler()), ('lr', LogisticRegression())] 
pipe_lr = Pipeline(steps = pipeline_steps_lr)
pipe_lr.set_params(lr__random_state = 0)
pipe_lr.set_params(lr__class_weight = 'balanced')
cross_validation_result_lr_pipe = pd.DataFrame(doCrossValidation(pipe_lr))
np.mean(cross_validation_result_lr_pipe)

# Primer pipeline-a za cross validaciju sa naive bayes-om
pipeline_steps_gnb = [('standardScaler', StandardScaler()), ('pca', PCA()), ('gnb', GaussianNB())]
pipe_gnb = Pipeline(steps = pipeline_steps_gnb)
cross_validation_result_gnb_pipe = pd.DataFrame(doCrossValidation(pipe_gnb))
np.mean(cross_validation_result_gnb_pipe)

# Primer pipeline-a za cross validaciju sa random forest algoritmom
pipeline_steps_rndfr = [('minMax', MinMaxScaler()), ('rndf', RandomForestClassifier())]
pipe_rndf = Pipeline(steps = pipeline_steps_rndfr)
pipe_rndf.set_params(rndf__random_state = 0)
pipe_rndf.set_params(rndf__class_weight = 'balanced')
cross_validation_result_rndf_pipe = pd.DataFrame(doCrossValidation(pipe_rndf))
np.mean(cross_validation_result_rndf_pipe)

# Primer pipeline-a za cross validaciju sa random knn algoritmom
pipeline_steps_knn = [('standardScaler', StandardScaler()), ('pca', PCA()), ('knn', KNeighborsClassifier())]
pipe_knn = Pipeline(steps = pipeline_steps_knn)
pipe_knn.set_params(knn__n_neighbors=3)
cross_validation_results_knn_pipe = pd.DataFrame(doCrossValidation(pipe_knn))
np.mean(cross_validation_results_knn_pipe)

# Primer pipeline-a za cross validaciju sa decision three algoritmom
pipeline_steps_dthree = [('minMax', MinMaxScaler()), ('dthree', DecisionTreeClassifier())]
pipe_dthree = Pipeline(steps = pipeline_steps_dthree)
pipe_dthree.set_params(dthree__class_weight = 'balanced')
cross_validation_results_dthree = pd.DataFrame(doCrossValidation(pipe_dthree))
np.mean(cross_validation_results_dthree)

#%% 8. Promenite minimalno 2 parametra kod najboljeg modela i ocenite gresku na test setu.
pipeline_steps_rndfr2 = [('minMax', MinMaxScaler()), ('rndf', RandomForestClassifier())]
pipe_rndf2 = Pipeline(steps = pipeline_steps_rndfr2)
pipe_rndf2.set_params(rndf__random_state = 0)
pipe_rndf2.set_params(rndf__class_weight = 'balanced')
pipe_rndf2.set_params(rndf__max_depth = 16)
pipe_rndf2.set_params(rndf__n_estimators = 40)
cross_validation_result_rndf_pipe2 = pd.DataFrame(cross_validate(pipe_rndf2, X_train, Y_train, cv=10, n_jobs=-1, return_train_score=True, scoring = ['precision', 'recall','accuracy','f1']))
np.mean(cross_validation_result_rndf_pipe2)

#%% 9. Testirajte prediktivne modele na atributima koji kumulativno nose 98% varijanse.

# Prvi nacin
minmax_scaled_data_alg = MinMaxScaler()
minmax_scaled_data = pd.DataFrame(minmax_scaled_data_alg.fit_transform(data), columns = data.columns) 
pca_98_percent_minmax_alg = PCA(0.98)
pca_98_percent_minmax = pca_98_percent_minmax_alg.fit_transform(minmax_scaled_data)
pca_98_percent_minmax.shape

# Drugi nacin (vaznost atributa) - eksperiment
alg_rndforest = RandomForestClassifier(random_state = 0, class_weight = 'balanced')
alg_rndforest.fit(X_train, Y_train)
dataframe = pd.DataFrame(alg_rndforest.feature_importances_)
dataframe.index += 1
dataframe_columns = pd.DataFrame(X_train.columns)
dataframe_columns.index += 1

varAnalisys = pd.concat([dataframe, dataframe_columns], axis=1, sort=True)
varAnalisys.columns = ['FeatureImportance', 'FeatureName']
varAnalisys = varAnalisys.sort_values(by = ['FeatureImportance'], ascending=False)
varAnalisys['CumSumImportance'] = np.cumsum(varAnalisys['FeatureImportance']).values

# Treci nacin rucno racunanje varijanse - eksperiment
variance_dataframe = pd.DataFrame([], columns = ['Index','FeatureName','Variance']) 

counter = 1
for column in minmax_scaled_data.columns:
	variance = minmax_scaled_data.loc[:,column].var()
	variance_dataframe = variance_dataframe.append({'Index':counter,'FeatureName' : column, 'Variance' :variance}, ignore_index = True)
	counter += 1

total_variance = np.sum(variance_dataframe['Variance'].values)
variance_dataframe['VariancePercentage'] = variance_dataframe['Variance'] /  total_variance
sorted_variance = variance_dataframe.sort_values(by = ['VariancePercentage'], ascending=False)


#%% 10. Podelite inicijalni skup po klasterima koje ste dobili i sacuvajte ih u posebnim promenljivima. 
# Na svakom od skupova. trenirajte jedan model i uporedite rezultate po razlicitim skupovima. 

original_clustered_data = pd.concat([pd.DataFrame(scaler.inverse_transform(data_scaled), columns = data_scaled.columns), labels_df, label], axis=1, sort=False)
original_clustered_data = original_clustered_data.dropna(axis='rows', how='any')

X_train_new, X_test_new, Y_train_new, Y_test_new = train_test_split(original_clustered_data.iloc[:,:-1], original_clustered_data.iloc[:,-1], test_size=0.3)
Y_train_new.value_counts()

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)

#X_train_new, Y_train_new = sm.fit_resample(X_train_new, Y_train_new)

# Podeliti podatke
original_cluster_0_data_x = X_train_new[X_train_new['Cluster'] == 0]
original_cluster_0_data_y = pd.DataFrame(Y_train_new.ix[Y_train_new.index.isin(original_cluster_0_data_x.index)], columns = ['y'])        

original_cluster_1_data_x = X_train_new[X_train_new['Cluster'] == 1]
original_cluster_1_data_y = Y_train_new.ix[Y_train_new.index.isin(original_cluster_1_data_x.index)]

original_cluster_2_data_x = X_train_new[X_train_new['Cluster'] == 2]
original_cluster_2_data_y = Y_train_new.ix[Y_train_new.index.isin(original_cluster_2_data_x.index)]

original_cluster_3_data_x = X_train_new[X_train_new['Cluster'] == 3]
original_cluster_3_data_y = Y_train_new.ix[Y_train_new.index.isin(original_cluster_3_data_x.index)]

scores = ['precision', 'recall','accuracy','f1']

# Pipe za 0 klaster sa logistickom regresijom
pipe_cluster0_steps = [('stancardScaler', StandardScaler()), ('pca', PCA()), ('lr', LogisticRegression())]

pipe_cluster0_lr_pipe = Pipeline(steps = pipe_cluster0_steps)
pipe_cluster0_lr_pipe.set_params(lr__random_state = 0)
pipe_cluster0_lr_pipe.set_params(lr__class_weight = 'balanced')
pipe_cluster0_lr_pipe.set_params(pca__n_components = 0.98) 

cross_validation_cluster_0_pipe_result = pd.DataFrame(cross_validate(pipe_cluster0_lr_pipe, original_cluster_0_data_x, original_cluster_0_data_y, cv=10, n_jobs=-1, return_train_score=True, scoring = scores))
np.mean(cross_validation_cluster_0_pipe_result)

# Pipe za 1 klaster sa random forest prediktorom
pipe_cluster1_steps = [('stancardScaler', StandardScaler()), ('pca', PCA()), ('rndf', RandomForestClassifier())]
pipe_cluster1_rndf_pipe = Pipeline(steps = pipe_cluster1_steps)
pipe_cluster1_rndf_pipe.set_params(rndf__random_state = 0)
pipe_cluster1_rndf_pipe.set_params(rndf__class_weight = 'balanced')
pipe_cluster1_rndf_pipe.set_params(pca__n_components = 0.98) 

cross_validation_cluster_1_pipe_result = pd.DataFrame(cross_validate(pipe_cluster1_rndf_pipe, original_cluster_1_data_x, original_cluster_1_data_y, cv=10, n_jobs=-1, return_train_score=True, scoring = scores))
np.mean(cross_validation_cluster_1_pipe_result)

# Pipe za 2 klaster sa KNN 
pipe_cluster2_steps = [('stancardScaler', StandardScaler()), ('pca', PCA()), ('knn', KNeighborsClassifier())]
pipe_cluster2_rndf_pipe = Pipeline(steps = pipe_cluster2_steps)
pipe_cluster2_rndf_pipe.set_params(pca__n_components = 0.98) 

cross_validation_cluster_2_pipe_result = pd.DataFrame(cross_validate(pipe_cluster2_rndf_pipe, original_cluster_2_data_x, original_cluster_2_data_y, cv=10, n_jobs=-1, return_train_score=True, scoring = scores))
np.mean(cross_validation_cluster_2_pipe_result)

# Pipe za 3 klaster sa GaussianNB 

pipe_cluster3_steps = [('stancardScaler', StandardScaler()), ('pca', PCA()), ('nb', GaussianNB())]
pipe_cluster3_rndf_pipe = Pipeline(steps = pipe_cluster3_steps)
pipe_cluster3_rndf_pipe.set_params(pca__n_components = 0.98) 

cross_validation_cluster_3_pipe_result = pd.DataFrame(cross_validate(pipe_cluster3_rndf_pipe, original_cluster_3_data_x, original_cluster_3_data_y, cv=10, n_jobs=-1, return_train_score=True, scoring = scores))
np.mean(cross_validation_cluster_3_pipe_result)

#%% 11. Odgovorite na sledeca pitanja:
# 	- Na kom podskupu dobijate najbolje performanse predikcije? 
#	Na drugom podskupu (klaster 1) koristeci random forest.

#	- Kako se razlikuju predikcije na kompletnom test setu i na parcijalnim?
#   Na parcijalnom setu (klaster 1) su dosta losiji rezultati

#	- Koji atributi imaju najvecu prediktivnu moc?
varAnalisys.head(10)
 
#	- Kako se razlikuju performanse modela sa optimizovanim parametrima u odnosu na modele sa default parametrima?
#	- Da li mislite da bi neka druga kombinacija bila bolja za vas dataset i zasto?
#	- Da li su vasi modeli pretrenirani (overfit)?

#%% BONUS: Pronadjite podskup atributa koji maksimizuje performanse prediktivnih algoritama. 
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest #, f_classif
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

X_train_new_resampled, Y_train_new_resampled = sm.fit_resample(X_train_new.iloc[:,:-1], Y_train_new)

# Metod 1 - Recursive feature elimination
rfe = RFE(estimator=ExtraTreesClassifier(), n_features_to_select=1, step=1)
rfe.fit(X_train_new.iloc[:,:-1], Y_train_new)
rfe.ranking_

# Metod 2
# Podesavam defaultni pipe
select_kbest_pipe_steps = [('minMax', MinMaxScaler()), ('selKbest', SelectKBest()), ('rndf', RandomForestClassifier())] 
select_kbest_pipe = Pipeline(steps = select_kbest_pipe_steps)
select_kbest_pipe.set_params(selKbest__score_func = chi2)
select_kbest_pipe.set_params(rndf__random_state = 0)
select_kbest_pipe.set_params(rndf__class_weight = 'balanced')

# podesavam parametre koje zelim da trazim 
param_grid = [
{ 
	  'rndf__n_estimators': list(range(10,30,5)), 
	  'selKbest__k': list(range(2, 60)), 
	  'rndf__min_samples_leaf': list(range(1,4)),
	  #'rndf__max_leaf_nodes': list(range(10,110,10))
}]

# pomocu grid search metode trazim najbolje parametre
search = GridSearchCV(select_kbest_pipe, param_grid, n_jobs=-1)
search.fit(X_train_new.iloc[:,:-1], Y_train_new)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

# setujem najbolje parametre
select_kbest_pipe.set_params(selKbest__k = search.best_params_['selKbest__k'])
select_kbest_pipe.set_params(rndf__n_estimators = search.best_params_['rndf__n_estimators'])

# pokrecem cross validaciju
calc_result = cross_validate(select_kbest_pipe, X_train_new_resampled, Y_train_new_resampled, cv=10, n_jobs=-1, return_train_score=True, scoring = scores)
cross_validation_kbest_pipe_result = pd.DataFrame(calc_result)
np.mean(cross_validation_kbest_pipe_result)


# Izvlacim nazive kolona best_k
best_k = search.best_params_['selKbest__k']
X_train_new_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X_train_new), columns = X_train_new.columns)
select_k_best = SelectKBest(score_func=chi2, k = best_k).fit(X_train_new_scaled, Y_train_new)
k_best_column_mask = select_k_best.get_support(indices=True)
features_df_new = X_train_new_scaled.iloc[:,k_best_column_mask]
print(features_df_new.columns)


import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRFRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
#%% Read and prepare data TIC-TAC-TOC set CLASSIFICATION
colnames=['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square','middle-middle-square','middle-right-square','bottom-left-square','bottom-middle-square','bottom-right-square', 'x_win'] 
tic_tac_mapping = {'positive':1, 'negative':0}
tic_tac_toc_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/tic-tac-toe/tic-tac-toe.csv", sep = ",", names=colnames, header=None)
tic_tac_toc_data['x_win'] = tic_tac_toc_data['x_win'].map(tic_tac_mapping) # change output variable
tic_tac_toc_label = tic_tac_toc_data['x_win']
tic_tac_toc_numerical = pd.get_dummies(tic_tac_toc_data.iloc[:,:-1])
tic_tac_toc_numerical['x_win'] = tic_tac_toc_label
X_tic_tac_toc = tic_tac_toc_numerical['x_win'] 
y_tic_tac_toc = tic_tac_toc_numerical.iloc[:,:-1]

#%% Read and prepare data HEART DISIEASE set CLASSIFICATION
heart_disiease_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/heart.csv", sep = ",")
X_heart = heart_disiease_data.iloc[:,:-1]
y_heart = heart_disiease_data.iloc[:,-1]

#%% Read and prepare data SKIN SegmentationCLASSIFICATION
skin_segmentation_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/skin/Skin_NonSkin.csv", sep = "	", header=None, names=['Blue','Green','Red','IsSkin'])
X_skin_segmentation = skin_segmentation_data.iloc[:, :-1]
y_skin_segmentation = skin_segmentation_data.iloc[:, -1]

#%% Read and prepare data happines CLASSIFICATION
happiness_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/happiness/happiness.csv", sep = ",")
happiness_data = happiness_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6','D']]
X_happiness = happiness_data.iloc[:,:-1]
y_happiness = happiness_data.iloc[:,-1]

#%% Read and prepare data mammographic\mammographic CLASSIFICATION
mammographic_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/mammographic/mammographic_masses.csv", sep = ",", header=None, names=['BI-RADS','Age','Shape','Margin', 'Density', 'Severity'])

for column in mammographic_data.columns:
	mode_value = mammographic_data[column].mode()
	mammographic_data[column] = mammographic_data[column].replace('?', mode_value[0]) 

mammographic_data = mammographic_data.astype({'BI-RADS': 'int64','Age': 'int64','Shape': 'int64','Margin': 'int64', 'Density': 'int64'})

X_mammographic = mammographic_data.iloc[:,:-1]
y_mammographic = mammographic_data.iloc[:,-1]

#%% Read and prepare data airfoil_self_noise REGRESSION
airfoil_columns = ['Frequency','AngleOfAttack','ChordLength','FreeStreamVelocity','SuctionSideDisplacementThickness', 'SoundPressureLevel'] 
airfoil_self_noise_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/airfoil_self_noise/airfoil_self_noise.csv", sep = "	", header=None, names=airfoil_columns)

X_airfoil_self_noise = airfoil_self_noise_data.iloc[:,:-1]
y_airfoil_self_noise = airfoil_self_noise_data.iloc[:,-1]

#%% Read and prepare data machine performance REGRESSION
machine_performance_columns = ['VendorName', 'ModeName','MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP','ERP_label']
machine_performance_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/machine-performance/machine.csv", sep = ",", header=None, names=machine_performance_columns)
machine_categoric_data = pd.get_dummies(machine_performance_data.iloc[:,0:2])
machine_numeric_data = machine_performance_data.iloc[:,2:10]
prepared_machine_data = pd.concat([machine_categoric_data, machine_numeric_data], axis=1, sort=False)

X_machine_data = prepared_machine_data.iloc[:,:-1]
y_machine_data = prepared_machine_data.iloc[:,-1]

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

#%% Read and prepare data TIC-TAC-TOC set
colnames=['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square','middle-middle-square','middle-right-square','bottom-left-square','bottom-middle-square','bottom-right-square', 'x_win'] 
tic_tac_mapping = {'positive':1, 'negative':0}
tic_tac_toc_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/tic-tac-toe/tic-tac-toe.csv", sep = ",", names=colnames, header=None)
tic_tac_toc_data['x_win'] = tic_tac_toc_data['x_win'].map(tic_tac_mapping) # change output variable
tic_tac_toc_label = tic_tac_toc_data['x_win']
tic_tac_toc_numerical = pd.get_dummies(tic_tac_toc_data.iloc[:,:-1])
tic_tac_toc_numerical['x_win'] = tic_tac_toc_label
tic_tac_toc_numerical.info()

#%% Read and prepare data HEART DISIEASE set
heart_disiease_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/heart.csv", sep = ",")


#%% Read and prepare data SKIN Segmentation 
skin_segmentation_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/skin/Skin_NonSkin.csv", sep = "	", header=None, names=['Blue','Green','Red','IsSkin'])

#%% Read and prepare data happines
happiness_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/happiness/happiness.csv", sep = ",")
happiness_data = happiness_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6','D']]
happiness_data.info()

#%% Read and prepare data mammographic\mammographic
mammographic_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/mammographic/mammographic_masses.csv", sep = ",", header=None, names=['BI-RADS','Age','Shape','Margin', 'Density', 'Severity'])

for column in mammographic_data.columns:
	mode_value = mammographic_data[column].mode()
	mammographic_data[column] = mammographic_data[column].replace('?', mode_value[0]) 

mammographic_data = mammographic_data.astype({'BI-RADS': 'int64','Age': 'int64','Shape': 'int64','Margin': 'int64', 'Density': 'int64'})
mammographic_data.info()
#%% 
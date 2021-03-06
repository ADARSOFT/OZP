import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from vecstack import stacking
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import time

#%% Global configuration part

seed = 7
test_size = 0.3

def doCrossValidation(alg_, X_train, Y_train):
	return cross_validate(alg_, X_train, Y_train, cv=10, n_jobs=-1, return_train_score=True, scoring = ['precision', 'recall','accuracy','f1'])

def kFoldMeanSquaredErrorHelper(X,y,pipe,folds = 10):
	start_time = time.time()
	kfold = KFold(folds, True, 1)
	average_mse = 0
	for train, test in kfold.split(X, y):
		pipe.fit(X.iloc[train, :], y.iloc[train])
		pred = [round(value) for value in pipe.predict(X.iloc[test,:])]
		mse = sum((y.iloc[test] - pred)**2)/len(pred)
		average_mse += mse
	print(time.time()-start_time)
	return average_mse/folds

#%% Read and prepare data TIC-TAC-TOC set CLASSIFICATION
colnames=['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square','middle-middle-square','middle-right-square','bottom-left-square','bottom-middle-square','bottom-right-square', 'x_win'] 
tic_tac_mapping = {'positive':1, 'negative':0}
tic_tac_toc_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/tic-tac-toe/tic-tac-toe.csv", sep = ",", names=colnames, header=None)
tic_tac_toc_data['x_win'] = tic_tac_toc_data['x_win'].map(tic_tac_mapping) # change output variable
tic_tac_toc_label = tic_tac_toc_data['x_win']
tic_tac_toc_numerical = pd.get_dummies(tic_tac_toc_data.iloc[:,:-1])
tic_tac_toc_numerical['x_win'] = tic_tac_toc_label
X_tic_tac_toc = tic_tac_toc_numerical.iloc[:,:-1]
y_tic_tac_toc = tic_tac_toc_numerical['x_win'] 

X_ttt_train, X_ttt_test, y_ttt_train, y_ttt_test = train_test_split(X_tic_tac_toc, y_tic_tac_toc, test_size=test_size, random_state=seed)

#%% Read and prepare data HEART DISIEASE set CLASSIFICATION
heart_disiease_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/heart.csv", sep = ",")
X_heart = heart_disiease_data.iloc[:,:-1]
y_heart = heart_disiease_data.iloc[:,-1]

X_h_train, X_h_test, y_h_train, y_h_test = train_test_split(X_heart, y_heart, test_size=test_size, random_state=seed)

#%% Read and prepare data SKIN SegmentationCLASSIFICATION
skin_segmentation_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/skin/Skin_NonSkin.csv", sep = "	", header=None, names=['Blue','Green','Red','IsSkin'])
X_skin_segmentation = skin_segmentation_data.iloc[:, :-1]
y_skin_segmentation = skin_segmentation_data.iloc[:, -1]

X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(X_skin_segmentation, y_skin_segmentation, test_size=test_size, random_state=seed)

#%% Read and prepare data happines CLASSIFICATION
happiness_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/happiness/happiness.csv", sep = ",")
happiness_data = happiness_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6','D']]
X_happiness = happiness_data.iloc[:,:-1]
y_happiness = happiness_data.iloc[:,-1]

X_happ_train, X_happ_test, y_happ_train, y_happ_test = train_test_split(X_happiness, y_happiness, test_size=test_size, random_state=seed)

#%% Read and prepare data mammographic\mammographic CLASSIFICATION
mammographic_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/mammographic/mammographic_masses.csv", sep = ",", header=None, names=['BI-RADS','Age','Shape','Margin', 'Density', 'Severity'])

for column in mammographic_data.columns:
	mode_value = mammographic_data[column].mode()
	mammographic_data[column] = mammographic_data[column].replace('?', mode_value[0]) 

mammographic_data = mammographic_data.astype({'BI-RADS': 'int64','Age': 'int64','Shape': 'int64','Margin': 'int64', 'Density': 'int64'})

X_mammographic = mammographic_data.iloc[:,:-1]
y_mammographic = mammographic_data.iloc[:,-1]

X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_mammographic, y_mammographic, test_size=test_size, random_state=seed)

#%% Read and prepare data airfoil_self_noise REGRESSION
airfoil_columns = ['Frequency','AngleOfAttack','ChordLength','FreeStreamVelocity','SuctionSideDisplacementThickness', 'SoundPressureLevel'] 
airfoil_self_noise_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/airfoil_self_noise/airfoil_self_noise.csv", sep = "	", header=None, names=airfoil_columns)

X_airfoil_self_noise = airfoil_self_noise_data.iloc[:,:-1]
y_airfoil_self_noise = airfoil_self_noise_data.iloc[:,-1]

X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(X_airfoil_self_noise, y_airfoil_self_noise, test_size=test_size, random_state=seed)

#%% Read and prepare data machine performance REGRESSION
machine_performance_columns = ['VendorName', 'ModeName','MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP','ERP_label']
machine_performance_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/machine-performance/machine.csv", sep = ",", header=None, names=machine_performance_columns)
machine_categoric_data = pd.get_dummies(machine_performance_data.iloc[:,0:2])
machine_numeric_data = machine_performance_data.iloc[:,2:10]
prepared_machine_data = pd.concat([machine_categoric_data, machine_numeric_data], axis=1, sort=False)

X_machine_data = prepared_machine_data.iloc[:,:-1]
y_machine_data = prepared_machine_data.iloc[:,-1]

X_mch_train, X_mch_test, y_mch_train, y_mch_test = train_test_split(X_machine_data, y_machine_data, test_size=test_size, random_state=seed)

#%% Read and prepare data red wine REGRESSION
red_wine_data = pd.read_csv("C:/Work/Repo_OZP/Data/kaggle/winequality-red.csv", sep = ",")
X_red_wine_data = red_wine_data.iloc[:,:-1]
y_red_wine_data = red_wine_data.iloc[:,-1]
X_red_wine_data.info()

X_rw_train, X_rw_test, y_rw_train, y_rw_test = train_test_split(X_red_wine_data, y_red_wine_data, test_size=test_size, random_state=seed)
#%% Read and prepare data parkinson REGRESSION

parkinson_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/parkinson/parkinsons_updrs.csv", sep = ",")
X_parkinson_data = parkinson_data.iloc[:,:-1]
y_parkinson_data = parkinson_data.iloc[:,-1]

X_par_train, X_par_test, y_par_train, y_par_test = train_test_split(X_parkinson_data, y_parkinson_data, test_size=test_size, random_state=seed)

#%% Read and prepare data CREDIT CARD FRAUD DETECTION - CLASSIFICATION
credit_card_fraud_data = pd.read_csv("C:/Work/Repo_OZP/Data/kaggle/creditcard.csv", sep = ",")
credit_card_fraud_data['Class'].value_counts()
# Veliki disbalans klasa
#0    284315
#1       492
X_cc_fraud_data = credit_card_fraud_data.iloc[:,:-1]
y_cc_fraud_data = credit_card_fraud_data.iloc[:,-1]
X_ccf_train, X_ccf_test, y_ccf_train, y_ccf_test = train_test_split(X_cc_fraud_data, y_cc_fraud_data, test_size=test_size, random_state=seed)

#%% Read and prepare data IOT traffic
iot_danmini_doorbell_benign_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/iot/danmini_doorbell/benign_traffic.csv", sep = ",")
iot_danmini_doorbell_benign_data['IsMalignAttack'] = 0;
iot_danmini_doorbell_malign_gafg_tcp_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/iot/danmini_doorbell/gafgt_attack/tcp.csv", sep = ",")
iot_danmini_doorbell_malign_gafg_tcp_data['IsMalignAttack'] = 1
iot_danmini_doorbell_malign_gafg_scan_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/iot/danmini_doorbell/gafgt_attack/scan.csv", sep = ",")
iot_danmini_doorbell_malign_gafg_scan_data['IsMalignAttack'] = 1
iot_danmini_doorbell_malign_gafg_udp_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/iot/danmini_doorbell/gafgt_attack/udp.csv", sep = ",")
iot_danmini_doorbell_malign_gafg_udp_data['IsMalignAttack'] = 1
iot_danmini_doorbell_malign_gafg_combo_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/iot/danmini_doorbell/gafgt_attack/combo.csv", sep = ",")
iot_danmini_doorbell_malign_gafg_combo_data['IsMalignAttack'] = 1
iot_danmini_doorbell_malign_gafg_junk_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/iot/danmini_doorbell/gafgt_attack/junk.csv", sep = ",")
iot_danmini_doorbell_malign_gafg_junk_data['IsMalignAttack'] = 1
iot_danmini_doorbell_malign_mirai_ack_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/iot/danmini_doorbell/mirai_attack/ack.csv", sep = ",")
iot_danmini_doorbell_malign_mirai_ack_data['IsMalignAttack'] = 1
iot_danmini_doorbell_malign_mirai_scan_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/iot/danmini_doorbell/mirai_attack/scan.csv", sep = ",")
iot_danmini_doorbell_malign_mirai_scan_data['IsMalignAttack'] = 1
iot_danmini_doorbell_malign_mirai_udp_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/iot/danmini_doorbell/mirai_attack/udp.csv", sep = ",")
iot_danmini_doorbell_malign_mirai_udp_data['IsMalignAttack'] = 1
iot_danmini_doorbell_malign_mirai_syn_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/iot/danmini_doorbell/mirai_attack/syn.csv", sep = ",")
iot_danmini_doorbell_malign_mirai_syn_data['IsMalignAttack'] = 1
iot_danmini_doorbell_malign_mirai_udpplain_data = pd.read_csv("C:/Work/Repo_OZP/Data/uci/iot/danmini_doorbell/mirai_attack/udpplain.csv", sep = ",")
iot_danmini_doorbell_malign_mirai_udpplain_data['IsMalignAttack'] = 1

frames = [iot_danmini_doorbell_benign_data, 
		  iot_danmini_doorbell_malign_gafg_tcp_data, 
		  iot_danmini_doorbell_malign_gafg_scan_data,
		  iot_danmini_doorbell_malign_gafg_udp_data,
		  iot_danmini_doorbell_malign_gafg_combo_data,
		  iot_danmini_doorbell_malign_gafg_junk_data,
		  iot_danmini_doorbell_malign_mirai_ack_data,
		  iot_danmini_doorbell_malign_mirai_scan_data,
		  iot_danmini_doorbell_malign_mirai_udp_data,
		  iot_danmini_doorbell_malign_mirai_syn_data,
		  iot_danmini_doorbell_malign_mirai_udpplain_data
		  ]

iot_danmini_doorbell_full_data = pd.concat(frames)

del iot_danmini_doorbell_benign_data
del iot_danmini_doorbell_malign_gafg_scan_data
del iot_danmini_doorbell_malign_gafg_udp_data
del iot_danmini_doorbell_malign_gafg_combo_data
del iot_danmini_doorbell_malign_gafg_junk_data
del iot_danmini_doorbell_malign_mirai_ack_data
del iot_danmini_doorbell_malign_mirai_scan_data
del iot_danmini_doorbell_malign_mirai_udp_data
del iot_danmini_doorbell_malign_mirai_syn_data
del iot_danmini_doorbell_malign_mirai_udpplain_data

# 1 -> 968750
# 0 -> 49548

X_iot_train, X_iot_test, y_iot_train, y_iot_test = train_test_split(iot_danmini_doorbell_full_data.iloc[:,:-1], iot_danmini_doorbell_full_data.iloc[:,-1], test_size=test_size, random_state=seed)

#%% Running models part 

# XGBoost CLASSIFIER measurements

xgb_classifier_pipe_steps = [('minMax', MinMaxScaler()), ('xgbClassifier', XGBClassifier())] 
xgb_classifier_pipe = Pipeline(steps = xgb_classifier_pipe_steps)

xgb_result_mammographia = pd.DataFrame(doCrossValidation(xgb_classifier_pipe, X_m_train, y_m_train)).mean()
xgb_result_skin = pd.DataFrame(doCrossValidation(xgb_classifier_pipe, X_s_train, y_s_train)).mean()
xgb_result_heart = pd.DataFrame(doCrossValidation(xgb_classifier_pipe, X_h_train, y_h_train)).mean()
xgb_result_tic_tac_toc = pd.DataFrame(doCrossValidation(xgb_classifier_pipe, X_ttt_train, y_ttt_train)).mean()
xgb_result_credit_card_fraud = pd.DataFrame(doCrossValidation(xgb_classifier_pipe, X_ccf_train, y_ccf_train)).mean()
xgb_result_iot = pd.DataFrame(doCrossValidation(xgb_classifier_pipe, X_iot_train, y_iot_train)).mean()

# XG BOOST REGRESSOR measurements

xgb_regressor_pipe_steps = [('minMax', MinMaxScaler()), ('xgbRegressor', XGBRegressor())] 
xgb_regressor_pipe = Pipeline(steps = xgb_regressor_pipe_steps)

xgboost_regressor_params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1, 'max_depth': 5, 'alpha': 10}
total_boosting_rounds = 100

mch_data_dmatrix = xgb.DMatrix(data = X_mch_train, label = y_mch_train)
mch_cv_results = pd.DataFrame(xgb.cv(dtrain=mch_data_dmatrix, params=xgboost_regressor_params, nfold=3, num_boost_round=total_boosting_rounds,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123))

airfoil_data_dmatrix = xgb.DMatrix(data = X_a_train, label = y_a_train)
airfoil_cv_results = pd.DataFrame(xgb.cv(dtrain=airfoil_data_dmatrix, params=xgboost_regressor_params, nfold=3, num_boost_round=total_boosting_rounds,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123))

avg_mse_xgb_airfoil = kFoldMeanSquaredErrorHelper(X_a_train, y_a_train, xgb_regressor_pipe)	
avg_mse_xgb_machine = kFoldMeanSquaredErrorHelper(X_mch_train, y_mch_train, xgb_regressor_pipe)	
avg_mse_xgb_parkinson = kFoldMeanSquaredErrorHelper(X_par_train, y_par_train, xgb_regressor_pipe)	
avg_mse_xgb_redwine = kFoldMeanSquaredErrorHelper(X_rw_train, y_rw_train, xgb_regressor_pipe)	


#%%
# RANDOM FOREST CLASSIFIER 

rfc_classifier_pipe_steps = [('minMax', MinMaxScaler()), ('rfClassifier', RandomForestClassifier())] 
rfc_classifier_pipe = Pipeline(steps = rfc_classifier_pipe_steps)

rfc_result_mammographia = pd.DataFrame(doCrossValidation(rfc_classifier_pipe, X_m_train, y_m_train)).mean()
rfc_result_skin = pd.DataFrame(doCrossValidation(rfc_classifier_pipe, X_s_train, y_s_train)).mean()
rfc_result_heart = pd.DataFrame(doCrossValidation(rfc_classifier_pipe, X_h_train, y_h_train)).mean()
rfc_result_tic_tac_toc = pd.DataFrame(doCrossValidation(rfc_classifier_pipe, X_ttt_train, y_ttt_train)).mean()
rfc_result_credit_card_fraud = pd.DataFrame(doCrossValidation(rfc_classifier_pipe, X_ccf_train, y_ccf_train)).mean()
rfc_result_iot = pd.DataFrame(doCrossValidation(rfc_classifier_pipe, X_iot_train, y_iot_train)).mean()

# RANDOM FOREST REGRESSOR

rfr_classifier_pipe_steps = [('minMax', MinMaxScaler()), ('rfRegressor', RandomForestRegressor())] 
rfr_classifier_pipe = Pipeline(steps = rfr_classifier_pipe_steps)

rfc_a_val_score = pd.DataFrame(cross_val_score(rfr_classifier_pipe, X_a_train, y_a_train, cv=100, scoring='neg_mean_squared_error')).abs()
rfc_mch_val_score = pd.DataFrame(cross_val_score(rfr_classifier_pipe, X_mch_train, y_mch_train, cv=100, scoring='neg_mean_squared_error')).abs()

avg_mse_rfc_airfoil = kFoldMeanSquaredErrorHelper(X_a_train, y_a_train, rfr_classifier_pipe)	
avg_mse_rfc_machine = kFoldMeanSquaredErrorHelper(X_mch_train, y_mch_train, rfr_classifier_pipe)	
avg_mse_rfc_parkinson = kFoldMeanSquaredErrorHelper(X_par_train, y_par_train, rfr_classifier_pipe)	
avg_mse_rfc_redwine = kFoldMeanSquaredErrorHelper(X_rw_train, y_rw_train, rfr_classifier_pipe)	

# ADABOOST CLASSIFIER

ab_classifier_pipe_steps = [('minMax', MinMaxScaler()), ('abClassifier', AdaBoostClassifier())] 
ab_classifier_pipe = Pipeline(steps = ab_classifier_pipe_steps)

ab_result_mammographia = pd.DataFrame(doCrossValidation(ab_classifier_pipe, X_m_train, y_m_train)).mean()
ab_result_skin = pd.DataFrame(doCrossValidation(ab_classifier_pipe, X_s_train, y_s_train)).mean()
ab_result_heart = pd.DataFrame(doCrossValidation(ab_classifier_pipe, X_h_train, y_h_train)).mean()
ab_result_tic_tac_toc = pd.DataFrame(doCrossValidation(ab_classifier_pipe, X_ttt_train, y_ttt_train)).mean()
ab_result_credit_card_fraud = pd.DataFrame(doCrossValidation(ab_classifier_pipe, X_ccf_train, y_ccf_train)).mean()
ab_result_iot = pd.DataFrame(doCrossValidation(ab_classifier_pipe, X_iot_train, y_iot_train)).mean()

# ADABOOST REGRESSOR
ab_regressor_pipe_steps = [('minMax', MinMaxScaler()), ('abRegressor', AdaBoostRegressor())] 
ab_regressor_pipe = Pipeline(steps = ab_regressor_pipe_steps)

avg_mse_ab_airfoil = kFoldMeanSquaredErrorHelper(X_a_train, y_a_train, ab_regressor_pipe)	
avg_mse_ab_machine = kFoldMeanSquaredErrorHelper(X_mch_train, y_mch_train, ab_regressor_pipe)
avg_mse_ab_parkinson = kFoldMeanSquaredErrorHelper(X_par_train, y_par_train, ab_regressor_pipe)	
avg_mse_ab_redwine = kFoldMeanSquaredErrorHelper(X_rw_train, y_rw_train, ab_regressor_pipe)

# GRADIENT BOOSTING CLASSIFIER

gb_classifier_pipe_steps = [('minMax', MinMaxScaler()), ('gbClassifier', GradientBoostingClassifier())] 
gb_classifier_pipe = Pipeline(steps = gb_classifier_pipe_steps)

gb_result_mammographia = pd.DataFrame(doCrossValidation(gb_classifier_pipe, X_m_train, y_m_train)).mean()
gb_result_skin = pd.DataFrame(doCrossValidation(gb_classifier_pipe, X_s_train, y_s_train)).mean()
gb_result_heart = pd.DataFrame(doCrossValidation(gb_classifier_pipe, X_h_train, y_h_train)).mean()
gb_result_tic_tac_toc = pd.DataFrame(doCrossValidation(gb_classifier_pipe, X_ttt_train, y_ttt_train)).mean()
gb_result_credit_card_fraud = pd.DataFrame(doCrossValidation(gb_classifier_pipe, X_ccf_train, y_ccf_train)).mean()
gb_result_iot = pd.DataFrame(doCrossValidation(gb_classifier_pipe, X_iot_train, y_iot_train)).mean()

# GRADIENT BOOSTING REGRESSOR
gb_regressor_pipe_steps = [('minMax', MinMaxScaler()), ('gbRegressor', GradientBoostingRegressor())] 
gb_regressor_pipe = Pipeline(steps = gb_regressor_pipe_steps)

avg_mse_gb_airfoil = kFoldMeanSquaredErrorHelper(X_a_train, y_a_train, gb_regressor_pipe)	
avg_mse_gb_machine = kFoldMeanSquaredErrorHelper(X_mch_train, y_mch_train, gb_regressor_pipe)
avg_mse_gb_parkinson = kFoldMeanSquaredErrorHelper(X_par_train, y_par_train, gb_regressor_pipe)
avg_mse_gb_redwine = kFoldMeanSquaredErrorHelper(X_rw_train, y_rw_train, gb_regressor_pipe)

# STACKING CLASSIFIER

stacking_models = [
    KNeighborsClassifier(n_neighbors=5, n_jobs=-1),   
    RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=100, max_depth=3),
    XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, n_estimators=100, max_depth=3)
]

meta_model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1,  n_estimators=100, max_depth=3)

m_train, m_test = stacking(stacking_models, X_m_train, y_m_train, X_m_test, regression=False, mode='oof_pred_bag',needs_proba=False, save_dir=None,metric=accuracy_score,n_folds=4,stratified=True,shuffle=True,random_state=0,verbose=2)
fit_m_meta_model = meta_model.fit(m_train, y_m_train)
m_y_pred = fit_m_meta_model.predict(m_test)
print('Final prediction score: [%.8f]' % accuracy_score(y_m_test, m_y_pred))

s_train, s_test = stacking(stacking_models, X_s_train, y_s_train, X_s_test, regression=False, mode='oof_pred_bag',needs_proba=False, save_dir=None,metric=accuracy_score,n_folds=4,stratified=True,shuffle=True,random_state=0,verbose=2)
fit_s_meta_model = meta_model.fit(s_train, y_s_train)
s_y_pred = fit_s_meta_model.predict(s_test)
print('Final prediction score: [%.8f]' % accuracy_score(y_s_test, s_y_pred))

h_train, h_test = stacking(stacking_models, X_h_train, y_h_train, X_h_test, regression=False, mode='oof_pred_bag',needs_proba=False, save_dir=None,metric=accuracy_score,n_folds=4,stratified=True,shuffle=True,random_state=0,verbose=2)
fit_h_meta_model = meta_model.fit(h_train, y_h_train)
h_y_pred = fit_h_meta_model.predict(h_test)
print('Final prediction score: [%.8f]' % accuracy_score(y_h_test, h_y_pred))

ttt_train, ttt_test = stacking(stacking_models, X_ttt_train, y_ttt_train, X_ttt_test, regression=False, mode='oof_pred_bag',needs_proba=False, save_dir=None,metric=accuracy_score,n_folds=4,stratified=True,shuffle=True,random_state=0,verbose=2)
fit_ttt_meta_model = meta_model.fit(ttt_train, y_ttt_train)
ttt_y_pred = fit_ttt_meta_model.predict(ttt_test)
print('Final prediction score: [%.8f]' % accuracy_score(y_ttt_test, ttt_y_pred))

iot_train, iot_test = stacking(stacking_models, X_iot_train, y_iot_train, X_iot_test, regression=False, mode='oof_pred_bag',needs_proba=False, save_dir=None,metric=accuracy_score,n_folds=4,stratified=True,shuffle=True,random_state=0,verbose=2)
fit_iot_meta_model = meta_model.fit(iot_train, y_iot_train)
iot_y_pred = fit_iot_meta_model.predict(iot_test)
print('Final prediction score: [%.8f]' % accuracy_score(y_iot_test, iot_y_pred))


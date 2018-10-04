###  Jamie Singleton 10/2018

## Next Steps
# /0 Error in Hits/Pageviews
# Use a time-series split cross validation so information from future events don't leak
# Use multiple regression model types (catboost, etc...)
# Use Bayesian hyperparameter optimization

## Sources/Inspiration
# https://www.kaggle.com/ashishpatel26/1-43-plb-feature-engineering-best-model-combined/notebook
# https://www.kaggle.com/xavierbourretsicotte/light-gbm-baseline

# https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields
def load_dataframe(csv_path, nrows=None):
		
	DTYPE_REPLACEMENTS = {'fullVisitorId': str}
	JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

	df = pd.read_csv(csv_path, 
					 converters={column: json.loads for column in JSON_COLUMNS}, 
					 dtype=DTYPE_REPLACEMENTS,
					 nrows=nrows)
	
	# PARSE JSON_COLUMNS
	for column in JSON_COLUMNS:
		column_as_df = json_normalize(df[column])
		column_as_df.columns = [f'{column}.{subcolumn}' for subcolumn in column_as_df.columns]
		df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

	print(f'Loaded {os.path.basename(csv_path)}. Shape: {df.shape}')
	
	return(df)

def process_format(df):

	print('Processing formats')

	# Set format to float
	for col in ['visitID', 'visitNumber', 'visitStartTime', 'totals.bounces', 'totals.hits', 'totals.newVisits', 'totals.pageviews', 'totals.transactionRevenue']:
		if col in df.columns:
			df[col] = df[col].astype(float)

	print('Processed formats')

	return df  

def process_na(df):

	print('Processing na columns')

	#print(df.columns[df.isna().any()].tolist())

	# Fill/Drop columns that have 1 constant value and NAs
	na_const_cols = [col for col in train_df.columns if train_df[col].nunique(dropna=True)==1 ]
	na_const_cols = ['totals.bounces', 'totals.newVisits', 'trafficSource.adwordsClickInfo.adNetworkType', 'trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.adwordsClickInfo.page', 'trafficSource.isTrueDirect']
	na_const_cols_replacements = [0, 0, 'Not Google Search', 'DROP THIS FIELD', 'DROP THIS FIELD', False]
	for idx, na_const_col in enumerate(na_const_cols):
		if na_const_col in df.columns:
			replacement_value = na_const_cols_replacements[idx]
			if replacement_value == 'DROP THIS FIELD':
				df.drop(na_const_col, axis=1, inplace=True)
			else:
				df[na_const_col].fillna(replacement_value, inplace=True)

	# Fill/Drop columns that have 2+ values and NAs
	na_cols = df.columns[df.isna().any()].tolist()
	na_cols = ['totals.pageviews', 'totals.transactionRevenue', 'trafficSource.adContent', 'trafficSource.adwordsClickInfo.slot', 'trafficSource.keyword', 'trafficSource.referralPath']
	na_cols_replacements = [0, 0, 'No AdContent', 'Not Google Search = Unknown Slot', '(not provided)', False]
	for idx, na_col in enumerate(na_cols):
		if na_col in df.columns:
			replacement_value = na_cols_replacements[idx]
			if replacement_value == 'DROP THIS FIELD':
				df.drop(na_col, axis=1, inplace=True)
			else:
				df[na_col].fillna(replacement_value, inplace=True)

	#print(df.columns[df.isna().any()].tolist())

	print('Processed na columns')

	return df  
	
def add_datetime_fields(df):

	print('Adding datetime fields')

	df['date'] = df['date'].astype(str)
	df['date'] = df['date'].apply(lambda x : x[:4] + '-' + x[4:6] + '-' + x[6:])
	df['date'] = pd.to_datetime(df['date'])  

	df['year'] = df['date'].dt.year

	df['month'] = df['date'].dt.month

	df['week_of_the_year'] = df['date'].dt.weekofyear

	df['day_of_the_week'] = df['date'].dt.weekday
	df['day_of_the_month'] = df['date'].dt.day
	df['day_of_the_year'] = df['date'].dt.dayofyear

	df['hour'] = df['date'].dt.hour

	# Remove date since key components have been extracted
	df.drop(['date'], axis=1, inplace=True)

	print('Added datetime fields')

	return(df)

def process_device(df):

	print("process device ...")

	df['browser_category'] = df['device.browser'] + '_' + df['device.deviceCategory']
	df['browser_operatingSystem'] = df['device.browser'] + '_' + df['device.operatingSystem']

	return df

def process_totals(df):

	print("process totals ...")

	df['mean_hits_per_day'] = df.groupby(['day_of_the_year'])['totals.hits'].transform('mean')
	df['sum_hits_per_day'] = df.groupby(['day_of_the_year'])['totals.hits'].transform('sum')
	df['max_hits_per_day'] = df.groupby(['day_of_the_year'])['totals.hits'].transform('max')
	df['min_hits_per_day'] = df.groupby(['day_of_the_year'])['totals.hits'].transform('min')
	df['var_hits_per_day'] = df.groupby(['day_of_the_year'])['totals.hits'].transform('var')

	df['totals.hits/totals.pageviews'] = df['totals.hits'] / df['totals.pageviews']

	return df

def process_geo_network(df):

	print("process geo network ...")

	df['sum_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('sum')
	df['count_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('count')
	df['mean_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('mean')

	df['sum_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
	df['count_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('count')
	df['mean_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('mean')

	return df

def encode_labels(train_df, test_df):

	from sklearn.preprocessing import LabelEncoder

	num_cols = list(train_df.select_dtypes(include=['number']).columns.values)
	date_cols = list(train_df.select_dtypes(include=['datetime']).columns.values)

	cat_cols = [col for col in train_df.columns if (col not in num_cols) and (col not in date_cols)]

	for col in cat_cols:
	    
	    #print(col)

	    lbl = LabelEncoder()
	    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))

	    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
	    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

	return (train_df, test_df)

def display_heatmap(train_df):

	plt.figure(figsize=(20,20))
	sns.heatmap(train_df.corr(),annot=True)
	plt.show()

def create_corr_df(x_cols, train_df, target_col):

	labels = []
	values = []

	for col in x_cols:
	    labels.append(col)
	    values.append(np.corrcoef(train_df_new[col].values, train_df_new[target_col].values)[0,1])

	corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
	corr_df = corr_df.sort_values(by='corr_values')

	return(corr_df)

def display_corr_df(corr_df):

	ind = np.arange(len(corr_df.col_labels))
	width = 0.9
	fig, ax = plt.subplots(figsize=(12,40))

	rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')
	ax.set_yticks(ind)
	ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
	ax.set_xlabel("Correlation coefficient")
	ax.set_title("Correlation coefficient of the variables")

	plt.show()

def create_polynomical_variable(corr_df, corr_cutoff, X_train, X_test):
	corr_df_sel = corr_df[(corr_df['corr_values']>corr_cutoff)]
	x_cols = corr_df_sel['col_labels'].values
	for i in x_cols:
	    X_train[i+'_squared'] =  X_train[i] ** 2
	    X_test[i+'_squared'] = X_test[i] ** 2
	return((X_train, X_test))

from catboost import CatBoostRegressor
import json
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas.io.json import json_normalize
import seaborn as sns
from sklearn.model_selection import KFold
import xgboost as xgb

## Load Data

#input_folder = '../input'
input_folder = 'input'
n_rows = None
#n_rows = 100000
train_df = load_dataframe(os.path.join(input_folder, 'train.csv'), n_rows)
test_df = load_dataframe(os.path.join(input_folder, 'test.csv'), n_rows)

## Pre-processing

target_col = 'totals.transactionRevenue'
target_groupby_col = 'fullVisitorId'

# Remove columns that are constant in train_df
const_cols = [col for col in train_df.columns if train_df[col].nunique(dropna=False)==1 ]
train_df.drop(const_cols, axis=1, inplace=True)
const_cols = [col for col in const_cols if col in test_df.columns]
test_df.drop(const_cols, axis=1, inplace=True)

# Remove columns in train_df but not in test_def
missing_cols = list(set(train_df.columns).difference(set(test_df.columns)))
missing_cols = [ col for col in missing_cols if col != target_col]
train_df.drop(missing_cols, axis=1, inplace=True)

# Remove identifier columns from train_df and test_df
# Do not remove the target_groupby_col
id_cols = ['sessionId', 'trafficSource.adwordsClickInfo.gclId']
train_df.drop(id_cols, axis=1, inplace=True)
test_df.drop(id_cols, axis=1, inplace=True)

# Fill NAs
train_df = process_na(train_df)
test_df = process_na(test_df)

# Fix formats
train_df = process_format(train_df)
test_df = process_format(test_df)

## Feature Engineering

train_df = add_datetime_fields(train_df)
test_df = add_datetime_fields(test_df)

train_df = process_device(train_df)
train_df = process_totals(train_df)
train_df = process_geo_network(train_df)

test_df = process_device(test_df)
test_df = process_totals(test_df)
test_df = process_geo_network(test_df)

## Label Encoding

(train_df, test_df) = encode_labels(train_df, test_df)

## Model

# unused_cols must contain the target column and target_groupby column as a minimum.
unused_cols = [target_col, target_groupby_col]
X_train = train_df.drop(unused_cols, axis=1)
y_train = np.log1p(train_df[target_col])
target_train = np.log1p(train_df.groupby(target_groupby_col)[target_col].sum())
X_test = test_df.drop([col for col in unused_cols if col in test_df.columns], axis=1)

# Correlation of features

display_heatmap(train_df)

train_df_new = train_df.copy()

# Change data type for numeric fields to match target column data type
target_col_type = train_df_new[target_col].dtype
num_cols = list(train_df_new.select_dtypes(include=['number']).columns.values)
for col in num_cols:
	train_df_new[col] = train_df_new[col].astype(target_col_type) 
x_cols = [col for col in train_df_new.columns if (col not in [target_col]) and train_df_new[col].dtype==target_col_type]

corr_df = create_corr_df(x_cols, train_df, target_col)

display_corr_df(corr_df)

corr_cutoff = 0.03
(X_train, X_test) = create_polynomical_variable(corr_df, corr_cutoff, X_train, X_test)

## Folds

folds = KFold(n_splits=10, shuffle=True, random_state=42)

## LightGBM

lightgbm_folds = KFold(n_splits=10, shuffle=True, random_state=42)

lightgbm_params = {
			"objective" : "regression", 
			"metric" : "rmse", 
			"num_leaves" : 30, 
			"learning_rate" : 0.01, 
			"bagging_fraction" : 0.9,
			"feature_fraction" : 0.3, 
			"bagging_seed" : 0}

NUM_ROUNDS = 20000
VERBOSE_EVAL = 1000
STOP_ROUNDS = 100

lgb_model = lgb.LGBMRegressor(**lightgbm_params, n_estimators = NUM_ROUNDS, nthread = 4, n_jobs = -1)

prediction = np.zeros(test_df.shape[0])

for fold_n, (train_index, test_index) in enumerate(lightgbm_folds.split(X_train)):
    print('Fold:', fold_n)
    X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    
    lgb_model.fit(X_train_fold, y_train_fold, 
            eval_set=[(X_train_fold, y_train_fold), (X_valid_fold, y_valid_fold)], eval_metric='rmse',
            verbose=VERBOSE_EVAL, early_stopping_rounds=STOP_ROUNDS)
    
    y_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration_)
    prediction += y_pred

prediction /= 10

## Feature Importance LGBM
fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(lgb_model, max_num_features=50, height=0.8, ax=ax)
lgb_features = lgb_model.feature_importances_
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=20)
plt.show()



# Submission
print("prepare submission ...")
submission = test_df[['fullVisitorId']].copy()
submission.loc[:, 'PredictedLogRevenue'] = prediction
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].fillna(0.0)
grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
grouped_test.to_csv('lgb.csv',index=False)


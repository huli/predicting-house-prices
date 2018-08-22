from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import math
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion, _transform_one
from sklearn.externals.joblib import Parallel, delayed
import locale

def rmse_score(y_t, y_pred):
    return math.sqrt(mean_squared_error(y_t, y_pred))

class NoFitMixin:
    def fit(self, X, y=None):
        return self

# class DFTransform(BaseEstimator, TransformerMixin, NoFitMixin):
class DFTransform(BaseEstimator, TransformerMixin, NoFitMixin):
    def __init__(self, func, copy=False):
        self.func = func
        self.copy = copy

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        return self.func(X_)
    
class DFFeatureUnion(FeatureUnion):
    def fit_transform(self, X, y=None, **fit_params):
        # non-optimized default implementation; override when a better
        # method is possible
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, weight, X)
            for _, trans, weight in self._iter())
        return pd.concat(Xs, axis=1, join='inner')

def prepare_inputs(X_val, y_val):
    outliers = X_val[X_val['GrLivArea'] >= 4000]
    return (X_val.drop(outliers.index), np.log1p(y_val.drop(outliers.index)))

def train_pipeline(transformation_pipeline, estimation_pipeline, size_test=.33, show_plot=True):
    
    print('Loading training data...')
    train_df =  pd.read_csv('data/train.csv')
    X_train = train_df.drop(['SalePrice','Id'], axis=1)
    y_train = train_df['SalePrice']
    X_test = pd.read_csv('data/test.csv').drop(['Id'], axis=1)
    X_train, y_train = prepare_inputs(X_train, y_train)
    
    print('Transforming input...')
    X_combined = pd.concat((X_train, X_test)).reset_index(drop=True) 
    X_tranformed = transformation_pipeline.fit_transform(X_combined)

    print('Create train/test split')
    X_train_trans = X_tranformed[:X_train.shape[0]] 
    X_test_trans = X_tranformed[X_train.shape[0]:]
    X_train, X_test, y_train, y_test = train_test_split(X_train_trans, y_train, test_size=size_test, random_state=42)
    estimation_pipeline.fit(X_train, y_train)
    
    print('Create predictions...(train)')
    predictions = estimation_pipeline.predict(X_train)
    print_benchmark(y_train, predictions)
    
    print('Create predictions...(test)')
    predictions = estimation_pipeline.predict(X_test)
    if show_plot:
        plot_benchmark(X_test, y_test, predictions)
    else:
        print_benchmark(y_test, predictions)
    
    print('Fitting the pipeline to all the data...')
    X_all = train_df.drop(['SalePrice','Id'], axis=1)
    y_all = train_df['SalePrice']
    _, y_all = prepare_inputs(X_all, y_all)
    estimation_pipeline.fit(X_train_trans, y_all)
    print('Score: %.8f' % estimation_pipeline.score(X_train_trans, y_all))
    
    return (transformation_pipeline, estimation_pipeline, X_test_trans)

def impute_special_cases(X_val):
    X_new = X_val.copy()
    
    # According to documentation
    X_new['Functional'] = X_new['Functional'].fillna('Typ')
    
    # Some features we inpute with the mode because NaN/None makes no sense
    for col in ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']:
        X_new[col] = X_new[col].fillna(value=X_new[col].mode()[0])

    return X_new


def fill_nans(X_val, mean_columns, zero_columns):
    ''' Function which replaced the nans in the columns
    with their corresponding median or zero '''
    mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    mean_imputer = mean_imputer.fit(X_val.loc[:, mean_columns])
    X_new = X_val.copy()
    X_new.loc[:, mean_columns] = mean_imputer.transform(X_val.loc[:, mean_columns])
    X_new.loc[:, zero_columns] = X_new.loc[:, zero_columns].fillna(value=0)
    return X_new

def drop_features(X_val):
    return X_val.drop(['Utilities'], axis=1)

def create_dummies(X_val):
    ''' Function which creates dummy variables for all the categoricals
    and hereby take possible resulting multi-collinearity into account'''
    print('Creating dummies...')
    print('Starting with input of shape: %s' % str(X_val.shape))
    X_new = X_val.copy()
    x_extended = pd.get_dummies(X_val, dummy_na=True, drop_first=True)
    print('Returning output of shape: %s' % str(x_extended.shape))
    return x_extended

def fill_numerical_nans(X_val):
    mean_columns = ['LotFrontage']
    zero_columns = ['MasVnrArea', 
                    'GarageYrBlt',
                    'BsmtFinSF1',
                    'BsmtFinSF2',
                    'BsmtUnfSF',
                    'TotalBsmtSF',
                    'BsmtFullBath',
                    'BsmtHalfBath',
                    'GarageCars',
                    'GarageArea']
    X_new = X_val.copy()
    mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    mean_imputer.fit(X_new.loc[:, mean_columns])
    X_new.loc[:, mean_columns] = mean_imputer.transform(X_new.loc[:, mean_columns])
    X_new.loc[:, zero_columns] = X_new.loc[:, zero_columns].fillna(value=0)
    return X_new

def check_nans(X_val):
    null_columns = X_val.columns[X_val.isnull().any(axis=0)]
    if len(null_columns) > 0:
        print('There are still NaNs in the data!')
        print(null_columns)
    return X_val

def impute_categorical(X_val):
    X_new = X_val.copy()
    for col in ['MasVnrType', 'GarageType', 'MiscFeature']:
        X_new[col] = X_new[col].fillna('None')
    return X_new

def execute_pipeline(transformation_pipeline, fitted_pipeline, transformed_x, show_plot=True):
    predictions = fitted_pipeline.predict(transformed_x)
    if show_plot:
        draw_sanity_check(predictions, False)
    return predictions

def encode_ordinals(X_val):
    X_new = X_val.copy()
    for col in ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold']:
        encoder = LabelEncoder()
        encoder.fit(list(X_val[col].values))
        X_new[col] = encoder.transform(list(X_val[col].values))
    return X_new

def rmse_log(y_actual, y_pred):
    assert y_pred.min() <= 0, "Negative price found in prediction! ¯\_(ツ)_/¯"
    return math.sqrt(mean_squared_error(np.log(y_pred), np.log(y_actual)))

def rmse(y_actual, y_pred):
    return math.sqrt(mean_squared_error(y_actual, y_pred))

# Removes outliers and normalized distribution of target variable
def prepare_inputs(X_val, y_val):
    outliers = X_val[X_val['GrLivArea'] >= 4000]
    return (X_val.drop(outliers.index), np.log1p(y_val.drop(outliers.index)))

def print_benchmark(y_actual, y_predicted, log_transform = False):
    print('R2-score: %s' % r2_score(y_actual, y_predicted))
    if log_transform:
        print('RMSE (log): %s' % rmse(np.log1p(y_actual), np.log1p(y_predicted)))
    else:
        print('RMSE (log): %s' % rmse(y_actual, y_predicted))
    
def plot_benchmark(X_t, y_t, pred):
    train_df =  pd.read_csv('data/train.csv')
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].scatter(train_df['GrLivArea'], train_df['SalePrice'], alpha=.4)
    axes[0].set_title('Baseline vs Predictions')
    axes[0].scatter(X_t['GrLivArea'], np.expm1(pred), alpha=.4)
    axes[1].set_title('Baseline vs Predictions')
    axes[1].hist(y_t, alpha=.3)
    axes[1].hist(pred, alpha=.6)
    plt.show()
    print_benchmark(y_t, pred)
    
def write_submission(pred, in_dollars=False):
    df = pd.read_csv('data/test.csv')
    if in_dollars == False:
        pred = np.expm1(pred)
    assert pred.max() > 1000, "Max is smaller than 1000!"
    assert pred.min() > 0, "Min is smaller than 0!"
    submission = pd.DataFrame({'Id': df.Id, 'SalePrice': pred})
    file_path = os.getcwd() + '\\submissions\\' + pd.datetime.now().strftime("%Y%m%d%H%M%S") + '.csv'
    submission.to_csv(file_path, index=False)
    print('File written to %s' % file_path)
    
def draw_sanity_check(pred, in_dollars = True):
    
    locale.setlocale(locale.LC_ALL, 'de-CH')
    thousand_formatter = plt.FuncFormatter(lambda x, _ : locale.format("%d", x, grouping=True))
    if in_dollars == False:
        pred = np.expm1(pred)
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    fig, ax = plt.subplots(1, figsize=(6, 5), sharey=True)
    ax.scatter(train_df['GrLivArea'], train_df['SalePrice'], alpha=.4)
    ax.yaxis.set_major_formatter(thousand_formatter)
    ax.xaxis.set_major_formatter(thousand_formatter)
    ax.set_ylabel('Price in $')
    ax.set_xlabel('Living space (squae feet)')
    ax.set_title('Baseline vs Prediction')
    ax.scatter(test_df['GrLivArea'], pred, alpha=.4)
    print('Mean of Salesprice in Training-Data: %.2f' % train_df['SalePrice'].mean())
    print('Mean of Salesprice in predictions: %.2f' % pred.mean())
    difference = pred.mean() - train_df['SalePrice'].mean()
    print('Difference in means is: %s' % difference)
    if np.abs(difference) > 5000:
        print('IMPORTANT: There is something wrong with your predictions!!!')
    plt.show()
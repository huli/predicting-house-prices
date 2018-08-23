# -*- coding: utf-8 -*-
""" Helper Functions

Reason Contains various often used function for the notebook files.
Author: Christoph Hilty
Date: 23.08.2018

TODO:
* Remove first ununsed parameter of execute pipeline
* Remove duplicate root mean squared error functions
"""

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
from scipy.stats import skew
from scipy.special import boxcox1p


def rmse_score(y_t, y_pred):
    ''' Function which returs the root mean 
    squared errorof the both vectors '''
    return math.sqrt(mean_squared_error(y_t, y_pred))


class NoFitMixin:
    ''' Default mixing which does nothing at all '''
    def fit(self, X, y=None):
        return self

class DFTransform(BaseEstimator, TransformerMixin, NoFitMixin):
    ''' Handles the use of data sets in sklearn pipelines '''
    def __init__(self, func, copy=False):
        self.func = func
        self.copy = copy

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        return self.func(X_)
    
def prepare_inputs(X_val, y_val):
    ''' Function for removing the outliers in the data as 
    well as log transforming the target variable '''
    outliers = X_val[X_val['GrLivArea'] >= 4000]
    return (X_val.drop(outliers.index), np.log1p(y_val.drop(outliers.index)))


def train_pipeline(transformation_pipeline, estimation_pipeline, size_test=.33, show_plot=True):
    ''' Function which executes both the transformation pipeline
    and the estimation pipeline on a train test split as well as
    on the complete training set for submission '''
    
    # Loading data and removing outliers and id column
    print('Loading training data...')
    train_df =  pd.read_csv('data/train.csv')
    X_train = train_df.drop(['SalePrice','Id'], axis=1)
    y_train = train_df['SalePrice']
    X_test = pd.read_csv('data/test.csv').drop(['Id'], axis=1)
    X_train, y_train = prepare_inputs(X_train, y_train)
    
    # Concating  and transforming training and testing set to have a unique 
    # transformation for all the steps (training, testing and submission)
    print('Transforming input...')
    X_combined = pd.concat((X_train, X_test)).reset_index(drop=True) 
    X_tranformed = transformation_pipeline.fit_transform(X_combined)

    # Create train and test split and fitting pipeline to the training set
    print('Create train/test split')
    X_train_trans = X_tranformed[:X_train.shape[0]] 
    X_test_trans = X_tranformed[X_train.shape[0]:]
    X_train, X_test, y_train, y_test = train_test_split(X_train_trans, y_train, test_size=size_test, random_state=42)
    estimation_pipeline.fit(X_train, y_train)
    
    # Create predictions on the training set (biased)
    print('Create predictions...(train)')
    predictions = estimation_pipeline.predict(X_train)
    print_benchmark(y_train, predictions)
    
    # Create predictions on the test set 
    print('Create predictions...(test)')
    predictions = estimation_pipeline.predict(X_test)
    if show_plot:
        plot_benchmark(X_test, y_test, predictions)
    else:
        print_benchmark(y_test, predictions)
    
    # Now the estimations pipeline is applied over the
    # whole training set for later submission
    print('Fitting the pipeline to all the data...')
    X_all = train_df.drop(['SalePrice','Id'], axis=1)
    y_all = train_df['SalePrice']
    _, y_all = prepare_inputs(X_all, y_all)
    estimation_pipeline.fit(X_train_trans, y_all)
    print('Score: %.8f' % estimation_pipeline.score(X_train_trans, y_all))
    
    return (transformation_pipeline, estimation_pipeline, X_test_trans)

def impute_special_cases(X_val):
    ''' We handle some special cases of categorical nans '''
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

def create_dummies(X_val):
    ''' Function which creates dummy variables for all the categoricals
    and hereby take possible resulting multi-collinearity into account'''
    print('Creating dummies...')
    print('Starting with input of shape: %s' % str(X_val.shape))
    X_new = X_val.copy()
    x_extended = pd.get_dummies(X_val, dummy_na=True, drop_first=True)
    print('Returning output of shape: %s' % str(x_extended.shape))
    return x_extended

def create_sellingage(df):
    ''' Create a combined variable for the age of the house when sold '''
    new_df = df.copy()
    new_df['SellingAge'] = new_df['YrSold'] - new_df['YearRemodAdd']
    return new_df

def combined_livingspace(df):
    ''' Create combined variable for all the living space in the buildings '''
    new_df = df.copy()
    new_df['TotalSF'] = new_df['TotalBsmtSF'] + new_df['GrLivArea']
    return new_df

def fill_numerical_nans(X_val):
    ''' Fills nans in numerical features of the data set '''
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
    ''' Prints message if there are still nans in the data set '''
    null_columns = X_val.columns[X_val.isnull().any(axis=0)]
    if len(null_columns) > 0:
        print('There are still NaNs in the data!')
        print(null_columns)
    return X_val

def impute_categorical(X_val):
    ''' We impute special cases of categorical nans so that there
    are not both nans and None's '''
    X_new = X_val.copy()
    for col in ['MasVnrType', 'GarageType', 'MiscFeature']:
        X_new[col] = X_new[col].fillna('None')
    return X_new

def execute_pipeline(transformation_pipeline, fitted_pipeline, transformed_x, show_plot=True):
    ''' Executed the fitted pipeline and returns predictions '''
    predictions = fitted_pipeline.predict(transformed_x)
    if show_plot:
        draw_sanity_check(predictions, False)
    return predictions

def encode_ordinals(X_val):
    ''' Method for label encoding the categorical variables
    which might have meaning in the ordering '''
    X_new = X_val.copy()
    for col in ['FireplaceQu', 'BsmtQual', 'BsmtCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'GarageQual', 'GarageCond', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond']:
        encoder = LabelEncoder()
        encoder.fit(list(X_val[col].values))
        X_new[col] = encoder.transform(list(X_val[col].values))
    return X_new

def rmse_log(y_actual, y_pred):
    ''' Returns the root mean squared log error of the vectors '''
    assert y_pred.min() <= 0, "Negative price found in prediction! ¯\_(ツ)_/¯"
    return math.sqrt(mean_squared_error(np.log(y_pred), np.log(y_actual)))

def rmse(y_actual, y_pred):
    ''' Function which returs the root mean 
    squared errorof the both vectors '''
    return math.sqrt(mean_squared_error(y_actual, y_pred))

# Removes outliers and normalized distribution of target variable
def prepare_inputs(X_val, y_val):
    outliers = X_val[X_val['GrLivArea'] >= 4000]
    return (X_val.drop(outliers.index), np.log1p(y_val.drop(outliers.index)))

def print_benchmark(y_actual, y_predicted, log_transform = False):
    ''' Prints root mean squared error of the vectors and
    optionally performs a log transformation '''
    print('R2-score: %s' % r2_score(y_actual, y_predicted))
    if log_transform:
        print('RMSE (log): %s' % rmse(np.log1p(y_actual), np.log1p(y_predicted)))
    else:
        print('RMSE (log): %s' % rmse(y_actual, y_predicted))
    
def plot_benchmark(X_t, y_t, pred):
    ''' Plots a benchmark for comparision '''
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
    
def normalize_skewed_features(df):
    ''' Does a box-cox transformation of features with
    an absolute skew of greather than 0.8 '''
    new_df = df.copy()
    numerical_features = new_df.select_dtypes(exclude=['object'])
    features_skewed = numerical_features.apply(lambda x: skew(x, nan_policy='omit')).sort_values(ascending=False)
    skewness = pd.DataFrame({'skew' :features_skewed})
    highly_skewed_features = skewness.loc[abs(skewness['skew']) > .8, :]
    for feature in highly_skewed_features.index:
        new_df[feature] = boxcox1p(new_df[feature], 0.15)
    return new_df

def write_submission(pred, in_dollars=False):
    ''' Writes a file for submission to kaggle and 
    does some rudimentary asserts to prevent wrong submissions '''
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
    ''' Perform a sanity check which performs the scatter plot
    of the predictions with the scatter plot in the initial training set '''
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
    ax.set_xlabel('Living space (square feet)')
    ax.set_title('Baseline vs Prediction')
    ax.scatter(test_df['GrLivArea'], pred, alpha=.4)
    print('Mean of Salesprice in Training-Data: %.2f' % train_df['SalePrice'].mean())
    print('Mean of Salesprice in predictions: %.2f' % pred.mean())
    difference = pred.mean() - train_df['SalePrice'].mean()
    print('Difference in means is: %s' % difference)
    if np.abs(difference) > 5000:
        print('IMPORTANT: There is something wrong with your predictions!!!')
    plt.show()
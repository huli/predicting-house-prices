from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import math
import pandas as pd
import os
import matplotlib.pyplot as plt

def rmse_log(y_actual, y_pred):
    assert y_pred.min() <= 0, "Negative price found in prediction! ¯\_(ツ)_/¯"
    return math.sqrt(mean_squared_error(np.log(y_pred), np.log(y_actual)))

def rmse(y_actual, y_pred):
    return math.sqrt(mean_squared_error(y_actual, y_pred))

def print_benchmark(y_actual, y_predicted, log_transform = False):
    print('R2-score: %s' % r2_score(y_actual, y_predicted))
    if log_transform:
        print('RMSE (log): %s' % rmse(np.log1p(y_actual), np.log1p(y_predicted)))
    else:
        print('RMSE (log): %s' % rmse(y_actual, y_predicted))
    
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
    if in_dollars == False:
        pred = np.expm1(pred)
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    fig, ax = plt.subplots(1, figsize=(8, 6), sharey=True)
    ax.scatter(train_df['GrLivArea'], train_df['SalePrice'], alpha=.4)
    ax.set_title('Baseline vs Prediction')
    ax.scatter(test_df['GrLivArea'], pred, alpha=.4)
    print('Mean of Salesprice in Training-Data: %.2f' % train_df['SalePrice'].mean())
    print('Mean of Salesprice in predictions: %.2f' % pred.mean())
    difference = pred.mean() - train_df['SalePrice'].mean()
    print('Difference in means is: %s' % difference)
    if np.abs(difference) > 5000:
        print('IMPORTANT: There is something wrong with your predictions!!!')
    plt.show()
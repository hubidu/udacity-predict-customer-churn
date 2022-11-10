'''
Created: 4.11.2022
Author: Stefan Huber
Description:
Provides functions to analyze, feature engineer and train
the churn machine learning model.
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, plot_roc_curve
sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(data_file):
    '''
    returns dataframe for the csv found at data_file

    input:
            data_file: a path to the csv
    output:
            dataframe: pandas dataframe
    '''
    dataframe = pd.read_csv(data_file)
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return dataframe


def perform_eda(dataframe):
    '''
    perform eda on dataframe and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    '''
    for column_name in ['Churn', 'Customer_Age']:
        plt.figure(figsize=(20, 10))
        dataframe.hist(column_name)
        plt.savefig(f"./images/eda/{column_name}_hist.png")

    plt.figure(figsize=(20, 10))
    dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig("./images/eda/Marital_Status_hist.png")

    plt.figure(figsize=(20, 10))
    sns.histplot(dataframe['Total_Trans_Ct'],
                 stat='density', kde=True).get_figure()
    plt.savefig("./images/eda/Total_TransCt_density.png")

    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig("./images/eda/Dark2_r_heat.png")


def encoder_helper(dataframe, category_lst=None, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            dataframe: pandas dataframe with new columns for
    '''
    if category_lst is None:
        category_lst = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

    for categorical in category_lst:
        category_vals = []
        category_groups = dataframe.groupby(categorical).mean()[response]

        for val in dataframe[categorical]:
            category_vals.append(category_groups.loc[val])

        dataframe[f"{categorical}_{response}"] = category_vals

    return dataframe


def perform_feature_engineering(dataframe, response='Churn'):
    '''
    input:
              dataframe: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y_values = dataframe[response]
    x_values = pd.DataFrame()
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']

    x_values[keep_cols] = dataframe[keep_cols]
    x_train, x_test, y_train, y_test = train_test_split(
        x_values, y_values, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/classification_report_rf.png')
    plt.close()

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/classification_report_lr.png')
    plt.close()


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)

def roc_plot(lrc, cv_rfc, x_test, y_test, output_pth):
    lrc_plot = plot_roc_curve(lrc, x_test, y_test)

    plt.figure(figsize=(15, 8))

    ax = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_, x_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)

    plt.savefig(output_pth)

def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # save best model
    joblib.dump(lrc, './models/logistic_model.pkl')

    classification_report_image(y_train, y_test, y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    feature_importance_plot(
        cv_rfc, x_train.append(x_test), './images/results/feature_importance_plot.png')

    roc_plot(lrc, cv_rfc, x_test, y_test, './images/results/roc.png')

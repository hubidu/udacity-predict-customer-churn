"""
Created: 4.11.2022
Author: Stefan Huber
Description: Tests for churn_library
"""
import logging
from os import listdir
from os.path import isfile, join
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')



def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        bank_data = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert bank_data.shape[0] > 0
        assert bank_data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    image_dir = './images/eda'

    bank_data = cls.import_data('./data/bank_data.csv')
    cls.perform_eda(bank_data)

    image_files = [f for f in sorted(
        listdir(image_dir)) if isfile(join(image_dir, f))]

    try:
        assert ['Churn_hist.png', 'Customer_Age_hist.png', 'Dark2_r_heat.png',
                'Marital_Status_hist.png', 'Total_TransCt_density.png'] == image_files
    except AssertionError as err:
        logging.error('perform_eda did not produce expected image files')
        raise err

    logging.info('Testing perform_eda: SUCCESS')


def test_encoder_helper():
    '''
    test encoder helper
    '''
    categorical_cols = ['Gender', 'Education_Level',
                        'Marital_Status', 'Income_Category', 'Card_Category']
    bank_data = cls.import_data('./data/bank_data.csv')
    bank_data = cls.encoder_helper(bank_data)

    try:
        for column in categorical_cols:
            bank_data[f"{column}_Churn"]
    except KeyError as err:
        logging.error('Expected categorical column not in dataframe')
        raise err

    logging.info('Testing encoder_helper: SUCCESS')

def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    bank_data = cls.import_data('./data/bank_data.csv')
    bank_data = cls.encoder_helper(bank_data)
    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(bank_data)

    try:
        assert x_train.shape[0] > 0
        assert x_train.shape[0] == y_train.shape[0]
        assert x_train.shape[0] < bank_data.shape[0]

        assert x_test.shape[0] > 0
        assert x_test.shape[0] == y_test.shape[0]
    except AssertionError as err:
        logging.error("Feature engineering did not produce correct output")
        raise err

    logging.info('Testing perform_feature_engineering: SUCCESS')


def test_train_models():
    '''
    test train_models
    '''
    bank_data = cls.import_data('./data/bank_data.csv')
    bank_data = cls.encoder_helper(bank_data)
    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(bank_data)
    cls.train_models(x_train, x_test, y_train, y_test)

    model_dir = './models'
    model_files = [f for f in sorted(
        listdir(model_dir)) if isfile(join(model_dir, f))]

    results_dir = './images/results'
    results_images = [f for f in sorted(
        listdir(results_dir)) if isfile(join(results_dir, f))]

    try:
        assert ['logistic_model.pkl', 'rfc_model.pkl'] == model_files
        assert ['classification_report_lr.png', 'classification_report_rf.png', 'feature_importance_plot.png'] == results_images
    except AssertionError as err:
        logging.error('train_models did not produce expected output files')
        raise err

    logging.info('Testing train_models: SUCCESS')
    logging.info('Please check ./models for generated models and ./images/results for model quality metrics')

if __name__ == '__main__':
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()

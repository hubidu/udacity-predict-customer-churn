# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

Python library which provides functions to analyze, feature engineer and train
models to predict bank customer churn.

## Files and data description

Overview of the files and data present in the root directory.

    - data			# folder for input data
      - bank_data.csv
    - images
      - eda 		# images from eda phase
      - results		# report images from training process
    - logs 			# log file output from running the training
    - models		# output directory for trained models
    - churn_library.py	# python library
    - churn_script_logging_and_tests.py # executable to run tests and train models
    - churn_notebook.ipynb	# jupyter notebook to fiddle with data

## Install dependencies

Install module dependencies

```bash
   python3 -m pip install -r requirements_py3.8.txt
```

You should also install autopep8

```bash
   python3 -m pip install autopep8
```

and pylint

```bash
   python3 -m pip install pylint
```

## Running Files

Below command

```bash
   ipython churn_script_logging_and_tests.py
```

will run tests on the library and if successful perform the training process using the functions
provided in _churn_library.py_.
If there are any errors the command will fail. Details of the error can be found
in the ./logs directory.
If the command runs successfully it will produce trained models in the ./models directory
and quality metrics of the models in ./images/results

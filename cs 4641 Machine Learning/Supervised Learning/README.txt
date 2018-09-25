# Required Directory setup
> mlemons7
    analysis_data.py
    proj1lemons.py
    README.txt
    mlemons7-analysis.pdf
    > magic04
        magic04.csv
        magic04-<ALGO>.csv
    > winequality-white
        winequality-white.csv
        winequality-white-<ALGO>.csv

# Python Setup:
The python code is in python 3.

Need the following libraries which can be pip installed:
    numpy
    scipy
    matplotlib
    pandas
    scikit-learn
    joblib

# Dataset Information:
>Wine Quality Dataset
    url: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
    Used the white wine subdirectory. It downloads seperated by ";" and need to convert to being seperated by ","

>Magic Gamma Telescope Dataset
    url: https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope

# Running code:
proj1lemons.py will run all the algorithms across all hyperparameters and save the results as a csv for each algorithm.
To edit the parameters, all checked ones will be permutations of the dictionaries returned by the setup_parameter method.
This is very important as the full spread can take a full day to run on a Dell xps 15
To change datasets, add/remove from the datasets list in the main script.

analysis_data.py will create the graphs seen in the paper. It iterates over both datasets.

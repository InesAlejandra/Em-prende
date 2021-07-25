#This file should be at /Src/utils/
# Library Import
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV, LogisticRegression, LogisticRegressionCV
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import dill as pickle
from .DataPreparation import *

# Ignoring all warnings
import warnings
warnings.filterwarnings('ignore')
import os

# read dataset
train_df = pd.read_csv('./Data/Set_22Jul_9AM_sin_missingvalues_todonormalizado11am_training.csv')# global random state
rand_state_ = 42
# Library Import
import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.preprocessing import LabelEncoder
import re

class DataPreparation():

    def __init__(self):
        pass
        
        
    def preprocess(self, df):
        df = self.fill_missing_values(df)
        df = self.feature_extraction(df)
        df = self.handle_categorical_variables(df)
        df = self.dimensionality_reduction(df)
        return df
        
    def fill_missing_values(self, df):
        return df
        
    def feature_extraction(self, df):
        return df
          
    def handle_categorical_variables(self, df):
        df['p'] = df.p.apply(lambda x: 1 if x == 'alta' else 0)
        return df
        
    def dimensionality_reduction(self, df):
        return df.drop(labels=['RUC'], axis=1)
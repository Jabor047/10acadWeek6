import logging

import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE


# defining the logging formt
form = logging.Formatter("%(asctime)s : %(levelname)-5.5s : %(message)s")
logger = logging.getLogger()

# setting the logging format
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(form)
logger.addHandler(consoleHandler)

# set the logger level
logger.setLevel(logging.DEBUG)

class preprocess():
    def __init__(self):
        pass

    # This function turns object columns to categorical columns
    def toCat(self, df):
        logger.info('\n turning object cols into category cols \n')
        colList = list(df.select_dtypes(include='object'))
        for col in colList:
            df[col] = df[col].astype('category')

        return df

    # transforms categorical columns to a one hot vector/ array
    def oneHot(self, df):
        logger.info('\n creating oneHot columns \n')
        colList = list(df.select_dtypes(include='category'))
        prefList = []

        # create a prefix on how to name the new dummy columns
        for col in colList:
            prefList.append("is_" + col)

        df = pd.get_dummies(df, columns=colList, prefix=prefList)
        return df

    # uses the StandardScaler to normalize our data
    def scalerDf(self, df, rmColList):
        logger.info('\n running StandardScaler() on the data \n')
        # select numerical columns not present in the remColList
        colList = list(df.select_dtypes(include=['int64', 'float64']))
        colList = [i for i in colList if i not in rmColList]

        scaler = StandardScaler()
        scaled = scaler.fit(df[colList].values)
        df[colList] = scaled.transform(df[colList].values)
        return df

    # Transforms the Y column into numbers
    def targetEnc(self, y):
        logger.info('\n running LabelEncoder() on the target col \n')
        labelEnc = LabelEncoder()
        y = labelEnc.fit_transform(y)

        return y

    # reduces the dimensions using the t-SNE technique
    def tsneFunc(self, df):
        logger.info('\n dimensional reductionality using TSNE \n')
        tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
        tsneResults = tsne.fit_transform(df)

        return tsneResults

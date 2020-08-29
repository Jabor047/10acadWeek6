import argparse

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, roc_curve

from Models import modelPlusFold
from Data import preprocess

parser = argparse.ArgumentParser(description="Process a csv file with ';' as a seperator")
parser.add_argument('filename')
args = parser.parse_args()

modelFold = modelPlusFold()
prePro = preprocess()

# load the csv file
df = pd.read_csv(args.filename, delimiter=';')

y = df['y']
df = df.drop(['y', 'duration'], axis=1)

# do preprocessing on the models
df = prePro.toCat(df)
df = prePro.oneHot(df)

exList = ['campaign', 'previous']
df = prePro.scalerDf(df, exList)

# dimesional reductionality
tsneResults = prePro.tsneFunc(df)

# Change the y column to numbers
y = prePro.targetEnc(y)

# split the data into training and validation sets
xTrain, xTest, yTrain, yTest = train_test_split(tsneResults, y, test_size=0.1, random_state=42)

# model training
trainedModel, rocAucScore, f1Score = modelFold.modelRun(tsneResults, y)

print('The results on the test set are as flows on: \n The roc_auc_score is {} \n The F1_score is {} '
      .format(rocAucScore, f1Score))

# predicting on the validation set
modelFold.validRun(xTest, yTest)

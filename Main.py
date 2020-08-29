import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
df = preprocess.toCat(df)
df = preprocess.oneHot(df)

exList = ['campaign', 'previous']
df = preprocess.scalerDf(df, exList)

# dimesional reductionality
tsneResults = preprocess.tsneFunc(df)

# split the data into training and validation sets
xTrain, xTest, yTrain, yTest = train_test_split(tsneResults, y, test_size=0.1, random_state=42)

# model training
trainedModel, rocAucScore, f1Score = modelFold.modelRun(tsneResults, y)

print('\n The roc_auc_score is {} \n The F1_score is {} '.format(rocAucScore, f1Score))

# predicting on the validation set
yPred = trainedModel.predict(xTest)

# plot and save the ROC_curve
mlpRoc = roc_auc_score(yTest, yPred)
fpr, tpr, thresholds = roc_curve(yTest, trainedModel.predict_proba(xTest)[:, 1])

plt.figure()
plt.plot(fpr, tpr, label='Multilayer perceptron classifier (area = %0.2f)' % mlpRoc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('MLP_ROC')
plt.show()

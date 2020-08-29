import logging
from statistics import mean

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix, roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

# defining the logging formt
form = logging.Formatter("%(asctime)s : %(levelname)-5.5s : %(message)s")
logger = logging.getLogger()

# setting the logging format
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(form)
logger.addHandler(consoleHandler)

# set the logger level
logger.setLevel(logging.DEBUG)

class modelPlusFold:
    def __init__(self):
        pass

    def modelKfold(self, model, x, y):
        logger.info("\n running KFold on {}".format(model))
        # A list to append all the score for each fold
        rocScores = []
        f1Scores = []

        # split the data further using KFold
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        for trainIndex, testIndex in cv.split(x):

            cvXTrain, cvXTest = x[trainIndex], x[testIndex]
            cvYTrain, cvYTest = y[trainIndex], y[testIndex]

            # Carry out Smote on the split data
            smote = SMOTE(sampling_strategy='minority')
            xSm, ySm = smote.fit_sample(cvXTrain, cvYTrain)

            # Train the model and predict on the test set
            modelTrained = model.fit(xSm, ySm)
            yPred = modelTrained.predict(cvXTest)

            # find the score (score for one fold) and append them to the scores list(all the scores)
            rocScores.append(round(roc_auc_score(cvYTest, yPred), 2))
            f1Scores.append(round(f1_score(cvYTest, yPred), 2))

        # Calculate the average roc and f1 scores of the folds
        avgRoc = mean(rocScores)
        avgF1 = mean(f1Scores)

        return modelTrained, avgRoc, avgF1

    def modelStratifiedFold(self, model, x, y):
        logger.info("\n running StratifiedKFold on {} \n".format(model))
        rocScores = []
        f1Scores = []

        # split the data further using StratifiedKFold
        cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        for trainIndex, testIndex in cv.split(x, y):

            cvXTrain, cvXTest = x[trainIndex], x[testIndex]
            cvYTrain, cvYTest = y[trainIndex], y[testIndex]

            # Carry out Smote on the split data
            smote = SMOTE(sampling_strategy='minority')
            xSm, ySm = smote.fit_sample(cvXTrain, cvYTrain)

            # Train the model and predict on the test set
            modelTrained = model.fit(xSm, ySm)
            yPred = modelTrained.predict(cvXTest)

            # find the score (score for one fold) and append them to the scores list(all the scores)
            rocScores.append(round(roc_auc_score(cvYTest, yPred), 2))
            f1Scores.append(round(f1_score(cvYTest, yPred), 2))

        # Calculate the average roc and f1 scores of the folds
        avgRoc = mean(rocScores)
        avgF1 = mean(f1Scores)

        return modelTrained, avgRoc, avgF1

    # runs our model for us
    def modelRun(self, x, y):
        logger.info('\n running the multilayered percpetron \n')
        model = MLPClassifier()

        trainedModel, roc, f1 = self.modelStratifiedFold(model, x, y)

        return trainedModel, roc, f1

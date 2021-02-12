from lifelines import CoxPHFitter
import pandas as pd
import numpy as np
import math
from sklearn.feature_selection import f_regression
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored


class FeatureImportance:
    def __init__(self, x_train, y_train, scores):
        """
        This class gets the MPSS product limit estimator and determines
        the feature importance through MPSS of scores
        using a Proportional Hazards Regression model.
        Two methods perform the proportional hazards regression model
        the difference being the package that is used, 'sksurv' or 'lifelines'.


        :param x_train: Feature data
        :param y_train: Label data
        :param scores: Scores from a classification model
        """
        self.x_train = x_train
        self.y_train = y_train
        self.scores = scores

    def select(self, method):
        """
        Determines the package used to perform the proportional hazards regression.
        :param method: MPSS-sksurv or MPSS-lifelines
        :return: feature importance
        """
        self.x_train, self.y_train, self.scores = self.truncate()
        if method == 'MPSS-sksurv':
            return self.mpss_ph_sksurv()
        elif method == 'MPSS-lifelines':
            return self.mpss_ph_lifelines()
    
    def truncate(self):
        """
        Truncates the observations where the label is in the negative class.
        :return: truncated x_train, y_train, scores
        """
        keep = np.where(self.y_train == 1)
        return self.x_train[keep], self.y_train[keep], self.scores[keep]

    def product_limit_estimator(self):
        """
        Gets the product limit estimator over the score.
        :return: product limit estimator score index, cumulative probability of positive label
        """
        x_train, y_train, scores = self.truncate()

        y_train = y_train.astype(bool)

        # Calculate the product limit estimator
        score, positive_prob = kaplan_meier_estimator(y_train, scores)
        return score, positive_prob

    def mpss_ph_sksurv(self):
        """

        Performs proportional hazards regression using sksurv package.

        :return: Feature importance
        """
        # Reformat for sksurv package
        x_train = pd.DataFrame(self.x_train)
        y_structured = [(ll, s) for ll, s in zip(self.y_train.astype(bool), self.scores)]
        y_structured = np.array(y_structured, dtype=[('class', 'bool_'), ('score', 'single')])

        # Remove any feature columns that are all 0 values, otherwise cannot run regression
        x_train_nonzero = x_train.loc[:, (x_train != 0).any(axis=0)]

        # Run proportional hazards regression
        estimator = CoxPHSurvivalAnalysis(alpha=0.1, verbose=1)
        estimator.fit(x_train_nonzero, y_structured)
        prediction = estimator.predict(x_train_nonzero)

        # Estimate p-values for each feature
        f, pvals = f_regression(x_train_nonzero, [x[1] for x in y_structured])
        approximate_se = pd.DataFrame(pd.Series(pvals,
                                                index=x_train_nonzero.columns).sort_values(ascending=False),
                                      columns=['p']).reset_index()

        # Calculate concordance indicating the goodness of fit
        concordance = concordance_index_censored(self.y_train.astype(bool), self.scores, prediction)
        print('concordance', concordance[0])

        # Dataframe with coefficients, absolute value of coefficients, and p-values
        importance = pd.DataFrame(estimator.coef_, columns=['coef'])
        importance['coef_abs'] = [math.fabs(c) for c in importance['coef']]
        importance['feature'] = importance.index.values
        importance = importance.merge(approximate_se, left_on='feature', right_on='index').drop('index', axis=1)

        # Sort feature importance
        importance = importance.sort_values('coef_abs', ascending=False).reset_index(drop=True)
        return importance

    def mpss_ph_lifelines(self):
        """

        Performs proportional hazards regression using lifelines package.

        :return: feature importance
        """
        x_train = pd.DataFrame(self.x_train)

        # Remove any feature columns that are all 0 values, otherwise cannot run regression
        lifelines_dataset = x_train.loc[:, (x_train != 0).any(axis=0)]

        # Reformat for lifelines package
        lifelines_dataset['scores'] = self.scores
        lifelines_dataset['event'] = 1

        # Run proportional hazards regression
        cph = CoxPHFitter(penalizer=5, alpha=1)
        cph.fit(lifelines_dataset, duration_col='scores', event_col='event')

        # Dataframe with coefficients, absolute value of coefficients, and p-values
        importance = cph.summary.reset_index()[['covariate', 'coef', 'p']]
        importance['feature'] = importance['covariate']
        importance['coef_abs'] = importance['coef'].apply(lambda x: math.fabs(x))

        # Sort feature importance
        importance = importance.sort_values('coef_abs', ascending=False).reset_index(drop=True)
        return importance

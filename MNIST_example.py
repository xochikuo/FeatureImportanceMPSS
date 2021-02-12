from core.feature_importance import FeatureImportance
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D


def load_data():
    """
    This will load MNIST dataset from tensorflow,
    select three and eight classes for binary classification,
    reshape and normalize feature matrix,
    and relabel the classes.

    :return: x_train, y_train, x_test, y_test
    """
    (_x_train, _y_train), (_x_test, _y_test) = tf.keras.datasets.mnist.load_data()
    # Subset MNIST images to include three and eight only
    _x_train = _x_train[(_y_train == 3) | (_y_train == 8)]
    _y_train = _y_train[(_y_train == 3) | (_y_train == 8)]
    _x_test = _x_test[(_y_test == 3) | (_y_test == 8)]
    _y_test = _y_test[(_y_test == 3) | (_y_test == 8)]
    # Reshape and normalize the feature matrix
    _x_train = _x_train.reshape(_x_train.shape[0], 28, 28, 1)
    _x_test = _x_test.reshape(_x_test.shape[0], 28, 28, 1)
    _x_train = _x_train.astype('float32')
    _x_test = _x_test.astype('float32')
    _x_train /= 255
    _x_test /= 255
    # Relabel images for binary classification
    relabel = {8: 1, 3: 0}
    _y_train = np.array([relabel[xi] for xi in _y_train])
    _y_test = np.array([relabel[xi] for xi in _y_test])
    return _x_train, _y_train, _x_test, _y_test


def feature_selection(n, importance_df, _x_train, _x_test):
    """
    Returns feature matrices from the top n features.
    :param n: Number of features
    :param importance_df: Sorted feature importance matrix
    :param _x_train: Training feature matrix
    :param _x_test: Testing feature matrix
    :return: x_train, x_test with top n features
    """
    # Create mask with 1 for top n_features and 0 for all other features
    keep_features = importance_df.iloc[:n, :]
    features = [int(x) for x in keep_features['feature'].values.tolist()]
    numbered_pixel = np.array([1 if i in features else 0 for i in range(28 * 28 * 1)]).reshape((28, 28))

    # Multiply the feature matrices by a mask to zero keep only n features
    _x_train = np.multiply(_x_train.reshape(_x_train.shape[0], 28, 28), numbered_pixel)
    _x_test = np.multiply(_x_test.reshape(_x_test.shape[0], 28, 28), numbered_pixel)
    _x_train = _x_train.reshape(_x_train.shape[0], 28, 28, 1)
    _x_test = _x_test.reshape(_x_test.shape[0], 28, 28, 1)
    return _x_train, _x_test


class DeepClassificationModel:
    def __init__(self):
        """
        Deep neural network to classify MNIST dataset.
        """
        input_shape = (28, 28, 1)
        self.model = Sequential()
        self.model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
        self.model.add(Dense(128, activation=tf.nn.relu))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation=tf.nn.sigmoid))
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def fit(self, x, y, *args, **kwargs):
        """
        Trains the classification model
        :param x: features
        :param y: labels
        :return: fitted model
        """
        self.model.fit(x, y, *args, **kwargs)

    def evaluate(self, x, y):
        """
        Evaluates fit of model by with metrics.
        :param x: features
        :param y: labels
        :return: metrics
        """
        return self.model.evaluate(x, y)

    def predict(self, x):
        """
        Predicts positive class
        :param x: features
        :return: score
        """
        return self.model.predict(x)


def plot_positive_probability(_score_index, _positive_probability, _thresholds, _recall):
    """
    Plots the product limit estimator and the recall curve on the same plot.
    :param _score_index: product limit estimator score index
    :param _positive_probability: product limit estimator cumulative positive probability
    :param _thresholds: thresholds for recall
    :param _recall: recall at thresholds
    :return: plot
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    ax2 = ax.twinx()
    ax.step(_score_index, _positive_probability, where="post", linewidth=3,
            label="$\hat{I}(s)$, \nProduct Limit Estimator")
    ax2.step(_thresholds, _recall, where="post", color='r', linestyle='--', label='Recall')
    ax.set_ylabel("Est. Probability of Positive Label $\hat{I}(s)$")
    ax2.set_ylabel("Recall")
    ax.set_xlabel("score $s$")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc='lower left')
    return fig


def plot_feature_heatmap(importance_df):
    """
    Plots the heatmap of feature importance for MPSS feature importance.
    :param importance_df: dataframe of feature importance
    :return: plot
    """
    fig = plt.figure(figsize=(5, 3))
    coef = {}
    for i, r in importance_df.iterrows():
        if r.p < 0.1:
            coef[r['feature']] = r['coef']
    pixels = np.array([coef.get(i, np.nan) for i in range(28 * 28)])
    plt.imshow(pixels.reshape(28, 28), cmap='cool')
    plt.colorbar()
    return fig


def calculate_recall(_y_train, _scores):
    """
    Recall is the ratio tp / (tp + fn)
    :param _y_train: labels
    :param _scores: score from classification model
    :return:
    """
    sort_index = np.argsort(_scores)
    _y_train = _y_train[sort_index]
    _scores = _scores[sort_index]
    positive = np.where(_y_train == 1)

    # Select thresholds to calculate recall
    _thresholds = list(set(_scores[positive].tolist()))
    _thresholds.sort()

    # Calculate recall
    recall_curve = []
    for t in _thresholds:
        tp = sum([ll if s > t else 0 for ll, s in zip(_y_train, _scores)])
        fn = sum([ll if s <= t else 0 for ll, s in zip(_y_train, _scores)])
        confusion_matrix_recall = tp / (tp + fn)
        recall_curve.append(confusion_matrix_recall)
    return recall_curve, _thresholds


def rounder():
    """
    Correction for python rounding errors
    :return: Round number to 9 digits
    """
    return lambda x: round(x, 9)


if __name__ == "__main__":
    # Set random seed
    random.seed(1)

    # Load data and train classification model
    x_train, y_train, x_test, y_test = load_data()
    c_model = DeepClassificationModel()
    c_model.fit(x_train, y_train, epochs=10)
    print(c_model.evaluate(x_test, y_test))
    scores = np.array([round(_[0], 5) for _ in c_model.predict(x_train)])

    # Evaluate Classification Model with Recall
    recall, thresholds = calculate_recall(y_train, scores)

    # Calculate MPSS Product Limit Estimator
    fi = FeatureImportance(x_train.reshape(x_train.shape[0], 28 * 28 * 1), y_train, scores)
    ple_score, positive_probability = fi.product_limit_estimator()

    # Assert that recall and product limit estimator are equal
    assert list(map(rounder(), recall)) == list(map(rounder(), positive_probability.tolist()))

    # Plot Product Limit Estimator and Recall
    plot = plot_positive_probability(ple_score, positive_probability, thresholds, recall)
    plot.savefig("product_limit_estimator.pdf", bbox_inches='tight')

    # Get MPSS proportional hazard feature importance
    importance = fi.select(method='MPSS-sksurv')

    # Plot heatmap of feature importance
    plot2 = plot_feature_heatmap(importance)
    plot2.savefig("feature_importance_heatmap.pdf", bbox_inches='tight')

    # Select features and turn off input from any other features
    x_train, x_test = feature_selection(n=50, importance_df=importance, _x_train=x_train, _x_test=x_test)

    # Retrain model only on selected features
    c_model = DeepClassificationModel()
    c_model.fit(x_train, y_train, epochs=10)
    print(c_model.evaluate(x_test, y_test))

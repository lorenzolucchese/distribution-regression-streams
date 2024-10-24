import numpy as np

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm as tqdm

from utils import bags_to_2D
from sklearn_transformers import LeadLag, AddTime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


def model(X, y, ll=None, at=False, mode='krr', NUM_TRIALS=5, cv=3, grid={}):
    """Performs a DR-Matern32 kernel-based distribution regression on ensembles (of possibly unequal cardinality)
       of univariate or multivariate time-series. 

       Input:
              X (list): list of lists such that

                        - len(X) = n_samples

                        - for any i, X[i] is a list of arrays of shape (length, dim)

                        - for any j, X[i][j] is an array of shape (length, dim)

              y (np.array): array of shape (n_samples,)

              ll (list of ints): dimensions to lag
              at (bool): if True pre-process the input path with add-time

              mode (str): "krr" -> Kernel Ridge Regression, 'svr' -> Support Vector Regresion

              NUM_TRIALS, cv : parameters for nested cross-validation

       Output: mean MSE (and std) (both scalars) of regression performance on a cv-folds cross-validation (NUM_TRIALS times)
    """

    assert mode in ['svr', 'krr'], "mode must be either 'svr' or 'krr' "
    
    # possibly augment the state space of the time series
    if ll is not None:
        X = LeadLag(ll).fit_transform(X)
    if at:
        X = AddTime().fit_transform(X)


    # transforming the data into a 2d array (N_bags, N_items_max x length_min x dim)
    X, max_items, common_T, dim_path = bags_to_2D(X)

    if mode == 'krr':
        parameters = {'clf__kernel': ['precomputed'], 'clf__alpha': [1e-3,1e-2, 1e-1, 1, 1e1, 1e2,1e3],
                      'rbf_matern__gamma_emb': [1e-3,1e-2, 1e-1, 1, 1e1, 1e2,1e3],
                      'rbf_matern__gamma_top': [1e-3,1e-2, 1e-1, 1, 1e1, 1e2,1e3]}

        # check if the user has not given an irrelevant entry
        assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
            list(grid.keys())), "keys should be in " + ' '.join([str(e) for e in parameters.keys()])
        # merge the user grid with the default one
        parameters.update(grid)

        clf = KernelRidge

    else:
        parameters = {'clf__kernel': ['precomputed'], 'clf__C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                      'rbf_matern__gamma_emb': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1. / (common_T * dim_path)],
                      'rbf_matern__gamma_top': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1. / (common_T * dim_path)]}

        # check if the user has not given an irrelevant entry
        assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
            list(grid.keys())), "keys should be in " + ' '.join(
            [str(e) for e in parameters.keys()])

        # merge the user grid with the default one
        parameters.update(grid)

        clf = SVR

    # building the RBF-Matern estimator
    pipe = Pipeline([('rbf_matern', RBF_Matern_Kernel(max_items=max_items, size_item=dim_path * common_T)),
                     ('clf', clf())])

    # Loop for each trial
    scores = np.zeros(NUM_TRIALS)
    results = {}
    for i in tqdm(range(NUM_TRIALS)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        # parameter search
        model = GridSearchCV(pipe, parameters, verbose=0, n_jobs=-1, scoring='neg_mean_squared_error', cv=cv)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        scores[i] = mean_squared_error(y_pred, y_test)
        results[i]={'pred':y_pred,'true':y_test}
    return scores.mean(), scores.std(), results



class RBF_Matern_Kernel(BaseEstimator, TransformerMixin):
    def __init__(self, max_items=None, size_item=None, gamma_emb=1.0, gamma_top=1.0):
        super(RBF_Matern_Kernel, self).__init__()
        self.gamma_emb = gamma_emb
        self.gamma_top = gamma_top
        self.size_item = size_item
        self.max_items = max_items

    def transform(self, X):

        alpha = 1. / (2 * self.gamma_top ** 2)
        x = X.reshape(-1, self.size_item)
        K = rbf_matern_mmd_mat(x, self.x_train, gamma=self.gamma_emb, max_items=self.max_items)

        return np.exp(-alpha * K)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        x_train = X.reshape(-1, self.size_item)  # x_train is [bag1_item1,bag1_item2,....bagN_itemN] some items are nans
        self.x_train = x_train
        return self


def rbf_matern_mmd_mat(X, Y, gamma=None, max_items=None):
    M = max_items

    rho = 1. / gamma
    sqrt3 = np.sqrt(3.0)

    if X.shape[0] == Y.shape[0]:

        # Matern 3/2
        XX = np.dot(X, X.T)
        X_sqnorms = np.diagonal(XX)
        r2 = -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]
        r = np.sqrt(r2)
        K_XX = (1.0 + sqrt3 * rho * r) * np.exp(-sqrt3 * rho * r)

        K_XX_blocks = [K_XX[i * M:(i + 1) * M, i * M:(i + 1) * M] for i in range(K_XX.shape[0] // M)]
        K_XX_means = [np.nanmean(bag) for bag in K_XX_blocks]

        K_XY_means = np.nanmean(K_XX.reshape(X.shape[0] // M, M, Y.shape[0] // M, M), axis=(1, 3))
        mmd = np.array(K_XX_means)[:, np.newaxis] + np.array(K_XX_means)[np.newaxis, :] - 2 * K_XY_means

    else:

        # Matern 3/2
        XX = np.dot(X, X.T)
        XY = np.dot(X, Y.T)
        YY = np.dot(Y, Y.T)

        X_sqnorms = np.diagonal(XX)
        Y_sqnorms = np.diagonal(YY)

        r2_XX = -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]
        r_XX = np.sqrt(r2_XX)

        r2_YY = -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]
        r_YY = np.sqrt(r2_YY)

        r2_XY = -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]
        r_XY = np.sqrt(r2_XY)

        K_XY = (1.0 + sqrt3 * rho * r_XY) * np.exp(-sqrt3 * rho * r_XY)
        K_XX = (1.0 + sqrt3 * rho * r_XX) * np.exp(-sqrt3 * rho * r_XX)
        K_YY = (1.0 + sqrt3 * rho * r_YY) * np.exp(-sqrt3 * rho * r_YY)


        # blocks of bags
        K_XX_blocks = [K_XX[i * M:(i + 1) * M, i * M:(i + 1) * M] for i in range(K_XX.shape[0] // M)]
        K_YY_blocks = [K_YY[i * M:(i + 1) * M, i * M:(i + 1) * M] for i in range(K_YY.shape[0] // M)]

        # nanmeans
        K_XX_means = [np.nanmean(bag) for bag in K_XX_blocks]  # n_bags_test
        K_YY_means = [np.nanmean(bag) for bag in K_YY_blocks]  # n_bags_train

        K_XY_means = np.nanmean(K_XY.reshape(X.shape[0] // M, M, Y.shape[0] // M, M), axis=(1, 3))

        mmd = np.array(K_XX_means)[:, np.newaxis] + np.array(K_YY_means)[np.newaxis, :] - 2 * K_XY_means

    return mmd

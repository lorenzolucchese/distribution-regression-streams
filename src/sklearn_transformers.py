import numpy as np
import copy
import random
import doctest
import iisignature
import warnings
from typing import Union

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import imp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import as_float_array


class AddTime(BaseEstimator, TransformerMixin):
    # sklearn-type estimator to add time as an extra dimension of a D-dimensional path.
    # Note that the input must be a list of arrays (i.e. a list of D-dimensional paths)

    def __init__(self, init_time=0., total_time=1.):
        self.init_time = init_time
        self.total_time = total_time

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        t = np.linspace(self.init_time, self.init_time + 1, len(X))
        return np.c_[t, X]

    def transform(self, X, y=None):
        return [[self.transform_instance(x) for x in bag] for bag in X]


class LeadLag(BaseEstimator, TransformerMixin):
    # sklearn-type estimator to compute the Lead-Lag transform of a D-dimensional path.
    # Note that the input must be a list of arrays (i.e. a list of D-dimensional paths)

    def __init__(self, dimensions_to_lag):
        if not isinstance(dimensions_to_lag, list):
            raise NameError('dimensions_to_lag must be a list')
        self.dimensions_to_lag = dimensions_to_lag

    def fit(self, X, y=None):
        return self

    def transform_instance_1D(self, x):

        lag = []
        lead = []

        for val_lag, val_lead in zip(x[:-1], x[1:]):
            lag.append(val_lag)
            lead.append(val_lag)
            lag.append(val_lag)
            lead.append(val_lead)

        lag.append(x[-1])
        lead.append(x[-1])

        return lead, lag

    def transform_instance_multiD(self, X):
        if not all(i < X.shape[1] and isinstance(i, int) for i in self.dimensions_to_lag):
            error_message = 'the input list "dimensions_to_lag" must contain integers which must be' \
                            ' < than the number of dimensions of the original feature space'
            raise NameError(error_message)

        lead_components = []
        lag_components = []

        for dim in range(X.shape[1]):
            lead, lag = self.transform_instance_1D(X[:, dim])
            lead_components.append(lead)
            if dim in self.dimensions_to_lag:
                lag_components.append(lag)

        return np.c_[lead_components + lag_components].T

    def transform(self, X, y=None):
        return [[self.transform_instance_multiD(x) for x in bag] for bag in X]


class ExpectedSignatureTransform(BaseEstimator, TransformerMixin):

    def __init__(self, order, martingale_indices=None):
        if not isinstance(order, int) or order < 1:
            raise NameError('The order must be a positive integer.')
        if martingale_indices is not None and (not isinstance(martingale_indices, list) or not all(isinstance(i, int) for i in martingale_indices)):
            raise NameError('The martingale_indices must be a list of integers.')
        self.order = order
        self.martingale_indices = martingale_indices

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # get the lengths of all time series (across items across bags)
        lengths = [item.shape[0] for bag in X for item in bag]
        if len(list(set(lengths))) == 1:
            # if all time series have the same length, the signatures can be computed in batch
            X = [compute_signature(bag, self.order, self.martingale_indices) for bag in X]
        else:
            X = [np.array([compute_signature(item, self.order, self.martingale_indices) for item in bag]) for bag in X]
        return [x.mean(0) for x in X]


class pathwiseExpectedSignatureTransform(BaseEstimator, TransformerMixin):

    def __init__(self, order, martingale_indices=None):
        if not isinstance(order, int) or order < 1:
            raise NameError('The order must be a positive integer.')
        if martingale_indices is not None and (not isinstance(martingale_indices, list) or not all(isinstance(i, int) for i in martingale_indices)):
            raise NameError('The martingale_indices must be a list of integers.')
        self.order = order
        self.martingale_indices = martingale_indices

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        pwES = []
        for bag in X:
            # get the lengths of all time series in the bag
            lengths = [item.shape[0] for item in bag]
            if len(list(set(lengths))) == 1:
                # if all time series have the same length, the (pathwise) signatures can be computed in batch
                pwES.append(compute_signature(bag, self.order, self.martingale_indices, stream=True))
            else:
                error_message = 'All time series in a bag must have the same length'
                raise NameError(error_message)

        return [x.mean(0) for x in pwES]


class SignatureTransform(BaseEstimator, TransformerMixin):

    def __init__(self, order):
        if not isinstance(order, int) or order < 1:
            raise NameError('The order must be a positive integer.')
        self.order = order

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # get the lengths of all pathwise expected signatures
        lengths = [pwES.shape[0] for pwES in X]
        if len(list(set(lengths))) == 1:
            # if all pathwise expected signatures have the same length, the signatures can be computed in batch
            return compute_signature(X, self.order)
        else:
            return [compute_signature(item, self.order) for item in X]


def get_signature_indices(depth: int, channels: int, ending_indices: list[int]):
    if not isinstance(ending_indices, list) or not all(isinstance(i, int) for i in ending_indices) or not all(0 <= i < channels for i in ending_indices):
        raise ValueError(f'The ending_indices argument must be a list of integers between 0 and channels={channels}, got ending_indices={ending_indices}.')
    sig_indices = []
    start_index = 0
    for i in range(1, depth + 1):
        # in each signature level, of size channels**i, we want to extract j*channels**(i-1):(j+1)*channels**(i-1) for j in ending_indices
        for j in ending_indices:
            sig_indices.extend(list(range(start_index+j*channels**(i-1), start_index+(j+1)*channels**(i-1))))
        start_index += channels**i
    return sig_indices


def compute_signature(X: Union[np.ndarray, list[np.ndarray]], order: int, martingale_indices: list[int] = None, stream: bool = False):
    if isinstance(X, list):
        X = np.stack(X, axis=0)
    batch, length, channels = X.shape
    signatures_stream = iisignature.sig(X, order, 2)                                                                        # shape: (batch, length - 1, channels + ... + channels**depth)
    signatures = signatures_stream if stream else signatures_stream[:, -1, :]                                               # shape: (batch, channels + ... + channels**depth) or (batch, length - 1, channels + ... + channels**depth)
    if martingale_indices:
        signatures_lower = signatures_stream[:, :-1, :-channels**order]                                                     # shape: (batch, length - 2, channels + ... + channels**(depth-1))
        # pre-pend signature starting values at zero
        signatures_start = np.concatenate([np.zeros((batch, 1, channels**i)) for i in range(1, order)], axis=2)             # shape: (batch, 1, channels + ... + channels**(depth-1))	
        signatures_lower = np.concatenate([signatures_start, signatures_lower], axis=1)                                     # shape: (batch, length - 1, channels + ... + channels**(depth-1))        
        # append 0-th order signature
        signatures_lower = np.concatenate([np.ones((batch, length - 1, 1)), signatures_lower], axis=2)                      # shape: (batch, length - 1, 1 + channels + ... + channels**(depth-1))
        if stream:
            corrections = np.einsum('ijk,ijl->ijkl', signatures_lower, np.diff(X, axis=1))                                  # shape: (batch, length - 1, 1 + channels + ... + channels**(depth-1), channels)
            corrections = np.cumsum(corrections, axis=1).reshape((batch, length - 1, -1))                                   # shape: (batch, length - 1, channels + ... + channels**depth)
            num = np.einsum('ijk,ijk->ijk', signatures, corrections).mean(axis=0)                                           # shape: (length - 1, channels + ... + channels**depth)
        else:
            corrections = np.einsum('ijk,ijl->ikl', signatures_lower, np.diff(X, axis=1))                                   # shape: (batch, 1 + channels + ... + channels**(depth-1), channels)
            corrections = corrections.reshape((batch, -1))                                                                  # shape: (batch, channels + ... + channels**depth)
            num = np.einsum('ij,ij->ij', signatures, corrections).mean(axis=0)                                              # shape: (channels + ... + channels**depth)
        #TODO: implement other way of estimating c_hat
        #NOTE: replacing zeros in denom with ones both c_hat and corrections are zero at those indices
        denom = (corrections**2).mean(axis=0)
        denom[denom == 0] = 1
        c_hat = num / denom                                                                                                 # shape: (channels + ... + channels**depth) or (length - 1, channels + ... + channels**depth)
        #TODO: pre-compute signature indices for efficiency
        sig_indices = get_signature_indices(order, channels, martingale_indices)
        signatures[..., sig_indices] -= (c_hat * corrections)[..., sig_indices]                                             
    else:
        pass
    return signatures                                                                                                       # shape: (batch, channels + ... + channels**depth) or (batch, length - 1, channels + ... + channels**depth)


if __name__ == "__main__":
    doctest.testmod()

import sys
import os
import time
import datetime
import numpy as np
import pickle
from tqdm import tqdm as tqdm

import utils_roughvol

import DR_RBF
import DR_GA
import DR_Matern
import KES
import SES

def run_experiment(N = 20, M = 50, length = 200, min_T = 1e-6, max_T = 1e0, ll = None):
    np.random.seed(0)
    y = np.array((max_T-min_T)*np.random.rand(M)+min_T)
    
    V = []
    P = []
    
    for a in tqdm(y):
        V_intermediate = []
        P_intermediate = []
        for _ in range(N):
            vol_path = np.exp(utils_roughvol.fOU_generator(a, length=length))
            price_path = utils_roughvol.price_generator(vol_path)

            V_intermediate.append(vol_path.reshape(-1,1))
            P_intermediate.append(price_path.reshape(-1,1))
        V.append(V_intermediate)
        P.append(P_intermediate)

    for X, features in zip([P, V], ['price', 'vol']):
        _, _, results_RBF = DR_RBF.model(X, y, ll=ll, at=True, mode='krr', NUM_TRIALS=5, cv=3) 
        _, _, results_Matern = DR_Matern.model(X, y, at=True, mode='krr', NUM_TRIALS=5, cv=3) 
        _, _, results_GA = DR_GA.model(X, y, ll=ll, at=True, mode='krr', NUM_TRIALS=5, cv=3, lambdas=[0.01])
        _, _, results_KES = KES.model(X, y, ll=ll, at=True, mode='krr', NUM_TRIALS=5, cv=3, scales=[0.35])
        _, _, results_SES = SES.model(X, y, ll=ll, at=True, NUM_TRIALS=5, cv=3, depths1=[2,3,4], depth2=2)
        _, _, results_SES_mvr = SES.model(X, y, ll=ll, at=True, NUM_TRIALS=5, cv=3, depths1=[2,3,4], depth2=2, martingale_indices=[0])

        results = {'rbf': results_RBF, 'Matern': results_Matern, 'GA': results_GA, 'KER': results_KES, 'SES': results_SES, 'SES_mvr': results_SES_mvr}
        os.makedirs('experiments/finance', exist_ok=True)
        pickle.dump(results, open(f'experiments/finance/{features}_N={N}_ll={ll}.pkl', 'wb'))


if __name__ == "__main__":
    # global config
    Ns = [20, 50, 100]
    lls = [None, [0]]

    # select specific job using PBS_ARRAY_INDEX
    PBS_ARRAY_INDEX = int(sys.argv[1])
    N = Ns[PBS_ARRAY_INDEX % 3]
    ll = lls[PBS_ARRAY_INDEX // 3] 

    print(f'Starting experiment N={N}, ll={ll}...')
    start_time = time.time()
    run_experiment(N=N, ll=ll)
    print(f'...ending experiment N={N}, ll={ll}. Tot time: {datetime.timedelta(seconds=(time.time() - start_time))}.')

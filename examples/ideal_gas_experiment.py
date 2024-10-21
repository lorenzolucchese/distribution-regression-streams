import sys
import os
import time
import datetime
import numpy as np
import pickle
from tqdm import tqdm as tqdm

import utils_particles

import DR_RBF
import DR_GA
import DR_Matern
import KES
import SES

def run_experiment(r = 3.5, M = 50, n_particles = 20, V = 3, max_time = 2, dt = 0.01, min_T = 1, max_T = 1000, ll = None):
    np.random.seed(0)
    temperatures = np.array((max_T-min_T)*np.random.rand(M)+min_T)

    # parameters which do not change from one gas to another
    R = (r/10)*(V/n_particles)**(1./3)
    
    gases = []
    for temp in tqdm(temperatures):
        
        Gas = utils_particles.Sim(n_particles, R, temp, V, max_time, dt)
        
        # To store the trajectories
        trajectories = [[] for _ in range(n_particles)]

        for t in np.linspace(0, Gas.max_time, Gas.Nt):
            positions = Gas._draw_frame(t)
            for i in range(len(trajectories)):
                # update particle i
                trajectories[i].append(list(positions[i]))
        trajectories = [np.array(particle) for particle in trajectories]
        
        gases.append(trajectories)

    _, _, results_RBF = DR_RBF.model(gases, temperatures/max_T, ll=ll, at=True, mode='krr', NUM_TRIALS=5, cv=3)
    _, _, results_Matern = DR_Matern.model(gases, temperatures/max_T, ll=ll, at=True, mode='krr', NUM_TRIALS=5, cv=3) 
    _, _, results_GA = results_GA = DR_GA.model(gases, temperatures/max_T, ll=ll, at=True, mode='krr', NUM_TRIALS=5, cv=3, lambdas=[0.01])
    _, _, results_KES = KES.model(gases, temperatures/max_T, ll=ll, at=True, mode='krr', NUM_TRIALS=5, cv=3, scales=[0.4])
    _, _, results_SES = SES.model(gases, temperatures/max_T, ll=ll, at=True, NUM_TRIALS=5, cv=3, depths1=[2,3], depth2=2) 
    _, _, results_SES_mvr = SES.model(gases, temperatures/max_T, ll=ll, at=True, NUM_TRIALS=5, cv=3, depths1=[2,3], depth2=2, martingale_indices=[0,1,2]) 

    results = {'rbf': results_RBF, 'Matern': results_Matern, 'GA': results_GA, 'KER': results_KES, 'SES': results_SES, 'SES_mvr': results_SES_mvr}
    os.makedirs('experiments/ideal_gas', exist_ok=True)
    pickle.dump(results, open(f'experiments/ideal_gas/r={r}_ll={ll}.pkl', 'wb'))


if __name__ == "__main__":
    # global config
    rs = [3.5, 6.5]
    lls = [None, [0, 1, 2]]

    # select specific job using PBS_ARRAY_INDEX
    PBS_ARRAY_INDEX = int(sys.argv[1])
    r = rs[PBS_ARRAY_INDEX % 2]
    ll = lls[PBS_ARRAY_INDEX // 2] 

    print(f'Starting experiment r={r}, ll={ll}...')
    start_time = time.time()
    run_experiment(r=r, ll=ll)
    print(f'...ending experiment r={r}, ll={ll}. Tot time: {datetime.timedelta(seconds=(time.time() - start_time))}.')

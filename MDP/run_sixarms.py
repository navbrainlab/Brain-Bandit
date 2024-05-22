
from environment import make_riverSwim,make_riverSwim_origin, make_SixArms
from feature_extractor import FeatureTrueState
from agent import PSRL, UCRL2, OptimisticPSRL, OTS_MDP, EpsilonGreedy,UBE, UBE_TS, UBE_UCB, UBE_Bio
from experiment import run_finite_tabular_experiment
import numpy as np
if __name__ == '__main__':
    # seed = 0
    # targetPath = 'saved/riverswim_OptimisticPSRL_Q.csv'
    # env = make_riverSwim()
    # f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)
    # agent = OptimisticPSRL_Q(nState=env.nState, nAction=env.nAction, epLen=env.epLen, alpha0=1/env.nState)
    # regret_list = []
    # seed = 2
    # regret = run_finite_tabular_experiment(agent, env, f_ext, 1000, seed,
    #                                        recFreq=100, fileFreq=100, targetPath=targetPath, save=True)
    # print(regret)
    regret_list = []
    base_path = 'saved/sixarms_UBE_Bio_neutral'
    seed = 0
    for i in range(10):
        env = make_SixArms()
        f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)
        agent = UBE_Bio(nState=env.nState, nAction=env.nAction, epLen=env.epLen, alpha0=1 / env.nState)
        seed += 1
        targetPath = base_path + f'_seed{seed}.csv'
        regret = run_finite_tabular_experiment(agent, env, f_ext, 1000, seed,
                        recFreq=100, fileFreq=1, targetPath=targetPath, save=True)
        regret_list.append(regret)
    print(np.mean(regret_list))
    print(np.std(regret_list))
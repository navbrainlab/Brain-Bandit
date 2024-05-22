
from environment import make_riverSwim,make_riverSwim_origin, make_SixArms, make_riverSwim_small_reward
from feature_extractor import FeatureTrueState
from agent import PSRL, UCRL2, OptimisticPSRL, OTS_MDP, EpsilonGreedy,UBE, UBE_Bio, EpsilonGreedy
from experiment import run_finite_tabular_experiment
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
    targetPath = 'saved/riverswim_OptimisticPSRL_Q_O.csv'
    seed = 0
    q_val_dis = []
    q_var_dis = []
    for i in range(10):
        env = make_riverSwim_origin()
        f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)
        agent = PSRL(nState=env.nState, nAction=env.nAction, epLen=env.epLen, alpha0=1./env.nState)
        seed += 1
        regret = run_finite_tabular_experiment(agent, env, f_ext, 50, seed,
                        recFreq=10, fileFreq=1, targetPath=targetPath, save=False)
        regret_list.append(regret)
        # q_val_dis.append(agent.q_val_distribution)
        # q_var_dis.append(agent.q_var_distribution)
    q_val_dis = np.array(q_val_dis).flatten()
    q_var_dis = np.array(q_var_dis).flatten()
    # sns.distplot(q_val_dis, bins=100)
    # plt.show()
    # sns.distplot(q_var_dis, bins=100)
    # plt.show()
    print(np.mean(regret_list))
    print(np.std(regret_list))
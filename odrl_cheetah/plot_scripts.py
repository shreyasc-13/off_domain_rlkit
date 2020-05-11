import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd

def plotting_evalreturns(log_dir, rl_on_real):
    '''
    Plotting the returns from progress.csv
    '''
    data = pd.read_csv(log_dir+'/progress.csv')

    returns_avg = data['evaluation/Returns Mean']
    returns_std = data['evaluation/Returns Std']

    path_length = returns_avg.shape[0]

    if rl_on_real:
        train_env = 'Real'
    else:
        train_env = 'Sim'

    plt.plot(range(1, path_length+1), returns_avg, 'g')
    plt.fill_between(range(1, path_length+1), returns_avg + returns_std, \
            returns_avg - returns_std, alpha = 0.4, facecolor='g')
    plt.title('Real Env Eval Returns: RL on %s'%train_env)
    plt.ylabel('Average returns')
    plt.xlabel('Epochs')
    # plt.show()
    plt.savefig(log_dir+'/eval_return_realenv_%s.png'%train_env)

def plot_realsteps(log_dir_real, log_dir_odrl, log_dir_sim):
    '''
    Plotting returns vs real steps
    '''
    data_real = pd.read_csv(log_dir_real+'/progress.csv')
    data_odrl = pd.read_csv(log_dir_odrl+'/progress.csv')
    data_sim = pd.read_csv(log_dir_sim+'/progress.csv')

    returns_real = data_real['evaluation/Returns Mean']
    returns_odrl = data_odrl['evaluation/Returns Mean']
    returns_sim = data_sim['evaluation/Returns Mean']

    returns_real_std = data_real['evaluation/Returns Std']
    returns_odrl_std = data_odrl['evaluation/Returns Std']
    returns_sim_std = data_sim['evaluation/Returns Std']

    realenv_steps_real = data_real['real_exploration/num steps total']
    realenv_steps_odrl = data_odrl['real_exploration/num steps total']
    realenv_steps_sim = data_sim['real_exploration/num steps total']

    plt.plot(realenv_steps_real, returns_real, 'r', label='rl_on_real')
    plt.plot(realenv_steps_odrl, returns_odrl, 'g', label='odrl')
    # plt.plot(realenv_steps_sim, returns_sim, 'b', label='rl_on_sim')
    plt.plot(realenv_steps_odrl, [list(returns_sim)[-1]]*len(realenv_steps_odrl), 'b-', label='rl_on_sim')

    plt.fill_between(realenv_steps_real, returns_real + returns_real_std, \
            returns_real - returns_real_std, alpha = 0.4, facecolor='r')
    plt.fill_between(realenv_steps_odrl, returns_odrl + returns_odrl_std, \
            returns_odrl - returns_odrl_std, alpha = 0.4, facecolor='g')
    # plt.fill_between(realenv_steps_sim, returns_sim + returns_sim_std, \
    #         returns_sim - returns_sim_std, alpha = 0.4, facecolor='b')

    plt.legend()
    plt.title('Returns vs real steps: evaluated on realenv')
    plt.xlabel('Real steps')
    plt.ylabel('Return')

    log_dir_odrl = log_dir_odrl.replace('/', '_')

    plt.savefig(log_dir_real+'/../../'+log_dir_odrl+'real_steps_eval.png')

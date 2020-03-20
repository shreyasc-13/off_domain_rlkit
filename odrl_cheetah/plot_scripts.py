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
    plt.savefig(log_dir+'/eval_return_realenv.png')

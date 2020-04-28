"""
Common evaluation utilities.
"""

from collections import OrderedDict
from numbers import Number

import numpy as np

import rlkit.pythonplusplus as ppp
import torch

def get_generic_path_information(paths, rl_on_real=True, classifier=None, stat_prefix=''):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    # import pdb
    # pdb.set_trace()
    if not paths:
        stats= OrderedDict([('Rewards Mean',0), ('Rewards Std', 0), ('Rewards Max', 0), ('Rewards Min', 0), 
                            ('Returns Mean', 0), ('Returns Std', 0), ('Returns Max',0), ('Returns Min', 0), 
                            ('Actions Mean', 0), ('Actions Std', 0), ('Actions Max', 0), ('Actions Min', 0), 
                            ('Num Paths', 0),
                             #('Average Returns', 0), 
                             ('deltaR_1_rollout Mean', 0), ('deltaR_1_rollout Std', 0), ('deltaR_1_rollout Max', 0), ('deltaR_1_rollout Min', 0)])
        # print(stats)
        return stats

    statistics = OrderedDict()
    returns = [sum(path["rewards"]) for path in paths]
    # deltaR= [sum(path["rewards"]) for path in paths]
    rewards = np.vstack([path["rewards"] for path in paths]) if paths else []
    statistics.update(create_stats_ordered_dict('Rewards', rewards,
                                                stat_prefix=stat_prefix))
    statistics.update(create_stats_ordered_dict('Returns', returns,
                                                stat_prefix=stat_prefix))
    actions = [path["actions"] for path in paths] if paths else []
    if actions:
        if len(actions[0].shape) == 1:
            actions = np.hstack([path["actions"] for path in paths]) 
        else:
            actions = np.vstack([path["actions"] for path in paths])
    statistics.update(create_stats_ordered_dict(
        'Actions', actions, stat_prefix=stat_prefix
    ))
    statistics['Num Paths'] = len(paths)
    # statistics[stat_prefix + 'Average Returns'] = get_average_returns(paths)
    if paths:
        for info_key in ['env_infos', 'agent_infos']:
            if info_key in paths[0]:
                all_env_infos = [
                    ppp.list_of_dicts__to__dict_of_lists(p[info_key])
                    for p in paths
                ]
                for k in all_env_infos[0].keys():
                    final_ks = np.array([info[k][-1] for info in all_env_infos])
                    first_ks = np.array([info[k][0] for info in all_env_infos])
                    all_ks = np.concatenate([info[k] for info in all_env_infos])
                    statistics.update(create_stats_ordered_dict(
                        stat_prefix + k,
                        final_ks,
                        stat_prefix='{}/final/'.format(info_key),
                    ))
                    statistics.update(create_stats_ordered_dict(
                        stat_prefix + k,
                        first_ks,
                        stat_prefix='{}/initial/'.format(info_key),
                    ))
                    statistics.update(create_stats_ordered_dict(
                        stat_prefix + k,
                        all_ks,
                        stat_prefix='{}/'.format(info_key),
                    ))
            # import pdb; pdb.set_trace()
            #DeltaR
    if paths and not rl_on_real:
        first_path_obs = torch.Tensor(paths[0]['observations'])
        first_path_actions = torch.Tensor(paths[0]['actions'])
        first_path_next_obs = torch.Tensor(paths[0]['next_observations'])

        classifier_input=  torch.cat((first_path_obs, first_path_actions), 1)
        classifier_input=  torch.cat((classifier_input, first_path_next_obs), 1)
        outSAS, outSA=classifier(classifier_input)
        # import pdb;pdb.set_trace()
        
        deltaR= (outSAS[:, 1] - outSAS[:, 0]).reshape((-1,1))
        # print(stat_prefix)
        # print(deltaR)
        statistics.update(create_stats_ordered_dict('deltaR_1_rollout', deltaR.cpu().numpy(),
                                            stat_prefix=stat_prefix))
    # print(statistics)
    return statistics


def get_average_returns(paths):
    returns = [sum(path["rewards"]) for path in paths]
    return np.mean(returns)


def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=True,
        exclude_max_min=False,
):  
    

    if stat_prefix is not None:
        name = "{}{}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})
     
    if len(data) == 0:
        empty_stats= OrderedDict([
            (name + ' Mean', 0),
            (name + ' Std', 0),
        ])
        if not exclude_max_min:
            empty_stats[name + ' Max'] = 0
            empty_stats[name + ' Min'] = 0

        return empty_stats

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats

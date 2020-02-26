import abc
from collections import OrderedDict

import gtimer as gt

from rlkit.core import logger, eval_util
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import DataCollector


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times


class BaseRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
<<<<<<< HEAD
            sim_data_collector: DataCollector,
            real_data_collector: DataCollector,
            evaluation_data_collector: DataCollector,
            sim_replay_buffer: ReplayBuffer,
            real_replay_buffer: ReplayBuffer
=======
            exploration_data_collector: DataCollector,
            evaluation_data_collector: DataCollector,
            replay_buffer: ReplayBuffer,
>>>>>>> 90195b24604f513403e4d0fe94db372d16700523
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
<<<<<<< HEAD
        self.sim_data_collector = sim_data_collector
        self.real_data_collector = real_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.sim_replay_buffer = sim_replay_buffer
        self.real_replay_buffer= real_replay_buffer
=======
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
>>>>>>> 90195b24604f513403e4d0fe94db372d16700523
        self._start_epoch = 0

        self.post_epoch_funcs = []

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    def _end_epoch(self, epoch):
        snapshot = self._get_snapshot()
        logger.save_itr_params(epoch, snapshot)
        gt.stamp('saving')
        self._log_stats(epoch)

<<<<<<< HEAD
        self.sim_data_collector.end_epoch(epoch)
        self.real_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.sim_replay_buffer.end_epoch(epoch)
        self.real_replay_buffer.end_epoch(epoch)
=======
        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
>>>>>>> 90195b24604f513403e4d0fe94db372d16700523
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
<<<<<<< HEAD
        for k, v in self.sim_data_collector.get_snapshot().items():
            snapshot['sim_exploration/' + k] = v
        for k, v in self.real_data_collector.get_snapshot().items():
            snapshot['real_exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.real_replay_buffer.get_snapshot().items():
            snapshot['real_replay_buffer/' + k] = v
        for k, v in self.sim_replay_buffer.get_snapshot().items():
            snapshot['sim_replay_buffer/' + k] = v
        return snapshot


    #FIXLOGGER
=======
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

>>>>>>> 90195b24604f513403e4d0fe94db372d16700523
    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
<<<<<<< HEAD
            self.sim_replay_buffer.get_diagnostics(),
=======
            self.replay_buffer.get_diagnostics(),
>>>>>>> 90195b24604f513403e4d0fe94db372d16700523
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
<<<<<<< HEAD
            self.sim_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.sim_data_collector.get_epoch_paths()
=======
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.expl_data_collector.get_epoch_paths()
>>>>>>> 90195b24604f513403e4d0fe94db372d16700523
        if hasattr(self.expl_env, 'get_diagnostics'):
            logger.record_dict(
                self.expl_env.get_diagnostics(expl_paths),
                prefix='exploration/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(expl_paths),
            prefix="exploration/",
        )
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if hasattr(self.eval_env, 'get_diagnostics'):
            logger.record_dict(
                self.eval_env.get_diagnostics(eval_paths),
                prefix='evaluation/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(eval_paths),
            prefix="evaluation/",
        )

        """
        Misc
        """
        gt.stamp('logging')
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

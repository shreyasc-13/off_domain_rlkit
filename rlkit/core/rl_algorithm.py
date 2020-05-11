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
            sim_exploration_env,
            real_exploration_env,
            evaluation_sim_env,
            evaluation_real_env,
            sim_data_collector: DataCollector,
            real_data_collector: DataCollector,
            # evaluation_data_collector: DataCollector,
            eval_real_data_collector:DataCollector,
            eval_sim_data_collector:DataCollector,
            sim_replay_buffer: ReplayBuffer,
            real_replay_buffer: ReplayBuffer,
            rl_on_real,
    ):
        self.trainer = trainer
        self.sim_expl_env = sim_exploration_env
        self.real_expl_env=real_exploration_env
        self.eval_sim_env = evaluation_sim_env
        self.eval_real_env = evaluation_real_env

        self.sim_data_collector = sim_data_collector
        self.real_data_collector = real_data_collector
        self.eval_real_data_collector = eval_real_data_collector
        self.eval_sim_data_collector= eval_sim_data_collector
        self.sim_replay_buffer = sim_replay_buffer
        self.real_replay_buffer= real_replay_buffer
        self._start_epoch = 0
        self.rl_on_real=rl_on_real
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

        self.sim_data_collector.end_epoch(epoch)
        self.real_data_collector.end_epoch(epoch)
        self.eval_sim_data_collector.end_epoch(epoch)
        self.eval_real_data_collector.end_epoch(epoch)
        self.sim_replay_buffer.end_epoch(epoch)
        self.real_replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.sim_data_collector.get_snapshot().items():
            snapshot['sim_exploration/' + k] = v
        for k, v in self.real_data_collector.get_snapshot().items():
            snapshot['real_exploration/' + k] = v
        for k, v in self.eval_sim_data_collector.get_snapshot().items():
            snapshot['sim_evaluation/' + k] = v
        for k, v in self.eval_real_data_collector.get_snapshot().items():
            snapshot['real_evaluation/' + k] = v
        for k, v in self.real_replay_buffer.get_snapshot().items():
            snapshot['real_replay_buffer/' + k] = v
        for k, v in self.sim_replay_buffer.get_snapshot().items():
            snapshot['sim_replay_buffer/' + k] = v
        return snapshot

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.sim_replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        if not self.rl_on_real:

            if self.num_classifier_train_steps_per_iter:
                logger.record_dict(
                    self.classifier.get_diagnostics(),
                    prefix='classifier',)

            # logger.record_dict(OrderedDict([('num_real_steps_total',self.num_real_steps_total)]))
            logger.record_dict(
                    self.sim_data_collector.get_diagnostics(),
                    prefix='sim_exploration/'
                )
            if self.num_sim_steps_per_epoch:
                sim_expl_paths = self.sim_data_collector.get_epoch_paths()
                if hasattr(self.sim_expl_env, 'get_diagnostics'):
                    logger.record_dict(
                        self.sim_expl_env.get_diagnostics(sim_expl_paths),
                        prefix='sim_exploration/',
                    )
                logger.record_dict(
                    eval_util.get_generic_path_information(sim_expl_paths, self.rl_on_real, classifier=self.classifier.predict if not self.rl_on_real else None ),
                    prefix="sim_exploration/",
                )

        logger.record_dict(
            self.real_data_collector.get_diagnostics(),
            prefix='real_exploration/'
        )

        if not self.training_SAC or self.num_real_steps_per_epoch:
            real_expl_paths = self.real_data_collector.get_epoch_paths()
            if hasattr(self.real_expl_env, 'get_diagnostics'):
                logger.record_dict(
                    self.real_expl_env.get_diagnostics(real_expl_paths),
                    prefix='real_exploration/',
                )
            # print(real_expl_paths)
            logger.record_dict(
                eval_util.get_generic_path_information(real_expl_paths, self.rl_on_real, classifier=self.classifier.predict if not self.rl_on_real else None),
                prefix="real_exploration/",
            )

        else:
            prefix='real_exploration/'
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_real_data_collector.get_diagnostics(),
            prefix='eval_real/',
        )
        eval_paths = self.eval_real_data_collector.get_epoch_paths()
        if hasattr(self.eval_real_env, 'get_diagnostics'):
            logger.record_dict(
                self.eval_real_env.get_diagnostics(eval_paths),
                prefix='eval_real/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(eval_paths, self.rl_on_real,classifier=self.classifier.predict if not self.rl_on_real else None),
            prefix="eval_real/",
        )


        logger.record_dict(
            self.eval_sim_data_collector.get_diagnostics(),
            prefix='eval_sim/',
        )
        eval_paths = self.eval_sim_data_collector.get_epoch_paths()
        if hasattr(self.eval_sim_env, 'get_diagnostics'):
            logger.record_dict(
                self.eval_sim_env.get_diagnostics(eval_paths),
                prefix='eval_sim/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(eval_paths, self.rl_on_real, classifier=self.classifier.predict if not self.rl_on_real else None),
            prefix="eval_sim/",
        )

        logger.record_dict( OrderedDict([('num_trains_per_train_loop', self.num_train_loops_per_epoch)]),
            prefix="")

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

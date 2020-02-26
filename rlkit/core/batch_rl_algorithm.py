import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
<<<<<<< HEAD
=======
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
>>>>>>> 90195b24604f513403e4d0fe94db372d16700523
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
<<<<<<< HEAD
            num_trains_per_train_loop,
            evaluation_data_collector: PathCollector,
            sim_data_collector: PathCollector,
            real_data_collector: PathCollector,
            sim_replay_buffer: ReplayBuffer,
            real_replay_buffer: ReplayBuffer,
            num_real_steps_at_init=10000,
            num_sim_steps_at_init=0,
            num_real_steps_per_epoch=100,
            num_sim_steps_per_epoch=0,
            num_rl_train_steps_per_iter=1,
            rl_on_real=True,
            modify_reward=False,
            num_classifier_train_steps_per_iter=1,
            num_train_loops_per_epoch=1

=======
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
>>>>>>> 90195b24604f513403e4d0fe94db372d16700523
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
<<<<<<< HEAD
            sim_data_collector,
            real_data_collector, 
            evaluation_data_collector,
            sim_replay_buffer,
            real_replay_buffer
=======
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
>>>>>>> 90195b24604f513403e4d0fe94db372d16700523
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
<<<<<<< HEAD
        # self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        # self.min_num_steps_before_training = min_num_steps_before_training
        self.sim_data_collector = sim_data_collector
        self.real_data_collector = real_data_collector
        self.num_real_steps_at_init = num_real_steps_at_init
        self.num_sim_steps_at_init = num_sim_steps_at_init 
        self.num_real_steps_per_epoch = num_real_steps_per_epoch
        self.num_sim_steps_per_epoch = num_sim_steps_per_epoch
        self.num_classifier_train_steps_per_iter = num_classifier_train_steps_per_iter
        # self.num_rl_train_steps_per_iter = num_rl_train_steps_per_iter
        self.rl_on_real = rl_on_real
        # self.modify_reward = modify_reward


    def _train(self):

        # INIT REPLAY BUFFER
        self.burn_in_memory()

        #If our method, train the classifier on INIT replay buffer.
        if not self.rl_on_real:
            self.trainer.classifier_init_training( self.sim_replay_buffer,self.real_replay_buffer)
        
        #TRAIN SAC 
        # for num_epochs(default 3000), evaluate 
        for epoch in gt.timed_for(range(self._start_epoch, self.num_epochs),save_itrs=True,):
            
            # collect real env evaluate steps #TODO where is it getting used?
            self.evaluate()

            #for train_loops_per_epoch(SAC default: 1000, my default:1), collect more data on sim/real as per the requirements and add it in the buffer)
            for _ in range(self.num_train_loops_per_epoch):
                #Collect new sim and real expl paths
                self.add_new_experince_to_buffer()

                self.training_mode(True)
                # for num_trains_per_train_loop, randomly sample from the buffer and train. 
                for _ in range(self.num_trains_per_train_loop):
                    if self.rl_on_real:
                        train_data =self.real_replay_buffer.random_batch(self.batch_size)
                        self.trainer.train(train_data)

                    else:
                        train_data = self.sim_replay_buffer.random_batch(self.batch_size)
                        classifier_train_data=torch.cat((self.sim_replay_buffer.random_batch(self.batch_size),
                                                        self.real_replay_buffer.random_batch(self.batch_size)),0)
                        self.trainer.train(train_data)
                        self.trainer.classifier.classifier_train_from_torch(classifier_train_data, num_classifier_train_steps_per_iter)

                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)




    def burn_in_memory(self):
        # Filling real replay buffer at the start with the real world steps
        if self.num_real_steps_at_init:
            init_real_expl_paths = self.real_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_real_steps_at_init,
                discard_incomplete_paths=False,
            )
            self.real_replay_buffer.add_paths(init_real_expl_paths)
            self.real_data_collector.end_epoch(-1)

        # Filling sim replay buffer at the start with the sim world steps
        if self.num_sim_steps_at_init:
            init_sim_expl_paths = self.sim_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_sim_steps_at_init,
                discard_incomplete_paths=False,
            )
            self.sim_replay_buffer.add_paths(init_sim_expl_paths)
            self.sim_data_collector.end_epoch(-1)

    def evaluate(self):
        self.eval_data_collector.collect_new_paths(
=======
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
>>>>>>> 90195b24604f513403e4d0fe94db372d16700523
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
<<<<<<< HEAD
        gt.stamp('evaluation sampling')


    def add_new_experince_to_buffer(self):
        #collect new sim expl paths

        new_sim_paths = self.sim_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_sim_steps_per_epoch,
            discard_incomplete_paths=False,
        )
        gt.stamp('exploration sim sampling', unique=False)
        self.sim_replay_buffer.add_paths(new_sim_paths)
        gt.stamp('data storing', unique=False)

        #collect new real expl paths
        new_real_paths = self.sim_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_real_steps_per_epoch,
            discard_incomplete_paths=False,
        )
        gt.stamp('exploration real sampling', unique=False)
        self.real_replay_buffer.add_paths(new_real_paths)
        gt.stamp('data storing', unique=False)




=======
            gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)
>>>>>>> 90195b24604f513403e4d0fe94db372d16700523

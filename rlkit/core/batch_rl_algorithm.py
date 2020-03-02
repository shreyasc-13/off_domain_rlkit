import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
from classifiers import classifier
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import torch
from pointenv import plot_env
import numpy as np
class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            sim_exploration_env,
            real_exploration_env,            
            evaluation_env,
            batch_size,
            max_path_length,
            max_episode_steps,
            num_epochs,
            num_eval_steps_per_epoch,
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
            num_train_loops_per_epoch=1,
            num_classifier_init_epoch=50,
            classifier_batch_size=1024,
            tolerance=1


    ):



        super().__init__(
            trainer,
            sim_exploration_env,
            real_exploration_env,
            # exploration_env,
            evaluation_env,
            sim_data_collector,
            real_data_collector, 
            evaluation_data_collector,
            sim_replay_buffer,
            real_replay_buffer,
            rl_on_real,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.max_episode_steps=max_episode_steps
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
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
        self.classifier_batch_size=classifier_batch_size
        self.num_classifier_init_epoch= num_classifier_init_epoch
        self.sim_exploration_env=sim_exploration_env
        self.real_exploration_env=real_exploration_env
        self.evaluation_env=evaluation_env
        self.training_SAC=False
        # self.num_real_steps_total=0
        # self.modify_reward = modify_reward
        self.tolerance=tolerance


    def _train(self):

        # INIT REPLAY BUFFER

        self.SAC_burn_in_memory()
 
        #If our method: train the classifier on INIT replay buffer.
        if not self.rl_on_real and self.num_sim_steps_at_init and self.num_real_steps_at_init :
            sim_init_memory=self.sim_replay_buffer.random_batch(self.num_sim_steps_at_init )
            real_init_memory=self.real_replay_buffer.random_batch(self.num_real_steps_at_init)
            self.classifier=classifier()
            self.classifier.classifier_init_training( sim_init_memory,real_init_memory, init_classifier_batch_size=self.classifier_batch_size, num_epochs=self.num_classifier_init_epoch)
        

        # for num_epochs(default 3000), evaluate 

        
        # self.plot_for_all_epochs() #plot all epoch results

        for epoch in gt.timed_for(range(self._start_epoch, self.num_epochs),save_itrs=True,):
            self.training_SAC=True

            # collect real env evaluate steps. It will just stores the stats which will be used in logging with end_epoch. 
            #putting end of epoch here because we also want to log at the very begining of the training. 

            #for train_loops_per_epoch(SAC default: 1000, my default:1),
            for _ in range(self.num_train_loops_per_epoch):
                #Collect new sim and real expl paths
                self.sim_new_paths, self.real_new_paths=self.add_new_experince_to_buffer()

                self.training_mode(True)
                # for num_trains_per_train_loop, randomly sample from the buffer and train. 
                for _ in range(self.num_trains_per_train_loop):
                    if self.rl_on_real:
                        train_data =self.real_replay_buffer.random_batch(self.batch_size)
                        self.trainer.train(train_data)

                    else:
                        train_data = self.sim_replay_buffer.random_batch(self.batch_size)
                        self.trainer.train(train_data,modify_reward=True, classifier=self.classifier.SAS_Network.Network)
            # self.plot_path_steps(epoch)
            if not self.rl_on_real:
                #train the classifier with random samples from both the buffers
                for _ in range(self.num_classifier_train_steps_per_iter):
                    self.classifier.classifier_train_from_batch(
                        self.sim_replay_buffer.random_batch(self.classifier_batch_size),
                        self.real_replay_buffer.random_batch(self.classifier_batch_size)
                        )
                gt.stamp('training', unique=False)
                self.training_mode(False)
            self.eval_new_paths=self.evaluate(epoch)
            self._end_epoch(epoch) # logs all statistics

        # self.eval_new_paths=self.evaluate(epoch)
        # self._end_epoch(epoch)
        # plt.show()


    def SAC_burn_in_memory(self):

        # Filling real replay buffer at the start with the real world steps
        if self.num_real_steps_at_init:
            init_real_expl_paths = self.real_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_real_steps_at_init,
                discard_incomplete_paths=False,
            )
            self.real_replay_buffer.add_paths(init_real_expl_paths)
            # print(self.num_real_steps_at_init, len(init_real_expl_paths), self.max_episode_steps)
            # self.num_real_steps_total+=self.num_real_steps_at_init
            self.real_data_collector.end_epoch(-1)

        # Filling sim replay buffer at the start with the sim world steps
        if not self.rl_on_real and self.num_sim_steps_at_init:
            init_sim_expl_paths = self.sim_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_sim_steps_at_init,
                discard_incomplete_paths=False,
            )
            self.sim_replay_buffer.add_paths(init_sim_expl_paths)
            self.sim_data_collector.end_epoch(-1)

    def evaluate(self, epoch):
        eval_paths=self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=False,
            )
        acc=0
        for path in eval_paths:
            if np.linalg.norm(path["next_observations"][-1])<self.tolerance:
                acc+=1
        acc/=len(eval_paths)
        self.acc=acc
        gt.stamp('evaluation sampling')
        return eval_paths


    def add_new_experince_to_buffer(self):
        new_sim_paths, new_real_paths=None, None
        #collect new sim expl paths
        if self.num_sim_steps_per_epoch:
            new_sim_paths = self.sim_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_sim_steps_per_epoch,
                discard_incomplete_paths=False,
            )
            gt.stamp('exploration sim sampling', unique=False)
            self.sim_replay_buffer.add_paths(new_sim_paths)
            gt.stamp('data storing', unique=False)

        if self.num_real_steps_per_epoch:
            #collect new real expl paths
            new_real_paths = self.real_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_real_steps_per_epoch,
                discard_incomplete_paths=False,
            )
            # self.num_real_steps_total +=self.num_real_steps_per_epoch
            gt.stamp('exploration real sampling', unique=False)
            self.real_replay_buffer.add_paths(new_real_paths)
            gt.stamp('data storing', unique=False)

        return new_sim_paths, new_real_paths

    def plot_for_all_epochs():
        # fig, axes= plt.subplots(nrows=self.num_epochs, ncols=10,figsize=[20,int(((self.num_epochs-1)/5)+1)* 5])
        # # cols=["SimExpPath0", "RealExpPath0", "RealEvalPath0","SimExpPath1", "RealExpPath1", "RealEvalPath1"]
        # cols=["Ep"+str(epoch+1) for epoch in range(self.num_epochs)]
        # for ax, col in zip(axes[0], cols):
        #     ax.set_title(col, size='small')
        # for ax, col in zip(axes[0], col):
        #     ax.set_ylabel(size='small')
        # plt.suptitle('ODRL Output With Rl On Real: '+str(self.rl_on_real)+
        #              ", Classifier Training Online: "+ str(self.num_classifier_train_steps_per_iter), 
        #              fontsize=20)
        # fig.tight_layout()
        # plt.show()
        # self._end_epoch(-1)
        pass

    def plot_path_steps(self, epoch):
         #trying to make a plot of all epocs
        for k in range(self.num_epochs):
            for i in range(min(2,len(self.sim_new_paths))):
                plt.subplot(self.num_epochs,6,epoch*6+(i*3+1))
                plt.subplot(self.num_epochs,6,epoch*6+(i*3+1))
                plt.subplot(self.num_epochs,6,epoch*6+(i*3+1))

        # for i in range(min(2,len(self.sim_new_paths))):
            sim_states=self.sim_new_paths[i]["next_observations"]
            real_states=self.real_new_paths[i]["next_observations"]
            eval_states=self.eval_new_paths[i]["next_observations"]
            # print(len(sim_states), len(real_states), len(eval_states))
            plot_env(self.evaluation_env)
            plt.subplot(2, self.num_epochs,epoch*2+(i+1))
            # plot_env(self.sim_exploration_env)
            plt.plot(sim_states[:,0].tolist(),sim_states[:,1].tolist(),"-b.", label="simExplpath")
            # plt.subplot(self.num_epochs,6,epoch*6+(i*3+2))
            # plot_env(self.real_exploration_env)
            plt.plot(real_states[:,0].tolist(),real_states[:,1].tolist(),"-r.", label="realExpPath")
            # plt.subplot(self.num_epochs,6,epoch*6+(i*3+3))
            plt.plot(eval_states[:,0].tolist(),eval_states[:,1].tolist(),"-y.", label="valEvalPath")




 
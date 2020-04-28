import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
from classifiers_redo import classifier_ensambler, classifier#, mixer, convert_to_SAS_input_form
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
# import torch
# from pointenv import plot_env
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            sim_exploration_env,
            real_exploration_env,
            evaluation_sim_env ,         
            evaluation_real_env,
            batch_size,
            max_path_length,
            # max_episode_steps,
            num_epochs,
            num_eval_steps_per_epoch,
            num_trains_per_train_loop,
            evaluation_real_data_collector: PathCollector,
            evaluation_sim_data_collector: PathCollector,
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
            # modify_reward=False,
            num_classifier_train_steps_per_iter=1,
            num_train_loops_per_epoch=1,
            num_classifier_init_epoch=50,
            classifier_batch_size=1024,
            tolerance=1,
            plot_episodes_period=10,
            hardcode_classifier=False, 
            init_paths_random=False,
            constant_start_state_init=True, 
            constant_start_state_while_training=True, 
            should_plot=False, 
            seed=1,
            num_SA=0,
            num_SAS=1, 
            iid_at_init=False, 
            render=False, 
            lamda=1,
            fixed_lamda=1, 
            max_real_collection_epoch= 1000
    ):



        super().__init__(
            trainer,
            sim_exploration_env,
            real_exploration_env,
            evaluation_sim_env,
            evaluation_real_env,
            sim_data_collector,
            real_data_collector, 
            evaluation_real_data_collector,
            evaluation_sim_data_collector,
            sim_replay_buffer,
            real_replay_buffer,
            rl_on_real,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        # self.max_episode_steps=max_episode_steps
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
        # self.evaluation_env=evaluation_env
        self.training_SAC=False
        # self.num_real_steps_total=0
        # self.modify_reward = modify_reward
        self.tolerance=tolerance
        self.plot_episodes_period=plot_episodes_period
        self.hardcode_classifier=hardcode_classifier

        self.init_paths_random=init_paths_random, 
        self.constant_start_state_init=constant_start_state_init
        self.constant_start_state_while_training=constant_start_state_while_training
        self.should_plot=should_plot
        self.seed=seed
        self.num_rows=0
        self.num_cols=0
        self.subplot_num=0
        self.num_SA=num_SA
        self.num_SAS=num_SAS
        self.iid_at_init=iid_at_init
        self.render=render
        self.fixed_lamda=fixed_lamda
        self.lamda=lamda
        self.max_real_collection_epoch= max_real_collection_epoch

    def _train(self):

        # INIT REPLAY BUFFER

        self.SAC_burn_in_memory(use_policy=False)
        # import pdb;pdb.set_trace()
        #If our method: train the classifier on INIT replay buffer.

        if not self.rl_on_real:
            self.classifier=classifier_ensambler(num_SA=self.num_SA, num_SAS=self.num_SAS, seed=self.seed, hardcode=self.hardcode_classifier,  real_env=self.real_exploration_env, sim_env=self.sim_exploration_env)
            if not self.hardcode_classifier and self.num_sim_steps_at_init and self.num_real_steps_at_init:
                sim_init_memory=self.sim_replay_buffer.random_batch(self.num_sim_steps_at_init )
                real_init_memory=self.real_replay_buffer.random_batch(self.num_real_steps_at_init)
                self.classifier.classifier_init_training( sim_init_memory,real_init_memory, 
                                                        init_classifier_batch_size=self.classifier_batch_size, 
                                                        num_epochs=self.num_classifier_init_epoch)
        else:
            self.classifier=None

        # for num_epochs(default 3000), evaluate 
        if self.should_plot:
            self.plot_path_before_training() #plot all epoch results

        for self.epoch in gt.timed_for(range(self._start_epoch, self.num_epochs),save_itrs=True,):
            self.training_SAC=True

            # collect real env evaluate steps. It will just stores the stats which will be used in logging with end_epoch. 
            #putting end of epoch here because we also want to log at the very begining of the training. 

            #for train_loops_per_epoch(SAC default: 1000, my default:1),
            for _ in range(self.num_train_loops_per_epoch):
                #Collect new sim and real expl paths
                self.sim_new_paths, self.real_new_paths=self.add_new_experince_to_buffer()

                self.training_mode(True)
                # for num_trains_per_train_loop, randomly sample from the buffer and train. 
                for train_num in range(self.num_trains_per_train_loop):
                    if self.rl_on_real:
                        train_data =self.real_replay_buffer.random_batch(self.batch_size)
                        self.trainer.train(train_data)


                    else:
                        train_data = self.sim_replay_buffer.random_batch(self.batch_size)
                        # import pdb;pdb.set_trace()
                        if not self.hardcode_classifier:
                            self.trainer.train(train_data,
                                            modify_reward=True, 
                                            classifier=self.classifier.predict,
                                            plot_classifier=True if (not (self.epoch-1)%self.plot_episodes_period and not train_num) else False, 
                                            subplot_num= (self.num_rows, self.num_cols, self.plot_num+ 4*self.num_cols) if (self.epoch and self.should_plot and not train_num) else None, 
                                            lamda=self.epoch*0.005 if not self.fixed_lamda else self.lamda
                                            #TODO: if aneealing lamda, logs will become wrong in 
                                            )
                        else:
                            self.trainer.train(train_data,
                                            modify_reward=True, 
                                            classifier=self.classifier.hardcode_predict)
            if not self.rl_on_real and not self.hardcode_classifier:
                #train the classifier with random samples from both the buffers
                for _ in range(self.num_classifier_train_steps_per_iter):
                    self.classifier.classifier_train_from_batch(
                        self.sim_replay_buffer.random_batch(self.classifier_batch_size),
                        self.real_replay_buffer.random_batch(self.classifier_batch_size)
                        )   
                gt.stamp('training', unique=False)
                self.training_mode(False)

                # batch=train_data
                # rewards = batch['rewards']
                # terminals = batch['terminals']
                # obs = batch['observations']
                # actions = batch['actions']
                # next_obs = batch['next_observations']
                # classifier_input=  torch.cat((obs, actions), 1)
                # classifier_input=  torch.cat((classifier_input, next_obs), 1)
                # outSAS=self.classifier.SAS_Network.predict(classifier_input)
                # deltaR= (torch.log(outSAS[:, 1]) - torch.log(outSAS[:, 0])).reshape((-1,1))
                # rewards=rewards+deltaR
                    

            self.evaluate()

            self._end_epoch(self.epoch) # logs all statistics
            if self.should_plot and not self.epoch%self.plot_episodes_period:
                self.plot_path_steps()
        # self.eval_new_paths=self.evaluate(epoch)
        # self._end_epoch(epoch)
        if self.should_plot:
            plt.show()


    def SAC_burn_in_memory(self, use_policy):
        # Filling real replay buffer at the start with the real world steps
        max_steps=1 if self.iid_at_init==True else self.max_path_length
        if self.num_real_steps_at_init:
            init_real_expl_paths = self.real_data_collector.collect_new_paths(
                max_steps,
                self.num_real_steps_at_init,
                discard_incomplete_paths=False,
                collect_random_path=self.init_paths_random, 
                constant_start_state=self.constant_start_state_init


                # use_policy=False

            )
            self.real_replay_buffer.add_paths(init_real_expl_paths)
            # print(self.num_real_steps_at_init, len(init_real_expl_paths), self.max_episode_steps)
            # self.num_real_steps_total+=self.num_real_steps_at_init
            self.real_data_collector.end_epoch(-1)

        # Filling sim replay buffer at the start with the sim world steps
        if not self.rl_on_real and self.num_sim_steps_at_init:
            init_sim_expl_paths = self.sim_data_collector.collect_new_paths(
                max_steps,
                self.num_sim_steps_at_init,
                discard_incomplete_paths=False,
                collect_random_path=self.init_paths_random, 
                constant_start_state=self.constant_start_state_init
                # use_policy=False
            )
            self.sim_replay_buffer.add_paths(init_sim_expl_paths)
            self.sim_data_collector.end_epoch(-1)





    def evaluate(self):
        self.eval_real_new_paths=self.eval_real_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=False,
                render=self.render
            )
        acc=0
        for path in self.eval_real_new_paths:
            if np.linalg.norm(path["next_observations"][-1])<self.tolerance:
                acc+=1
        acc/=len(self.eval_real_new_paths)
        self.eval_real_acc=acc
        gt.stamp('real evaluation sampling')


        self.eval_sim_new_paths=self.eval_sim_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.max_path_length*2, # just devided by 5 to save some computation as eval on sim world is less imp than eval on real world
                    discard_incomplete_paths=False,
                )
        acc=0
        for path in self.eval_sim_new_paths:
            if np.linalg.norm(path["next_observations"][-1])<self.tolerance:
                acc+=1
        acc/=len(self.eval_sim_new_paths)
        self.eval_sim_acc=acc
        gt.stamp('evaluation sampling')


    def add_new_experince_to_buffer(self):
        new_sim_paths, new_real_paths=None, None
        #collect new sim expl paths
        if self.num_sim_steps_per_epoch:
            new_sim_paths = self.sim_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_sim_steps_per_epoch,
                discard_incomplete_paths=False,
                constant_start_state=self.constant_start_state_while_training
            )
            gt.stamp('exploration sim sampling', unique=False)
            self.sim_replay_buffer.add_paths(new_sim_paths)
            gt.stamp('data storing', unique=False)

        if ((self.num_real_steps_per_epoch) and (self.epoch<self.max_real_collection_epoch)):
            #collect new real expl paths
            new_real_paths = self.real_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_real_steps_per_epoch,
                discard_incomplete_paths=False,
                constant_start_state=self.constant_start_state_while_training

            )
            # self.num_real_steps_total +=self.num_real_steps_per_epoch
            gt.stamp('exploration real sampling', unique=False)
            self.real_replay_buffer.add_paths(new_real_paths)
            gt.stamp('data storing', unique=False)

        return new_sim_paths, new_real_paths    
    def plot_path_before_training(self):
        # fig, axes= plt.subplots(nrows=self.num_epochs, ncols=10,figsize=[20,int(((self.num_epochs-1)/5)+1)* 5])
        # cols=["SimExpPath0", "RealExpPath0", "RealEvalPath0","SimExpPath1", "RealExpPath1", "RealEvalPath1"]
        # cols=["Ep"+str(self.epoch+1) for self.epoch in range(self.num_epochs)]
        # for ax, col in zip(axes[0], cols):
        #     ax.set_title(col, size='small')
        # for ax, col in zip(axes[0], col):
        #     ax.set_ylabel(size='small')
        # plt.suptitle('ODRL Output With Rl On Real: '+str(self.rl_on_real)+
        #              ", Classifier Training Online: "+ str(bool(self.num_classifier_train_steps_per_iter)), 
        #              fontsize=20)
        # fig.tight_layout()
        self.fig=plt.figure(figsize=((self.num_epochs+1)/10*6,20))
        # self._end_epoch(-1)
        # self.eval_new_paths=self.evaluate(-1)
        self.plot_num=0
        # self.plot_path_steps()

    def plot_path_steps(self):
        num_paths=15
        self.num_rows=6
        self.num_cols=int(self.num_epochs/self.plot_episodes_period)+1
        #trying to make a plot of all epocs
        self.plot_num+=1
        if self.plot_num>self.num_rows*self.num_cols:
            return


        colormap = plt.cm.gist_ncar
        plt.gca().set_prop_cycle(color=[colormap(i) for i in np.linspace(0, 0.9, num_paths)])

        eval_new_paths=[self.eval_real_new_paths, self.eval_sim_new_paths]
        

        for j in range(len(eval_new_paths)):
            plt.subplot(self.num_rows, self.num_cols, self.plot_num+ j*self.num_cols)
            plt.title("epoch"+ str(self.epoch))
            # m_expl_env.observation_space.low[0]
            plt.ylim(self.sim_exploration_env.observation_space.low[0], self.sim_exploration_env.observation_space.high[0])
            plt.xlim(self.sim_exploration_env.observation_space.low[1], self.sim_exploration_env.observation_space.high[1])
            # print(i,j)
            # plt.title str(self.epoch))
            print(len(eval_new_paths[j]))
            for i in range(min(num_paths,len(eval_new_paths[j]))):
                eval_states=eval_new_paths[j][i]["next_observations"]
                plt.plot(eval_states[:,0].tolist(),eval_states[:,1].tolist(),"-")

            #plot the state distribution
            # import pdb;pdb.set_trace()
            self.threeD_data_distribution( eval_new_paths[j], 
                                            j, 
                                            bin_size=7,
                                        env_range=[self.sim_exploration_env.observation_space.low[0], self.sim_exploration_env.observation_space.high[0]], 
                                        subplot_num= (self.num_rows, self.num_cols, self.plot_num+ (2+j)*self.num_cols),
                                        title= "RealWorld Evaluation State Distribution")
        plot_classifier=True
        # import pdb;pdb.set_trace()
        # if plot_classifier:
        #     sim_classification_paths=convert_to_SAS_input_form(self.sim_replay_buffer.random_batch(1000))
        #     real_classification_paths=convert_to_SAS_input_form(self.real_replay_buffer.random_batch(1000))
        #     paths_combined, Y=mixer(sim_classification_paths, real_classification_paths)
        #     outSAS=self.classifier.SAS_Network.predict(paths_combined)
        #     deltaR= (torch.log(outSAS[:, 1]) - torch.log(outSAS[:, 0])).reshape((-1,1))
        #     plt.subplot(self.num_rows, self.num_cols,self.plot_num+ (4)*self.num_cols)
        #     cm = plt.cm.get_cmap('RdYlBu')
        #     sc = plt.scatter(obs[:,0].tolist(),obs[:,1].tolist(), marker='.',  c=deltaR, cmap=cm)
        #     plt.colorbar(sc)
            # plt.subplot(self.num_rows, self.num_cols,self.plot_num+ (6)*self.num_cols)
            # cm = plt.cm.get_cmap('RdYlBu')
            # sc = plt.scatter(obs[:,0].tolist(),obs[:,1].tolist(), marker='.',  c=rewards, cmap=cm)
            # plt.colorbar(sc)

        # plt.subplot(2, self.num_cols,  self.plot_num+ 2*self.num_cols)
        # for i in range(min(num_paths,len(self.eval_real_new_paths))):
        #     eval_states=self.eval_real_new_paths[i]["next_observations"]
        #     plt.plot(eval_states[:,0].tolist(),eval_states[:,1].tolist(),"-")



 
    def threeD_data_distribution(self,eval_new_paths, j, bin_size, env_range, subplot_num, title):
        ax = self.fig.add_subplot(*subplot_num, projection='3d')
        # ax.set_title(title)

        next_obs=np.array([eval_new_paths[i]["next_observations"] for i in range(len(eval_new_paths))])
        xy=np.concatenate(next_obs, axis=0)
        # import pdb
        # pdb.set_trace()
        hist, xedges, yedges = np.histogram2d(xy[:,0], xy[:,1], bins=bin_size, range=[env_range, env_range])
        xpos, ypos = np.meshgrid(xedges[:-1] , yedges[:-1], indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = np.zeros_like(xpos)
        dx = 0.5 * np.ones_like(zpos)
        dy = dx.copy()
        dz = hist.flatten()
        colors = plt.cm.jet(dz/float(dz.max()))
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average')




 #        for k in range(self.num_epochs):

 #            for i in range(min(2,len(self.sim_new_paths))):
 #                plt.subplot(self.num_epochs,6,epoch*6+(i*3+1))
 #                plt.subplot(self.num_epochs,6,epoch*6+(i*3+1))
 #                plt.subplot(self.num_epochs,6,epoch*6+(i*3+1))

 #        # for i in range(min(2,len(self.sim_new_paths))):
 #            sim_states=self.sim_new_paths[i]["next_observations"]
 #            real_states=self.real_new_paths[i]["next_observations"]
 #            eval_states=self.eval_new_paths[i]["next_observations"]
 #            # print(len(sim_states), len(real_states), len(eval_states))
 #            plot_env(self.evaluation_env)
 #            plt.subplot(2, self.num_epochs,epoch*2+(i+1))
 #            # plot_env(self.sim_exploration_env)
 #            plt.plot(sim_states[:,0].tolist(),sim_states[:,1].tolist(),"-b.", label="simExplpath")
 #            # plt.subplot(self.num_epochs,6,epoch*6+(i*3+2))
 #            # plot_env(self.real_exploration_env)
 #            plt.plot(real_states[:,0].tolist(),real_states[:,1].tolist(),"-r.", label="realExpPath")
 #            # plt.subplot(self.num_epochs,6,epoch*6+(i*3+3))
 #            plt.plot(eval_states[:,0].tolist(),eval_states[:,1].tolist(),"-y.", label="valEvalPath")




 # 
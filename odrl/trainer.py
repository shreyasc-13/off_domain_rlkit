from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.torch.core import np_to_pytorch_batch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import  ReduceLROnPlateau
import copy
from torch.autograd import Variable

class SACTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None, 
            classifier=None

    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.logs={"policy_training_loss":[], "policy_val_loss":[], }
        self.classifier=classifier




    def train_from_torch(self, batch, modify_reward=False):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        if modify_reward:
            classifier_input=  torch.cat((obs, actions), 1)
            classifier_input=  torch.cat((classifier_input, next_obs), 1)
            outSAS=self.classifier(classifier_input)
            deltaR= (torch.log(outSAS[:, 1]) - torch.log(outSAS[:, 0])).reshape((-1,1))
            rewards=(alpha*rewards+deltaR)/(alpha+1)
        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.

            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
        )


    def classifier_init_training(self,  sim_replay_buffer,real_replay_buffer):
        batch_size=1024
        SAS_model = Network(input_size =6, output_size = 2, unit_count = 32)
        SAS_model.apply(init_model)
        SAS_optimizer= torch.optim.Adam(SAS_model.parameters(), lr=10e-3)
        SAS_scheduler= ReduceLROnPlateau(SAS_optimizer, 'min')
        trainX,trainY,valX,valY,testX,testY=get_data(sim_replay_buffer,real_replay_buffer)
        train_dataset =SAS_loader(trainX,trainY)
        train_dataloader=DataLoader(train_dataset,shuffle=False, batch_size=batch_size, drop_last=True)
        val_dataset =SAS_loader(valX,valY)
        val_dataloader=DataLoader(val_dataset,shuffle=False, batch_size=batch_size, drop_last=True)
        self.SAS_Network =   Networks( Network=SAS_model ,
                                    optimizer=SAS_optimizer,     
                                    batch_size=batch_size, 
                                    scheduler=SAS_scheduler, 
                                    train_loader= train_dataloader,
                                    val_loader=val_dataloader,
                                    model_name="SAS"
                                    )
        self.SAS_Network.init_train(50)

    def classifier_train_from_torch(self, sim_batch, real_batch):
        sim_SAS_in = convert_to_SAS_input_form(sim_batch)
        real_SAS_in = convert_to_SAS_input_form(sim_batch)
        data=torch.cat((sim_SAS_in, real_SAS_in), 0)
        label=torch.cat((torch.Tensor([0]*len(sim_SAS_in)), torch.Tensor([1]*len(sim_SAS_in))), 0)
        self.SAS_Network.train(data, label)



def get_data(sim_memory,real_memory):
    sim_memory=convert_to_SAS_input_form(sim_memory)
    real_memory=convert_to_SAS_input_form(real_memory)
    X,Y=mixer(sim_memory, real_memory)
    len_data=len(Y)
    trainX=X[:int(len_data*.80)]
    trainY=Y[:int(len_data*.80)]
    valX=X[int(len_data*.80): int(len_data*.90)]
    valY=Y[int(len_data*.80): int(len_data*.90)]
    testX=X[int(len_data*.90):]
    testY=Y[int(len_data*.90):]
    return trainX,trainY,valX,valY,testX,testY

 
def mixer(X, X1):
    X2=torch.cat((X,X1), 0)
    Y=torch.Tensor([[0] for i in range(len(X))])
    Y1=torch.Tensor([[1] for i in range(len(X))])
    Y2=torch.cat((Y,Y1),0)

    XY=torch.cat((X2,Y2),1)
    XYshuffled=XY[torch.randperm(XY.shape[0])]

    # X2=copy.deepcopy(X)
    # Y2=copy.deepcopy(Y)
    # torch.cat((Y2,Y1),0)

    # data = list(zip(X2, Y2))
    # random.shuffle(data)
    # X2, Y2 = zip(*data)
    return  XYshuffled[:,0:6],XYshuffled[:,-1]  


def convert_to_SAS_input_form(batch):
    obs = batch['observations']
    actions = batch['actions']
    next_obs = batch['next_observations']
    mem = torch.cat((torch.Tensor(obs), torch.Tensor(actions)),1)
    mem = torch.cat((torch.Tensor(mem), torch.Tensor(next_obs)),1)
    return mem

class Network(nn.Module):
    def __init__(self, input_size, output_size, unit_count):
        super().__init__()
        self.layer1 = nn.Sequential(
                          nn.Linear(input_size, unit_count),
                          nn.Tanh())
        self.layer2 = nn.Sequential(
                        nn.Linear(unit_count, unit_count),
                        nn.Tanh())
        self.layer3 = nn.Sequential(
                        nn.Linear(unit_count, unit_count),
                        nn.Tanh())
        self.layer4 = nn.Sequential(
                          nn.Linear(unit_count, output_size),
                          nn.Softmax(dim=1))

        return

    def forward(self, x):
        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def init_model(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)
        return
cuda = torch.cuda.is_available()

device = torch.device("cuda" if cuda else "cpu")

class  Networks(object ):
    def __init__(self, Network , optimizer, 
                        batch_size ,scheduler, 
                        train_loader,val_loader,
                        model_name="SAS"):

        self.Network=Network
        self.optimizer=optimizer

        self.batch_size=batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler=scheduler

        self.train_loader= train_loader
        self.val_loader=val_loader
        self.metrics={
                    "loss": [],
                    "acc":[],
                    }
        self.model_name=model_name
        #early stopping params.
        self.best_val_loss=10e14
        self.patience=10
        self.wait=0
        self.min_delta=10e-4

    def early_stop(self,val_loss=10e14):
        if self.best_val_loss-val_loss>self.min_delta:
            self.best_val_loss=val_loss
            self.wait=0
            return False
        self.wait+=1
        if self.wait>self.patience:
            return True

    def save_models(self):
        torch.save(self.Network.state_dict(), path+self.model_name+"Network.pt")


    def validate(self,epoch):
        self.Network.eval()
        with torch.no_grad():
            val_acc=0
            val_loss=0
            for batch_idx, (data, label) in enumerate(self.val_loader):
                data = data.to(device)
                label = label.long().to(device)
                self.optimizer.zero_grad()
                Y = Variable(label)
                inp = Variable(data)
                out = self.Network(inp.float() )
                val_loss+=self.criterion(out, Y).data
                predictions=out[:,1]>0.5
                val_acc += (predictions.long() == label).float().sum()
            total=(batch_idx+1)*len(label)
            val_acc=val_acc/total
            val_loss=val_loss
            self.scheduler.step(val_loss) 
            self.metrics["loss"].append(val_loss)
            self.metrics["acc"].append(val_acc)

            print("Epoch {},  Val Loss: {}, Val Accuracy: {}".format(epoch+1, val_loss, val_acc))
            return val_loss

    def init_train(self, max_epoch):
        print("training")
        self.Network.train()
        self.Network.to(device)

        loss=10e14
        epoch=0     
        while(not self.early_stop(loss) and epoch<max_epoch):

            self.Network.train()
            train_loss = 0
            train_acc = 0
            for batch_idx, (data, label) in enumerate(self.train_loader):
                self.train(data, label)
            loss=self.validate(epoch)
            epoch+=1
        # if save_data:
        #   self.save_models()
        #   save_pickle( self.model_name+"metrics.pkl", self.metrics)
        return

    def train(self, data, label):
        data = data.to(device)
        label = label.long().to(device)
        self.optimizer.zero_grad()
        inp = Variable(data)
        Y = Variable(label)

        out = self.Network(inp.float())
        loss = self.criterion(out, Y) 
        loss.backward()
        print("mini_batch train loss: ", loss.data)
        self.optimizer.step()

class SA_loader(Dataset):
    def __init__(self, X, Y):

        self.X=torch.Tensor(X)
        self.Y=torch.Tensor(Y)

    def __getitem__(self, index):
        return self.X[index][0:4], self.Y[index]
        
    def __len__(self):
        return len(self.Y)

class SAS_loader(Dataset):
    def __init__(self, X, Y):

        self.X=torch.Tensor(X)
        self.Y=torch.Tensor(Y)

    def __getitem__(self, index):
        return self.X[index][0:6], self.Y[index]
        
    def __len__(self):
        return len(self.Y)





# def save_data(trainX,trainY,valX,valY, testX, testY):
#     save_pickle("trainX.pkl",trainX)
#     save_pickle("trainY.pkl", trainY)
#     save_pickle("valX.pkl",valX)
#     save_pickle("valY.pkl", valY)    
#     save_pickle("testX.pkl",testX)
#     save_pickle("testY.pkl", testY) 
#     return

# def load_data():
#     trainX=load_pickle("trainX.pkl")
#     trainY=load_pickle("trainY.pkl")
#     valX=load_pickle("valX.pkl")
#     valY=load_pickle("valY.pkl")    
#     testX=load_pickle("testX.pkl")
#     testY=load_pickle("testY.pkl")
#     return trainX,trainY,valX,valY,testX,testY



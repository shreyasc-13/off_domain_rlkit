import torch
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import  ReduceLROnPlateau
from rlkit.torch.networks import FlattenMlp
from collections import OrderedDict
import math
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
import numpy as np
from torch.nn import functional as F


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
        # self.layer4 = nn.Sequential(
        #                 nn.Linear(unit_count, unit_count),
        #                 nn.Tanh())
        self.layer4 = nn.Sequential(
                          nn.Linear(unit_count, output_size),
                          nn.Softmax(dim=1))

        return

    def forward(self, x):
        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.layer5(x)
        return x


class classifier:   
    def __init__( self,  init_classifier_batch_size=1024,hardcode=False, real_env=None,  sim_env=None, seed=1, SA=False ):

        torch.manual_seed(seed)
        cuda = torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        self.SA=SA
        self.name="SA_" if self.SA else "SAS_"
        self.obs_dim = sim_env.observation_space.low.size
        self.action_dim = sim_env.action_space.low.size
        if hardcode==True:
            self.hardcode=hardcode(sim_env, real_env, SA, self.obs_dim, self.action_dim)
        else:
            self.input_size=self.obs_dim+self.action_dim if self.SA else 2*self.obs_dim+self.action_dim
            self.model =Network(input_size=self.input_size, output_size=2, unit_count=32)
            # self.model = FlattenMlp(input_size = self.input_size, output_size = 2, hidden_sizes= [64, 64,  64], hidden_activation=nn.tanh,  output_activation=nn.Softmax(dim=1), layer_norm=True,)
            self.optimizer= torch.optim.Adam(self.model.parameters(), lr=10e-3)
            self.scheduler= ReduceLROnPlateau(self.optimizer, 'min')
            self._train_loss=[]
            self._train_acc=[]
            self.criterion = nn.CrossEntropyLoss()
            self.metrics={"loss": [],"acc":[]}

    def  classifier_init_training(self, sim_replay_buffer, real_replay_buffer, init_classifier_batch_size, num_epochs):
        self.batch_size=init_classifier_batch_size
        trainX,trainY,valX,valY,testX,testY=self.get_data(sim_replay_buffer,real_replay_buffer)
        train_dataset =loader(trainX,trainY)
        self.train_loader=DataLoader(train_dataset,shuffle=False, batch_size=self.batch_size, drop_last=True)
        val_dataset =loader(valX,valY)
        self.val_loader=DataLoader(val_dataset,shuffle=False, batch_size=self.batch_size, drop_last=True)     
        self.best_val_loss=10e14; self.patience=10; self.wait=0; self.min_delta=10e-4
        self.init_train(num_epochs)


    def init_train(self, max_epoch):
        print("training")
        self.model.train()
        self.model.to(device)
        loss=10e14
        epoch=0     
        while(not self.early_stop(loss) and epoch<max_epoch):
            self.model.train()
            train_loss = 0
            train_acc = 0
            for batch_idx, (data, label) in enumerate(self.train_loader):
                train_loss, train_acc=self.train(data, label)
            loss=self.validate(epoch)
            epoch+=1
        return

    def train(self, data, label):
        self.model.train()
        self.optimizer.zero_grad()
        inp = Variable(data.to(device))
        Y = Variable(label.long().to(device))
        out = self.model(inp.float())
        loss = self.criterion(out, Y) 
        loss.backward()
        self.optimizer.step()
        predictions=out[:,1]>0.5; acc = ((predictions.long() == Y).float().sum())/len(label)
        return loss.data, acc

 
    def classifier_train_from_batch(self, sim_batch, real_batch):
        sim_in = self.convert_to_input_form(sim_batch)
        real_in = self.convert_to_input_form(real_batch)
        data=torch.cat((sim_in, real_in), 0)
        label=torch.cat((torch.Tensor([0]*len(sim_in)), torch.Tensor([1]*len(real_in))), 0)
        loss, acc=self.train(data, label)
        self._train_loss.append(loss), self._train_acc.append(acc)



    def validate(self,epoch):
        val_acc=0; val_loss=0
        for batch_idx, (data, label) in enumerate(self.val_loader):
            self.optimizer.zero_grad()
            inp = Variable(data.to(device))
            Y = Variable(label.long().to(device))
            out = self.predict(inp)
            predictions=out[:,1]>0.5 
            val_loss+=self.criterion(out, Y).data; 
            val_acc += (predictions.long() == label.long().to(device)).float().sum()
        total=(batch_idx+1)*len(label)
        val_acc=val_acc/total ; val_loss=val_loss/total
        self.scheduler.step(val_loss) 
        self.metrics["loss"].append(val_loss); self.metrics["acc"].append(val_acc)
        print("Epoch {},  Val Loss: {}, Val Accuracy: {}".format(epoch+1, val_loss, val_acc))
        return val_loss

    def predict(self, data):
            inp=Variable(data[:,:self.input_size].to(device))
            self.model.eval()
            with torch.no_grad():
                out = self.model(inp.float())
            return out

    def early_stop(self,val_loss=10e14):
        if self.best_val_loss-val_loss>self.min_delta:
            self.best_val_loss=val_loss
            self.wait=0
            return False
        self.wait+=1
        if self.wait>self.patience:
            return True

    def get_diagnostics(self):
        loss=torch.mean(torch.Tensor(self._train_loss))
        acc=torch.mean(torch.Tensor(self._train_acc))
        res=OrderedDict([( self.name+' classifier train loss',loss.data[0] ), (self.name+' classifier train acc', acc.data[0] )])
        self._train_loss, self._train_acc=[],[]
        return res

    def mixer(self, X, X1):
        X2=torch.cat((X,X1), 0)
        Y=torch.Tensor([[0] for i in range(len(X))])
        Y1=torch.Tensor([[1] for i in range(len(X1))])
        Y2=torch.cat((Y,Y1),0)
        XY=torch.cat((X2,Y2),1)
        XYshuffled=XY[torch.randperm(XY.shape[0])]
        return  XYshuffled[:,0:self.input_size],XYshuffled[:,-1]  

    def get_data(self, sim_memory,real_memory):
        sim_memory=self.convert_to_input_form(sim_memory)
        real_memory=self.convert_to_input_form(real_memory)
        X,Y=self.mixer(sim_memory, real_memory)
        len_data=len(Y)
        trainX=X[:int(len_data*.90)]
        trainY=Y[:int(len_data*.90)]
        valX=X[int(len_data*.90):]
        valY=Y[int(len_data*.90):]
        return trainX,trainY,valX,valY, None, None

    # TODO: should I normalize the input
    def convert_to_input_form(self, batch):
        obs = batch['observations']
        actions = batch['actions']
        mem = torch.cat((torch.Tensor(obs), torch.Tensor(actions)),1)
        if not self.SA:
            next_obs = batch['next_observations']
            mem = torch.cat((torch.Tensor(mem), torch.Tensor(next_obs)),1)
        return mem

    def save_models(self):
         
        torch.save(self.model.state_dict(), self.name+"Network.pt")



class hardcode():
    def __init__(self,sim_env, real_env , is_sa=False, obs_dim=2 ,action_dim=2):
        self.sim_env=sim_env 
        self.real_env=real_env
        self.SA=SA
        self.obs_dim ,self.action_dim= obs_dim ,action_dim


    def predict(self, input):
        states=input[:,0:self.obs_dim]

        if not self.SA:
            next_states=input[:, self.obs_dim+self.action_dim:self.obs_dim+self.action_dim+self.obs_dim]
            p_real=math.exp(-100)*torch.Tensor(
                                        np.logical_or(self.real_env._is_blocked_parellel(states.data.numpy()),
                                                        self.real_env._is_blocked_parellel(next_states.data.numpy())).astype(int))
        else:
            p_real=math.exp(-100)*torch.Tensor(self.real_env._is_blocked_parellel(states.data.numpy()))
        p_real[p_real==0]=0.5

        #for plotting P distrbution distribution on the 2D pointenv. 
        # blocking=[[0 for i in range(7)] for j in range (7)]
        # for i in range(7):
        #     for j in range(7):
        #         blocking[i][j]=self.real_env._is_blocked(np.array((i+0.5,j+0.5)))631  111.91  160 2
        # print(blocking)
        # import matplotlib.pyplot as plt
        # cm = plt.cm.get_cmap('RdYlBu')
        # print(next_states)
        # sc = plt.scatter(next_states[:,0].tolist(),next_states[:,1].tolist(), marker='.',  c=p_real, cmap=cm)
        # plt.colorbar(sc)
        # plt.show()
        return torch.cat([(1 - p_real)[:,None], p_real[:, None]], dim=1)





class loader(Dataset):   
    def __init__(self, X, Y):
        self.X=torch.Tensor(X)
        self.Y=torch.Tensor(Y)

    def __getitem__(self, index):
        return self.X[index][0:6], self.Y[index]
        
    def __len__(self):
        return len(self.Y)



class classifier_ensambler():
    def __init__( self, num_SA=0, num_SAS=1,  init_classifier_batch_size=1024,hardcode=False, real_env=None,  sim_env=None, seed=1):
        if  (not num_SA and not num_SAS) or (num_SA <0 or num_SAS<0):
            print(" num_SA or SAS are either both 0 or one of them is negative value, program won't work")
            return
        self.num_SA=num_SA
        self.num_SAS=num_SAS
        self.SAS_classifiers=[];
        self.SA_classifiers=[]
        for i in range(self.num_SAS):
            self.SAS_classifiers.append(classifier(  init_classifier_batch_size=init_classifier_batch_size,hardcode=hardcode, real_env=real_env,  sim_env=sim_env, seed=i+num_SAS*seed, SA=False))
        if num_SA:
            SA_classifier=[]
            for i in range(self.num_SA):
                self.SA_classifiers.append(classifier(  init_classifier_batch_size=init_classifier_batch_size,hardcode=hardcode, real_env=real_env,  sim_env=sim_env, seed=i+num_SAS*seed, SA=True))

    def classifier_init_training(self, sim_replay_buffer, real_replay_buffer, init_classifier_batch_size, num_epochs):
        for i in range(self.num_SAS):
            self.SAS_classifiers[i].classifier_init_training(sim_replay_buffer, real_replay_buffer, init_classifier_batch_size, num_epochs)
        for i in range(self.num_SA):
            self.SA_classifiers[i].classifier_init_training(sim_replay_buffer, real_replay_buffer, init_classifier_batch_size, num_epochs)

    def train(self, data, label):
        for i in range(self.num_SAS):
            self.SAS_classifiers[i].train(data, label)
        for i in range(self.num_SA):
            self.SA_classifiers[i].train(data, label)

    def  predict(self, data, result_type="mean"):
        out, out_SA=[], []
        for i in range(self.num_SAS):
            out.append(self.SAS_classifiers[i].predict(data))
        for i in range(self.num_SA):
            out_SA.append(self.SA_classifiers[i].predict(data))
        out_SA=[0] if not self.num_SA else out_SA
           
        if result_type=='min':
            return min(out), min(out_SA)  
        elif result_type=='max':
            return max(out), max(out_SA)
        else:
            return sum(out)/len(out), sum(out_SA)/len(out_SA)
        return 
    def hardcode_predict(self,sim_env, real_env , is_sa=False, obs_dim=2 ,action_dim=2, result_type="mean"):
        out, out_SA=[],[]
        for i in range(self.num_SAS):
            out.append(self.SAS_classifiers[i].hardcode.predict(data))
        for i in range(self.num_SA):
            out_SA.append(self.SA_classifiers[i].hardcode.predict(data))
        out_SA=[0] if not self.num_SA else out_SA

        if result_type=='min':
            return min(out), min(out_SA)
        elif result_type=='max':
            return max(out), max(out_SA)
        else:
            return sum(out)/len(out), sum(out_SA)/len(out_SA)
        return 


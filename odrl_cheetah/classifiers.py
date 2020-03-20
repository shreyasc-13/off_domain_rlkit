import torch
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import  ReduceLROnPlateau
from collections import OrderedDict
import math

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

'''
Util functions for classifer
'''
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
    Y1=torch.Tensor([[1] for i in range(len(X1))])
    Y2=torch.cat((Y,Y1),0)
    XY=torch.cat((X2,Y2),1)
    XYshuffled=XY[torch.randperm(XY.shape[0])]
    return  XYshuffled[:,0:58],XYshuffled[:,-1]

def convert_to_SAS_input_form(batch):
    obs = batch['observations']
    actions = batch['actions']
    next_obs = batch['next_observations']
    mem = torch.cat((torch.Tensor(obs), torch.Tensor(actions)),1)
    mem = torch.cat((torch.Tensor(mem), torch.Tensor(next_obs)),1)
    return mem

def init_model(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)
        return

'''
Network template:
Three hidden layers of size 'unit_count'
'''
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

'''
Class defining the NN
'''
class  Networks(object):
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
        self.path='models/'

    def early_stop(self,val_loss=10e14):
        if self.best_val_loss-val_loss>self.min_delta:
            self.best_val_loss=val_loss
            self.wait=0
            return False
        self.wait+=1
        if self.wait>self.patience:
            return True

    def save_models(self):
        torch.save(self.Network.state_dict(), self.path+self.model_name+"Network.pt")


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
        # train_epoch_loss=[0]
        loss=10e14
        epoch=0
        while(not self.early_stop(loss) and epoch<max_epoch):

            self.Network.train()
            train_loss = 0
            train_acc = 0
            for batch_idx, (data, label) in enumerate(self.train_loader):
                train_loss, train_acc=self.train(data, label)
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
        predictions=out[:,1]>0.5
        acc = ((predictions.long() == label).float().sum())/len(label)
        self.optimizer.step()
        return loss.data, acc

    def predict(self, data):
        data = data.to(device)
        inp=Variable(data)
        self.Network.eval()
        with torch.no_grad():
            out = self.Network(inp.float())
        return out

'''
Hardcoded classifier: Currently not used
'''
# class SAS_hardcode():
#     def __init__(self,sim_env, real_env ):
#         self.sim_env=sim_env
#         self.real_env=real_env
#
#     def predict(self, SAS_input):
#         import pdb
#         state=SAS_input[:, 0:1]
#         next_states=SAS_input[:, 3:4]
#         p_real=math.exp(-200)*torch.Tensor(
#                             [ 1 if(
#                             self.real_env._is_blocked(state[i].data[0]) or self.real_env._is_blocked(next_states[i].data[0]) )
#                             else 0 for i in range(len(state)) ])
#         p_real[p_real==0]=0.5  #0.5 chance of real or fake
#
#         return torch.cat([p_real[:, None], (1 - p_real)[:,None]], dim=1)#torch.cat((1-p_real, p_real), axis=-1)

'''
Classifier class
'''
# sas_dim = 0
#Shreyas TODO: Fix hardcong to applyt to all envs
class classifier:
    def __init__( self,  init_classifier_batch_size=1024,hardcode=False, real_env=None,  sim_env=None):
        self.env_state_dim = 26
        self.env_action_dim = 6
        self.sas_dim = 2*self.env_state_dim + self.env_action_dim
        if hardcode==True:
            self.SAS_hardcode=SAS_hardcode(sim_env, real_env)
        else:
            self.SAS_model = Network(input_size =self.sas_dim, output_size = 2, unit_count = 100)
            self.SAS_model.apply(init_model)
            self.SAS_optimizer= torch.optim.Adam(self.SAS_model.parameters(), lr=10e-3)
            self.SAS_scheduler= ReduceLROnPlateau(self.SAS_optimizer, 'min')
            self._train_loss=[]
            self._train_acc=[]

    def  classifier_init_training(self, sim_replay_buffer,real_replay_buffer, init_classifier_batch_size, num_epochs):
        self.batch_size=init_classifier_batch_size
        trainX,trainY,valX,valY,testX,testY=get_data(sim_replay_buffer,real_replay_buffer)
        train_dataset =SAS_loader(trainX,trainY)
        train_dataloader=DataLoader(train_dataset,shuffle=False, batch_size=self.batch_size, drop_last=True)
        val_dataset =SAS_loader(valX,valY)
        val_dataloader=DataLoader(val_dataset,shuffle=False, batch_size=self.batch_size, drop_last=True)
        self.SAS_Network =   Networks( Network=self.SAS_model ,
                                    optimizer=self.SAS_optimizer,
                                    batch_size=self.batch_size,
                                    scheduler=self.SAS_scheduler,
                                    train_loader= train_dataloader,
                                    val_loader=val_dataloader,
                                    model_name="SAS"
                                    )
        self.SAS_Network.init_train(num_epochs)

        # return SAS_Network


    def classifier_train_from_batch(self, sim_batch, real_batch):
        sim_SAS_in = convert_to_SAS_input_form(sim_batch)
        real_SAS_in = convert_to_SAS_input_form(real_batch)
        data=torch.cat((sim_SAS_in, real_SAS_in), 0)
        label=torch.cat((torch.Tensor([0]*len(sim_SAS_in)), torch.Tensor([1]*len(real_SAS_in))), 0)
        loss, acc=self.SAS_Network.train(data, label)
        self._train_loss.append(loss), self._train_acc.append(acc)


    def get_diagnostics(self):
        loss=torch.mean(torch.Tensor(self._train_loss))
        acc=torch.mean(torch.Tensor(self._train_acc))
        # print("Classifier Train loss",self._train_loss," Classifier Train acc", self._train_acc)
        res=OrderedDict([('SAS classifier train loss',loss.data[0] ), ('SAS classifier train acc', acc.data[0] )])
        self._train_loss, self._train_acc=[],[]
        return res

'''
Data loader
'''
class SAS_loader(Dataset):
    def __init__(self, X, Y):

        self.X=torch.Tensor(X)
        self.Y=torch.Tensor(Y)

    def __getitem__(self, index):
        return self.X[index][0:58], self.Y[index]

    def __len__(self):
        return len(self.Y)

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import pandas as pd
import csv

real=pd.read_csv("/home/swapnil/DRL/offDyna/rlkit-master/rlkit/data/name-of-experiment/sparse_real/progress.csv")
simoffline=pd.read_csv("/home/swapnil/DRL/offDyna/rlkit-master/rlkit/data/name-of-experiment/sparse_offline/progress.csv")
simonline=pd.read_csv("/home/swapnil/DtheRL/offDyna/rlkit-master/rlkit/data/name-of-experiment/sparse_online/progress.csv")
# simoffline2=pd.read_csv("/home/swapnil/DRL/offDyna/rlkit-master/rlkit/data/name-of-experiment/simoffline6/progress.csv")

rx=real["real_exploration/num steps total"].tolist()
ry=real['evaluation/Returns Mean'].tolist()
snx=simonline["real_exploration/num steps total"].tolist()
sny=simonline['evaluation/Returns Mean'].tolist()
sfx=simoffline["real_exploration/num steps total"].tolist()
sfy=simoffline['evaluation/Returns Mean'].tolist()
# sfx2=simoffline2["real_exploration/num steps total"].tolist()
# sfy2=simoffline2['evaluation/Returns Mean'].tolist()
plt.subplot(1,2,1)
plt.title("Mean Returns Over num Real Samples, Sparse", size="small");plt.xlabel("num real samples"); plt.ylabel("Returns Mean");
plt.plot(rx,ry,'-r')
plt.plot(snx,sny,'-g')
plt.plot(sfx,sfy,'-b')
# plt.plot(sfx2,sfy2,'-c')
plt.subplot(1,2,2)
plt.title("Mean Returns Over Epochs, Sparse", size="small"); plt.xlabel("epoches"); plt.ylabel("Returns Mean")
line1=plt.plot([i+1 for i in range(len(rx))], ry, "-r");
line2=plt.plot([i+1 for i in range(len(sfy))], sfy, "-b")
# line3=plt.plot([i+1 for i in range(len(sfy2))], sfy2, "-c")
line4=plt.plot([i+1 for i in range(len(sny))], sny, "-g")
plt.legend(("real, no modified reward(MR)", "MR,only new sim data","MR,only new sim data", "MR,also new real data"), fontsize=5)

plt.tight_layout()
plt.show()


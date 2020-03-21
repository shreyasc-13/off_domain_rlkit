import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
# real=pd.read_csv("/home/swapnil/DRL/offDyna/rlkit-master/rlkit/data/name-of-experiment/sparse_real/progress.csv")
# simoffline=pd.read_csv("/home/swapnil/DRL/offDyna/rlkit-master/rlkit/data/name-of-experiment/sparse_offline/progress.csv")
# simonline=pd.read_csv("/home/swapnil/DtheRL/offDyna/rlkit-master/rlkit/data/name-of-experiment/sparse_online/progress.csv")
# # simoffline2=pd.read_csv("/home/swapnil/DRL/offDyna/rlkit-master/rlkit/data/name-of-experiment/simoffline6/progress.csv")

# rx=real["real_exploration/num steps total"].tolist()
# ry=real['evaluation/Returns Mean'].tolist()
# snx=simonline["real_exploration/num steps total"].tolist()
# sny=simonline['evaluation/Returns Mean'].tolist()
# sfx=simoffline["real_exploration/num steps total"].tolist()
# sfy=simoffline['evaluation/Returns Mean'].tolist()
# # sfx2=simoffline2["real_exploration/num steps total"].tolist()
# # sfy2=simoffline2['evaluation/Returns Mean'].tolist()
# plt.subplot(1,2,1)
# plt.title("Mean Returns Over num Real Samples, Sparse", size="small");plt.xlabel("num real samples"); plt.ylabel("Returns Mean");
# plt.plot(rx,ry,'-r')
# plt.plot(snx,sny,'-g')
# plt.plot(sfx,sfy,'-b')
# # plt.plot(sfx2,sfy2,'-c')
# plt.subplot(1,2,2)
# plt.title("Mean Returns Over Epochs, Sparse", size="small"); plt.xlabel("epoches"); plt.ylabel("Returns Mean")
# line1=plt.plot([i+1 for i in range(len(rx))], ry, "-r");
# line2=plt.plot([i+1 for i in range(len(sfy))], sfy, "-b")
# # line3=plt.plot([i+1 for i in range(len(sfy2))], sfy2, "-c")
# line4=plt.plot([i+1 for i in range(len(sny))], sny, "-g")
# plt.legend(("real, no modified reward(MR)", "MR,only new sim data","MR,only new sim data", "MR,also new real data"), fontsize=5)

# plt.tight_layout()
# plt.show()



# plot_folder="/home/swapnil/DRL/offDyna/rlkit-master/rlkit/data/name-of-experiment/without_modified_rewards"

# dirs=os.listdir(plot_folder)
# print(dirs)

# # print(dirs)
# dirs= [os.path.join(plot_folder, dir_) for dir_ in dirs]
# progresses=[pd.read_csv(os.path.join(folder,"progress.csv")) for folder in dirs if os.path.isfile(os.path.join(folder,"progress.csv")) ]# and "on_real" in os.path.join(folder,"progress.csv") ]

# # [print(key) for key in progresses[0].keys()]
# xes=[progress["real_exploration/num steps total"].tolist() for progress in progresses]
# yes=[progress['eval_real/Returns Mean'].tolist() for progress in progresses]
# # print(xes[0], yes[0])
# plt.subplot(1,2,1)
# plt.title("Mean Returns Over #real Samples, Sparse, Stocastic_val_set", size="small"); plt.xlabel("Num real samples"); plt.ylabel("Returns Mean")
# lines=[]
# # colors=["r", "brown","b" ]

# for i in range(len(xes)):
# 	line, =plt.plot(xes[i], yes[i],alpha=0.25,  label=str(i))#, color=colors[i])
# 	lines.append(line)
# 	# print(dirs[i])
# plt.legend(handles=lines, loc='lower right')
# # #plot per real step
# # xe,ye=[plt.plot(x,y) for x,y in list(zip(xes,yes))] 
# # # print(xe, ye)
# plt.subplot(1,2,2)
# plt.title("Mean Returns Over Epochs, Sparse, Stocastic_val_set", size="small"); plt.xlabel("epoches"); plt.ylabel("Returns Mean")
# lines=[]
# for i in range(len(xes)):
# 	line, =plt.plot(yes[i],  label=str(i))#, color= colors[i])
# 	lines.append(line)
# plt.legend(handles=lines, loc='lower right')
# plt.show()


plot_folder="/home/swapnil/DRL/offDyna/rlkit-master/rlkit/data/name-of-experiment/batch2/"

dirs=os.listdir(plot_folder)
dirs= [os.path.join(plot_folder, dir_) for dir_ in dirs]
# print(dirs)
substrs=["-real","-batch_real",  "-hardcode"]# , "-classifier_online", "-classifier_offline", "-hardcode"]
color=["r", "m",  "g"]#, "c", "b"]

resize=["5", "7", "9"]
plt.figure(figsize=(20,20))
for j in range(len(resize)):
	for i in range(len(substrs)):
		for folder in dirs:
			print(folder)
			if  resize[j]+substrs[i] in folder and os.path.isfile(os.path.join(folder,"progress.csv")):
				print(folder,resize[j]+substrs[i]  )

				data=pd.read_csv(os.path.join(folder,"progress.csv"))
				num_real=data["real_exploration/num steps total"].tolist()
				returns=data['eval_real/Returns Mean'].tolist()
				pathbreak=folder.split("-")
				# print(pathbreak)
				# plt.title("resized env factor: "+resize[j])
				plt.subplot(2,3, j+1)
				plt.plot(num_real, returns,alpha=0.4, color=color[i],label=substrs[i])
				plt.subplot(2,3, j+4)
				plt.plot(returns, alpha=0.4, color=color[i], label=substrs[i])
plt.show()
			

# progresses_real=[pd.read_csv(os.path.join(folder,"progress.csv")) for folder in dirs if  "-real" in folder and os.path.isfile(os.path.join(folder,"progress.csv"))  ]# and "on_real" in os.path.join(folder,"progress.csv") ]
# progresses_classifier_online=[pd.read_csv(os.path.join(folder,"progress.csv")) for folder in dirs if  "-online" in folder  os.path.isfile(os.path.join(folder,"progress.csv")) ]# and "on_real" in os.path.join(folder,"progress.csv") ]
# progresses_classifier_offline=[pd.read_csv(os.path.join(folder,"progress.csv")) for folder in dirs if  "-offline" in folder os.path.isfile(os.path.join(folder,"progress.csv")) ]# and "on_real" in os.path.join(folder,"progress.csv") ]
# progresses_classifier_hardcode=[pd.read_csv(os.path.join(folder,"progress.csv")) for folder in dirs if  "-hardcode" in folder os.path.isfile(os.path.join(folder,"progress.csv")) ]# and "on_real" in os.path.join(folder,"progress.csv") ]

# print(dirs)
# dirs= [os.path.join(plot_folder, dir_) for dir_ in dirs]
# # progresses_real=[pd.read_csv(os.path.join(folder,"progress.csv")) for folder in dirs if  "-real" in folder and os.path.isfile(os.path.join(folder,"progress.csv"))  ]# and "on_real" in os.path.join(folder,"progress.csv") ]
# # progresses_classifier_online=[pd.read_csv(os.path.join(folder,"progress.csv")) for folder in dirs if  "-online" in folder  os.path.isfile(os.path.join(folder,"progress.csv")) ]# and "on_real" in os.path.join(folder,"progress.csv") ]
# # progresses_classifier_offline=[pd.read_csv(os.path.join(folder,"progress.csv")) for folder in dirs if  "-offline" in folder os.path.isfile(os.path.join(folder,"progress.csv")) ]# and "on_real" in os.path.join(folder,"progress.csv") ]
# # progresses_classifier_hardcode=[pd.read_csv(os.path.join(folder,"progress.csv")) for folder in dirs if  "-hardcode" in folder os.path.isfile(os.path.join(folder,"progress.csv")) ]# and "on_real" in os.path.join(folder,"progress.csv") ]

# # [print(key) for key in progresses[0].keys()]
# xes=[progress["real_exploration/num steps total"].tolist() for progress in progresses]
# yes=[progress['eval_real/Returns Mean'].tolist() for progress in progresses]
# # print(xes[0], yes[0])
# plt.subplot(1,2,1)
# plt.title("Mean Returns Over #real Samples, Sparse, Stocastic_val_set", size="small"); plt.xlabel("Num real samples"); plt.ylabel("Returns Mean")
# lines=[]
# # colors=["r", "brown","b" ]

# for i in range(len(xes)):

# 	line, =plt.plot(xes[i], yes[i],alpha=0.25,  label=str(i))#, color=colors[i])
# 	lines.append(line)
# 	# print(dirs[i])
# plt.legend(handles=lines, loc='lower right')
# # #plot per real step
# # xe,ye=[plt.plot(x,y) for x,y in list(zip(xes,yes))] 
# # # print(xe, ye)
# plt.subplot(1,2,2)
# plt.title("Mean Returns Over Epochs, Sparse, Stocastic_val_set", size="small"); plt.xlabel("epoches"); plt.ylabel("Returns Mean")
# lines=[]
# for i in range(len(xes)):
# 	line, =plt.plot(yes[i],  label=str(i))#, color= colors[i])
# 	lines.append(line)
# plt.legend(handles=lines, loc='lower right')
# plt.show()



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
# ["init_episodes1-mean0.2-nameLunar_lander-nctspi1-num_trains_per_train_loop", "-resize_factor1-rl_on_real0-seed0-std0.3"]



plot_folder="/home/swapnil/DRL/offDyna/rlkit-master/rlkit/data/new_expts/"#name-of-experiment#batch3/"

dirs=os.listdir(plot_folder)
dirs= [os.path.join(plot_folder, dir_) for dir_ in dirs]
# print(dirs)
# for dir_ in dirs:
# if  "classifier_online-" in dir_:
# 	os.rename(dir_,dir_[-9]+"ff"+dir_[-8:])

# import pdb; pdb.set_trace()
# substrs=[  "-classifier_offline_iid", "-ensamble_3_SAS_0_SA_mean_classifier_offline_iid",  "-ensamble_3_SAS_3_SA_mean_classifier_offline_iid",]#,"-batch_rl_on_real_iid", "-barch_rl_iid", #, "-classifier_online2", "-sim_online"]#, "-batch_rl_big"]# , "-classifier_online", "-classifier_offline", "-hardcode"]
# substrs=["-Lunar_lander_classifier_online_max_epi_steps_1000"]#["-Lunar_lander_rl_on_real", "-Lunar_lander_rl_on_real_hyperparamters_changed", "-Lunar_lander_rl_on_real_max_epi_steps_3000"]
# substrs=[("init_episodes1-mean0.2-nameLunar_lander-nctspi1-num_trains_per_train_loop", "-resize_factor1-rl_on_real0-seed0-std0.3"), 
# 		("init_episodes1-mean0.2-nameLunar_lander-nctspi1-num_trains_per_train_loop", "-resize_factor1-rl_on_real1-seed0-std0.3"), 
# 		("init_episodes1-mean0.2-nameLunar_lander_rl_on_sim_with_unmodified_R-nctspi1-num_trains_per_train_loop", "-resize_factor1-rl_on_real1-seed0-std0.3")]
substrs=[("init_episodes1-mean0.6-nameLunar_lander_delta_scaled-nctspi10-num_trains_per_train_loop2000-resize_factor1-rl_on_real0-seed0-std", "-unmodified_reward")]
# /home/swapnil/DRL/offDyna/rlkit-master/rlkit/data/name-of-experiment/name-of-experiment_2020_04_12_17_44_15_0000--s-init_episodes12000-resize_factor1-rl_on_real1-seed0-std0.3
# names=["rl_on_sim_with_modified_rewards"," rl_on_real", "rl_on_sim_with_unmodified_rewards"]#, "SAC_hyperparameters_changed, max_episode_length_1000", "max_episode_length_3000"]

color= ["blue", "black", "red"]
# color= [ "b", "black", "red"]#,[ "m", "m",  "b"]#, "c", "b"]#, "pink"]#"r",, "c", "b"]

# resize=["1"]#"5", 
# init=["-10"]#"-1","-2", "-5","-7", ]#[  "-10", "-100", "-1000"]#"-5","-10","-12", "-15", "-20", , "-500"
# init=["-10","-12", "-15", "-20","-30",  "-50", "-100", "-200"]

# subepoch=["2000", "4000", "16000", "64000"]
std=["0.3", "1.0"]
modified=["0", "1"]
names=["modified rewards", "unmodified rewards"]
num_cols=len(std)
# num_cols=len(subepoch)
# num_cols=len(init)
# num_cols=len(subepoch)
fig=plt.figure(figsize=(30,4))
# for k in range(len(subepoch)):
for k in range(len(std)):
	# for k in range(len(init)):
	for j in range(len(modified)):
		for i in range(len(substrs)):
			for folder in dirs:
				# print(resize[j]+substrs[i]+ init[k],folder)
				if  folder.endswith(substrs[i][0]+std[k]+substrs[i][1]+modified[j]) and os.path.isfile(os.path.join(folder,"progress.csv")):
					# if  folder.endswith(resize[j]+substrs[i]+ init[k]) and os.path.isfile(os.path.join(folder,"progress.csv")):

					# print(folder,resize[j]+substrs[i], init[k] )
					data=pd.read_csv(os.path.join(folder,"progress.csv"))
					# import pdb;pdb.set_trace()
					num_real=data["real_exploration/num steps total"].tolist()
					returns=data['eval_real/Returns Mean'].tolist()
					pathbreak=folder.split("-")
					ax = fig.add_subplot(2,num_cols, k+num_cols+1)
					ax.tick_params(axis='both', which='major', labelsize=5)

					# plt.subplot(2,8, k+9)
					plt.xlabel("num of epoch", fontsize="x-small");
					plt.ylabel("returns",  fontsize="x-small")
					plt.title("sim std:"+ str(std[k]))
					# plt.title("num of subepoch"+ subepoch[k] )
					# plt.title("num of real steps per epoch: " +str(init[k][1:])+" * max episode len",  fontsize="small")
					plt.plot(returns, alpha=0.4, color=color[i], label=names[i])# if substrs[i]=="-batch_rl" else "classifier_offline")
					
					ax = fig.add_subplot(2,num_cols, k+1)
					ax.tick_params(axis='both', which='major', labelsize=5)

					plt.xlabel("num of real steps",  fontsize="x-small");
					plt.ylabel("returns",  fontsize="x-small")
					# plt.title("num of real steps per epoch: " +str(init[k][1:])+" * max episode len",  fontsize="small")

					plt.plot(num_real, returns,alpha=0.4, color=color[i],label=names[i])# if substrs[i]=="-batch_rl" else "classifier_offline" )
					plt.tight_layout()
handles, labels = ax.get_legend_handles_labels()
# import pdb;pdb.set_trace()

set_labels=[]
set_handes=[]
for i in range(len(labels)):
	if labels[i] not in set_labels:
		set_labels.append(labels[i])
		set_handes.append(handles[i])

# fig.legend(handles, labels, loc='bottom right', fontsize="small")
# 
fig.legend(set_handes, set_labels, loc='best')
fig.suptitle("Lunar Lander, sim mean is 0.6 away from real, real var fixed at 0.3")
	# handles, labels = pyplot.gca().get_legend_handles_labels()
	# newLabels, newHandles = [], []
	# for handle, label in zip(handles, labels):
	#   if label not in newLabels:
	#     newLabels.appe	nd(label)
 #    	newHandles.append(handle)
	# pyplot.legend(newHandles, newLabels)
	# plt.legend( loc="lower right", fontsize="x-small")
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



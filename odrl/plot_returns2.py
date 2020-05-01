import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os

items=[

[
("sim_exploration/Returns Mean" , "sim_exploration/Returns Std"), 
("real_exploration/Returns Mean" , "real_exploration/Returns Std"), 
("eval_sim/Returns Mean" , "eval_sim/Returns Std"),
("eval_real/Returns Mean" , "eval_real/Returns Std"),], 


[
("sim_exploration/deltaR_1_rollout Mean" , "sim_exploration/deltaR_1_rollout Std"),
("real_exploration/deltaR_1_rollout Mean" , "real_exploration/deltaR_1_rollout Std"),
("eval_sim/deltaR_1_rollout Mean" , "eval_sim/deltaR_1_rollout Std"),
("eval_real/deltaR_1_rollout Mean" , "eval_real/deltaR_1_rollout Std"),], 


[
("sim_exploration/num steps total",), 
("real_exploration/num steps total",),],
# [("eval_sim/num steps total",), 
# ("eval_real/num steps total",),  ],  


[("classifierSAS__classifier_train_acc_ensamble_num_0",), 
("classifierSAS__classifier_train_loss_ensamble_num_0",)], 

[("trainer/QF1 Loss",),
 ("trainer/QF2 Loss",)], 

[("trainer/Policy Loss",)]

]
plot_folder="/home/swapnil/DRL/offDyna/data/Apr_29th/"
# mean0.6-nameLunar_lander_delta_fixed_-nctspi10-num_trains_per_train_loop2000-real_episodes_per_epoch1-resize_factor1-rl_on_real0-seed0-sim_episodes_per_epoch5-std1.0
# lmda1-max_real_ep200-mean0.6-nameunnamed-nctspi5-real_epi_p_ep1-real_freq2-resize1-rl_on_real0-seed1-sim_epi_p_ep5-std1-sub_ep2000
dirs=os.listdir(plot_folder)
dirs= [os.path.join(plot_folder, dir_) for dir_ in dirs]
# -mean0.6-nameLunar_lander-nctspi10-real_epi_p_ep1-real_freq2-resize1-rl_on_real0-seed0-sim_epi_p_ep5-std0.3-sub_ep2000
# 'lmda0.05-max_real_ep200-mean0.6-nameLunar_lander-nctspi10-real_epi_p_ep1-real_freq2-resize1-rl_on_real0-seed0-sim_epi_p_ep5-std0.3-sub_ep2000
# 'lmda0.05-max_real_ep200-mean0.6-nameLunar_lander-nctspi10-real_epi_p_ep1-real_freq4-resize1-rl_on_real0-seed1-sim_epi_p_ep5-std0.3-sub_ep2000
substrs=[("lmda0.05-max_real_ep200-mean0.6-nameLunar_lander-nctspi10-real_epi_p_ep1-real_freq", "-resize1-rl_on_real0-seed", "-sim_epi_p_ep5-std0.3-sub_ep2000")]
color= ["orange", "green", "blue", "red", ]
real_freqs=['2', '4']
# max_real_eps=[ '50', '100', '200'] #'10000',
# std=["0.3"]#, "1.0"]
seeds=["0", "1", "2"]
# modified=["0", "1"]
num_rows=len(seeds)
num_cols=len(items)
# lamdas=['0.05']#"0.01", "0.03", "0.06", "0.08", "0.12", "0.15", "0.0","0.1", "0.05", "0.5","0.75", "1.0","10.0" ]
# for lamda in lamdas:
# for max_real_ep in max_real_eps:
for real_freq in real_freqs:
	fig=plt.figure(figsize=(26, 5))
	for i in range(len(substrs)):
		for folder in dirs:
			for k in range(len(seeds)):
				# import pdb;pdb.set_trace()
				if  folder.endswith(substrs[i][0]+real_freq+substrs[i][1]+seeds[k]+substrs[i][2] ) and os.path.isfile(os.path.join(folder,"progress.csv")):
					data=pd.read_csv(os.path.join(folder,"progress.csv"))
					epoch=data['Epoch']
					
					for j in range(len(items)):
						handles=[]
						plt.subplot(num_rows,len(items), k*num_cols+j+1)
						# import pdb;pdb.set_trace()
						for l in range(len(items[j])):
							# print(j, l,items[j][l] )
							curve=items[j][l]

							# ax.tick_params(axis='both', which='major', labelsize=5)
							plt.xlabel("epoch", fontsize="x-small");
							plt.title("seed:"+ str(seeds[k]) + " lamda 0.05" ,  fontsize="xx-small")
							# print(curve[0])
							# import pdb;pdb.set_trace()
							handles.append( plt.plot(epoch,data[curve[0]], alpha=0.5, color=color[l], label=curve[0][0:min(len(curve[0]),36)], markersize=1)[0].__dict__['_label'])
							if j<2:
								plt.fill_between(epoch,  data[curve[0]] - data[curve[1]], data[curve[0]] + data[curve[1]],color=color[l], alpha=0.2)
							# handles, labels = ax.get_legend_handles_labels()
						plt.legend(handles, loc='best', fontsize='xx-small')
	plt.tight_layout()
	plt.suptitle("LunarLander. rl on sim with deltaR. 5.12k init real expl steps. Sampled 1k additional real exploration steps with " +str( real_freq)+" epochs. Max allowed real samples: 200k")
	
	plt.show()


# mean=data["eval_sim/deltaR_1_rollout Mean"]
# plt.plot(data["eval_sim/deltaR_1_rollout Mean"], '-', color=color[0])
# 	plt.plot(epoch,data["eval_sim/deltaR_1_rollout Std"]

# plt.plot(data["eval_real/deltaR_1_rollout Mean"], '-', color=color[1])
# plt.plot(epoch,data["eval_real/deltaR_1_rollout Std"]
# plt.plot(data["sim_exploration/deltaR_1_rollout Mean"], '-'5 olor=color[2])
# plt.plot(epoch,data["sim_exploration/deltaR_1_rollout Std"]
# plt.plot(data["real_exploration/deltaR_1_rollout Mean"], '-', color=color[3])
# plt.plot(epoch,data["real_exploration/deltaR_1_rollout Std"]


# 				num_real=data["real_exploration/num steps total"].tolist()
# 				returns=data['eval_real/Returns Mean'].tolist()
# 				pathbreak=folder.split("-")
# 				ax = fig.add_subplot(2,num_cols, k+num_cols+1)
# 				ax.tick_params(axis='both', which='major', labelsize=5)
# 				plt.xlabel("num of epoch", fontsize="x-small");
# 				plt.ylabel("returns",  fontsize="x-small")
# 				plt.title("sim std:"+ str(std[k]))
# 				plt.plot(returns, alpha=0.4, color=color[i], label=names[i])
# 				ax = fig.add_subplot(2,num_cols, k+1)
# 				ax.tick_params(axis='both', which='major', labelsize=5)
# 				plt.xlabel("num of real steps",  fontsize="x-small");
# 				plt.ylabel("returns",  fontsize="x-small")
# 				plt.plot(num_real, returns,alpha=0.4, color=color[i],label=names[i])# if substrs[i]=="-batch_rl" else "classifier_offline" )
# 				plt.tight_layout()
# handles, labels = ax.get_legend_handles_labels()
# # import pdb;pdb.set_trace()

# set_labels=[]
# set_handes=[]
# for i in range(len(labels)):
# 	if labels[i] not in set_labels:
# 		set_labels.append(labels[i])
# 		set_handes.append(handles[i])

# # fig.legend(handles, labels, loc='bottom right', fontsize="small")
# # 
# fig.legend(set_handes, set_labels, loc='best')
# fig.suptitle("Lunar Lander, sim mean is 0.6 away from real, real var fixed at 0.3")
# plt.show() 

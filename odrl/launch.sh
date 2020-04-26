name="Lunar_lander_delta_fixed_"
for seed in 0 1 2
do
for resize_factor in 1
do
for init_episodes in 1
do
for nctpe in 10
do
for mean in 0.6
do
for std in 1. 0.3
do
for subepoch in 2000
do
for rl_on_real in 0
do
for lamda in  0.05 #0.05 #0.03 0.06 0.08 0.12 0.15 #0.1 0.05 0.5 0.75 10 #0 1 #.1 #0 0.1 #0.5 1 5 10 
do
EXPT="${name}_rl_on_real${rl_on_real}_${seed}-${resize_factor}-${init_episodes}-${nctpe}-${mean}-${std}-${subepoch}-${lamda}"
screen -dm -S $EXPT srun -t 20:00:00 --pty /bin/bash
screen -S $EXPT -p 0 -X stuff "module load  python/anaconda3.5-4.2.0; conda env create -f environment/linux-cpu-env.yml; source activate rlkit; module load python/anaconda3.5-4.2.0; pip install --user gym; pip install --user box2d-py==2.3.5; PYTHONPATH='/ihome/pmunro/swa12/rlkit/rlkit-master:/ihome/crc/install/python/anaconda3.5-4.2.0';pip install --user gtimer; python odrl/odrl.py --seed ${seed} --resize_factor ${resize_factor} -n ${name} -i ${init_episodes} -c ${nctpe} -t ${subepoch} -m ${mean} -l ${rl_on_real} --lamda ${lamda} -d ${std}`echo -ne '\015'`"
done
done
done
done
done
done
done
done
done
     

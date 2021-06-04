# Centralized Model and Exploration for Multi-Agent RL

This repository is the official implementation of Centralized Model and Exploration for Multi-Agent RL. 
The implementation is based on https://github.com/oxwhirl/pymarl
## Requirements

To install requirements:

```
cd docker
bash build.sh
cd ..
git clone https://github.com/openai/multiagent-particle-envs.git
cd multiagent-particle-envs
pip install -e .
```
If any difficulties encourtered during building the docker container, consult https://github.com/oxwhirl/pymarl

When building the container, mount to /home, for example
```
singularity shell --nv --writable -H $HOME:/home sandbox_directory
```

## Training Instructions
specify hyper-prameters with commandline or config file '''sc2_xxx''' in '''arx/config/envs'''

### Training for switch tasks

- Training model-free baseline:
```
python3 /home/MARCO/src/main_switch_mf.py --config=vdn --env-config=sc2_switch_mf with n_bridges=3
``` 

- Training MARCO without centralized exploration policy 
```
python3 /home/MARCO/src/main_switch_mb.py --config=vdn --env-config=sc2_switch_mb with n_bridges=3
```

- Training MARCO with centralized exploration policy 
```
python3 /home/MARCO/src/main_switch_explore.py --config=vdn --env-config=sc2_switch_exp with n_bridges=3 central_explore=True beta3=3.0
```

### Training for mpe tasks
- Training model-free baseline:
```
python3 src/main_mpe.py --config=qmix --env-config=sc2_mpe_mf with mb=0 
```

- Training MARCO without centralized exploration policy 
```
python3 src/main_mpe.py --config=qmix --env-config=sc2_mpe_mb with mb=1
```

- Training MARCO with centralized exploration policy 
```
python3 src/main_mpe.py --config=qmix --env-config=sc2_mpe_mb with mb=2
```

## Results 
- See results and saved models, see folder created during training. Results path is specificed by ```local_results_path```

# Centralized Model and Exploration for Multi-Agent RL

This repository is the official implementation of Centralized Model and Exploration for Multi-Agent RL. 

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

When building the container, mount to /home, for example
```
singularity shell --nv --writable -H $HOME:/home sandbox_directory
```

## Training for switch tasks
- use n_bridges=0, switch task does not have bridge.
Training model-freee baseline:
```
python3 /home/MARCO/src/main_switch_mf.py --config=vdn --env-config=sc2_switch_mf with n_bridges=3
``` 
Training MARCO without centralized exploration policy 
```
python3 /home/MARCO/src/main_switch_mb.py --config=vdn --env-config=sc2_switch_mb
```
Training MARCO with centralized exploration policy 
```
python3 /home/MARCO/src/main_switch_explore.py --config=vdn --env-config=sc2_switch_exp with central_explore=True beta3=3.0
```


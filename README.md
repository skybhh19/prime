# PRIME: Scaffolding Manipulation Tasks with Behavior Primitives for Data-Efficient Imitation Learning
<br>

This is the internal codebase for the [**PRIME**](https://ut-austin-rpl.github.io/PRIME/) paper:

**PRIME: Scaffolding Manipulation Tasks with Behavior Primitives for Data-Efficient Imitation Learning**
<br> [Tian Gao](https://skybhh19.github.io/), [Soroush Nasiriany](http://snasiriany.me/), [Huihan Liu](https://https://huihanl.github.io/), [Quantao Yang](https://yquantao.github.io/), [Yuke Zhu](https://www.cs.utexas.edu/~yukez/) 
<br> [UT Austin Robot Perception and Learning Lab](https://rpl.cs.utexas.edu/)
<br> IEEE Robotics and Automation Letters (RA-L), 2024
<br> **[[Paper]](http://arxiv.org/abs/2403.00929)** &nbsp;**[[Project Website]](https://ut-austin-rpl.github.io/PRIME/)** 

<a href="https://ut-austin-rpl.github.io/PRIME/" target="_blank"><img src="src/pull_figure.png" width="90%" /></a>

<br>

## Installation
```commandline
git clone 
cd prime
conda env create -f environment.yml
conda activate prime
pip install -e robosuite
pip install -e robomimic
```

## Data collection

### Human demonstration collection

We collect human demonstrations using [Spacemouse](https://ut-austin-rpl.github.io/deoxys-docs/html/tutorials/using_teleoperation_devices.html). 
```commandline
cd robosuite/robosuite
python scripts/collect_human_demonstrations.py --environment CleanUpMediumSmallInitD2 --directory /path/to/your/directory --only-yaw --only-sucess --device spacemouse
python scripts/collect_human_demonstrations.py --environment NutAssemblyRoundSmallInit --directory /path/to/your/directory --only-yaw --only-sucess --device spacemouse
python scripts/collect_human_demonstrations.py --environment PickPlaceMilk --directory /path/to/your/directory --only-yaw --only-sucess --device spacemouse
```

### Data collection for Inverse Dynamics Models (IDMs)
```commandline
python train.py --collect-demo --reformat-rollout-data --data-dir /path/to/your/directory --env NutAssemblyRoundSmallInit --num-trajs 120 --num-primitives 15 --save --num-data-workers 45 --num-others-per-traj 60 --policy-pretrain --verbose
```

## Training Trajectory Parser

### Training IDMs

```commandline
cd robomimic/robomimic
python scripts/train.py --config exps/primitive/NutAssembly/idm/type/seed1.json 
python scripts/train.py --config exps/primitive/NutAssembly/idm/params/seed1.json 
```

### Segmenting human demonstrations
```commandline
python train.py --segment-demos --demo-path /path/to/human_demos --idm-type-model-path=/path/to/idm_type_ckpt --idm-params-model-path=/path/to/idm_params_ckpt --segmented-data-dir /path/to/segmented_trajectories --save-failed-trajs --max-primitive-horizon=200 --playback-segmented-trajs --verbose --num-augmentation-type 50 --num-augmentation-params 100 
```

## Training Policy with primitives

### Pre-training policy
```commandline
python scripts/train.py --config exps/primitive/NutAssembly/policy/pt/params/seed1.json 
```

### Fine-tuning policy
```commandline
cd robomimic/robomimic
python scripts/train.py --config exps/primitive/NutAssembly/policy/ft/type/seed1.json 
python scripts/train.py --config exps/primitive/NutAssembly/policy/ft/params/seed1.json 
``` 

### Policy evaluation
```commandline
python eval.py --policy-type-path /path/to/policy_type_ckpt --policy-params-path /path/to/policy_params_ckpt --env-horizon 1000
```

## Citation
```bibtex
@article{gao2024prime,
  title={PRIME: Scaffolding Manipulation Tasks with Behavior Primitives for Data-Efficient Imitation Learning},
  author={Tian Gao and Soroush Nasiriany and Huihan Liu and Quantao Yang and Yuke Zhu},
  journal={IEEE Robotics and Automation Letters (RA-L)},
  year={2024}
}
```


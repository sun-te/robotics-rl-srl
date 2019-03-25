#!/bin/bash

BS=32  # Batch size for training SRL model
ENV='KukaButtonGymEnv-v0'
DATASET_NAME='dream_kuka_relative_discrete'
N_ITER=3  # Number of random seeds for training RL
N_CPU=16  # Number of cpu for training PPO
N_EPISODES=1000  # For generating data
N_SRL_SAMPLES=30000
N_TIMESTEPS=3000000
ENV_DIM=5  # Only for priors, state dimension
N_EPOCHS=20  # NUM_EPOCHS for training SRL model

rm -rf logs/dream
rm -rf srl_zoo/data/$DATASET_NAME*
mkdir logs/dream


if [ "$1" != "full" ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "RUNNING DRY RUN MODE, PLEASE USE './run_exp.sh full' FOR NORMAL RUN (waiting 5s)"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo
    sleep 5s
    N_EPISODES=$N_CPU
    N_SRL_SAMPLES=2000
    N_TIMESTEPS=5000
    N_EPOCHS=2
    N_ITER=1
fi


#########
# DATASET
#########

python -m environments.dataset_generator --num-cpu $N_CPU --name $DATASET_NAME --num-episode $N_EPISODES --random-target --env $ENV --action-repeat 8
if [ $? != 0 ]; then
    printf "Error when creating dataset, halting.\n"
	exit $ERROR_CODE
fi



###########################
# GROUND_TRUTH & RAW_PIXELS
###########################

python -m rl_baselines.pipeline --algo ppo2 --random-target --srl-model ground_truth raw_pixels --num-timesteps $N_TIMESTEPS --env $ENV --num-iteration $N_ITER --num-cpu $N_CPU --no-vis --log-dir logs/dream/
if [ $? != 0 ]; then
    printf "Error when training RL ground_truth & raw_pixels model, halting.\n"
	exit $ERROR_CODE
fi


##################
## AUTOENCODERS
##################

#########
##### AE
#########
pushd srl_zoo
python train.py --data-folder data/$DATASET_NAME/  -bs $BS --epochs $N_EPOCHS --state-dim 200 --training-set-size $N_SRL_SAMPLES --losses autoencoder --no-display-plots
if [ $? != 0 ]; then
    printf "Error when training SRL autoencoder model, halting.\n"
        exit $ERROR_CODE
fi
popd
for i in `seq 1 $N_ITER`; do
    python -m rl_baselines.train --algo ppo2 --random-target --latest --srl-model autoencoder --num-timesteps $N_TIMESTEPS --env $ENV --num-cpu $N_CPU --no-vis --log-dir logs/dream/ --seed $i
    if [ $? != 0 ]; then
        printf "Error when training RL SRL autoencoder model, halting.\n"
        exit $ERROR_CODE
    fi
done


##################
## SRL SPLITS
##################

#########
##### AE
#########
pushd srl_zoo
python train.py --data-folder data/$DATASET_NAME/  -bs $BS --epochs $N_EPOCHS --state-dim 200 --training-set-size $N_SRL_SAMPLES --losses autoencoder:1:197 reward:2:-1 inverse:1:3 --no-display-plots
if [ $? != 0 ]; then
    printf "Error when training SRL autoencoder model, halting.\n"
        exit $ERROR_CODE
fi
popd
for i in `seq 1 $N_ITER`; do
    python -m rl_baselines.train --algo ppo2 --random-target --latest --srl-model srl_splits --num-timesteps $N_TIMESTEPS --env $ENV --num-cpu $N_CPU --no-vis --log-dir logs/dream/ --seed $i
    if [ $? != 0 ]; then
        printf "Error when training RL SRL autoencoder model, halting.\n"
        exit $ERROR_CODE
    fi
done

############
# SUPERVISED
############

pushd srl_zoo
python -m srl_baselines.supervised --data-folder data/$DATASET_NAME/ --epochs $N_EPOCHS --training-set-size $N_SRL_SAMPLES --relative-pos --no-display-plots
if [ $? != 0 ]; then
    printf "Error when training SRL supervised model, halting.\n"
	exit $ERROR_CODE
fi
popd

for i in `seq 1 $N_ITER`; do
    python -m rl_baselines.train --algo ppo2 --random-target --srl-model supervised --num-timesteps $N_TIMESTEPS --env $ENV --num-cpu $N_CPU --no-vis --log-dir logs/dream/ --seed $i
    if [ $? != 0 ]; then
        printf "Error when training RL supervised model, halting.\n"
        exit $ERROR_CODE
    fi
done
import os
import torch
import wandb
import argparse
import numpy as np
import random
from arguments import get_args
from EvoRainbow_sac_agent import sac_agent
from utils import env_wrapper
import utils

# Set CPU threads
cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

def parse_args():
    parser = argparse.ArgumentParser(description="Script for running EvoRainbow with SAC")
    parser.add_argument('--wandb_api_key', type=str, required=True, 
                        help='Your Weights & Biases API key')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command-line arguments
    cli_args = parse_args()
    os.environ["WANDB_API_KEY"] = cli_args.wandb_api_key
    
    # Initialize other arguments
    args = get_args()

    # Set up experiment name
    name = (
        f"EvoRainbow_k_{args.K}_{args.EA_tau}_CEM_{args.damp}_{args.damp_limit}_"
        f"SAC_Env_{args.H}_{args.theta}_{args.pop_size}_{args.policy_representation_dim}_"
        f"{args.batch_size}_{args.env_name}_steps_{args.total_timesteps}"
    )
    
    # Initialize Weights & Biases
    our_wandb = wandb.init(project="MetaWorld-v2", name=name)

    # Build the environments
    env = utils.make_metaworld_env(args, args.seed)
    env = env_wrapper(env, args)
    eval_env = utils.make_metaworld_env(args, args.seed + 100)
    eval_env = env_wrapper(eval_env, args)

    # Set seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Create the agent and start training
    sac_trainer = sac_agent(env, eval_env, args, our_wandb)
    sac_trainer.learn()

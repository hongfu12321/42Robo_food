import sys
sys.path.append('..')

from src.game import Game as PizzaGame
from trainer.env_manager import AIPlayer

import tensorflow as tf
from collections import deque
import numpy as np
import argparse
import os



def gen_random_board(rows, cols):
    selection = ['M', 'T']
    return [''.join(np.random.choice(selection) for i in range(cols)) for j in range(rows)]

def main(args):
    
    ROWS = args.rows
    COLS = args.columns
    board = gen_random_board(ROWS, COLS)
    OBSERVATION_DIM = ROWS * COLS * 3 + 3
    ACTIONS = ['up', 'down', 'left', 'right', 'toggle']

    STATE_FEATURES = [
        'ingredients_map',
        'slices_map',
        'cursor_position',
        'slice_mode',
        'min_each_ingredient_per_slice',
        'max_ingredients_per_slice'
    ]

    policy_settings = {
        'n_actions': len(ACTIONS),
        'n_features': OBSERVATION_DIM,
        'learning_rate': args.learning_rate,
        'reward_decay': args.gamma,
        'output_graph': args.output_graph,
        'saving_model': args.saving_model,
        'policy_name': args.model_name,
        'output_dir': args.output_dir,
        'job_dir': args.job_dir
    }

    env_settings = {
        'actions': ACTIONS,
        'input_size': OBSERVATION_DIM,
        'render': args.render,
        'state_features': STATE_FEATURES,
    }

    pizza_config = {
        'pizza_lines': board,
        'r': ROWS,
        'c': COLS,
        'l': args.min_ingred,
        'h': args.max_ingred
    }

    ai = AIPlayer(PizzaGame, env_settings, pizza_config, policy_settings)
    if args.restore:
        try:
            ai.policy.restore_model(args.model_name)
        except:
            print('No model found: ', args.model_name)
            exit()
    for epc in range(args.n_epoch):
        for eps in range(args.n_episodes):
            print(f'Epoch: {epc} Game: {eps}')
            reward, score = ai.play_game(args.max_steps, epc)
            episode_score = reward + score
        ai.policy.learn(epc)
        #ai.policy.add_metrics(str(epc))
        ai.policy.clear_rollout()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('pizza_ai trainer')
    parser.add_argument(
        '--rows',
        type=int,
        default=12)
    parser.add_argument(
        '--columns',
        type=int,
        default=12)
    parser.add_argument(
        '--min-ingred',
        type=int,
        default=1)
    parser.add_argument(
        '--max-ingred',
        type=int,
        default=6)
    parser.add_argument(
        '--n-epoch',
        type=int,
        default=6000)
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=200)
    parser.add_argument(
        '--max-steps',
        type=int,
        default=1000)
    parser.add_argument(
        '--restore',
        default=False,
        action='store_true')
    parser.add_argument(
        '--output-graph',
        default=True,
        action='store_true')
    parser.add_argument(
        '--job-dir',
        type=str,
        default='/tmp/pizza_ai_output')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/tmp/pizza_ai_output')
    parser.add_argument(
        '--render',
        default=False,
        action='store_true')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001)
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.95)
    parser.add_argument(
        '--saving-model',
        default=False,
        action='store_true')
    parser.add_argument(
        '--model-name',
        type=str,
        default='default')
    
    args = parser.parse_args()
    
    main(args)
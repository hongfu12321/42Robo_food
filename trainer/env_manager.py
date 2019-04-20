
import subprocess as sp
import sys
import select
sys.path.append('..')
import tensorflow as tf
import numpy as np
from trainer.policy_gradient import PolicyGradient
from src.game import Game
import time


class MapReverseMap(object):
    def __init__(self):
        self.id = 1
        self.element_map = {}
        self.reverse_map = {}

    def getIdFor(self, obj):
        obj_id = self.element_map.get(obj)
        if obj_id is None:
            obj_id = self.id
            self.element_map[obj] = obj_id
            self.reverse_map[obj_id] = obj
            self.id += 1
        return obj_id

    def getObj(self, id):
        return self.reverse_map.get(id)

class GameViewer:
    def __init__(self, render_fn, fps, rendering=False):
        self.render_fn = render_fn
        self.fps = fps
        self.rendering=rendering

    def render(self):
        time.sleep(self.fps)
        sp.call('clear',shell=True)
        self.render_fn()

class AIPlayer():

    def __init__(self, Game_class, env_settings, init_config, policy_settings):
        self.init_config = init_config
        self.Game = Game_class
        self.current_game = None
        self.game_memory = {}
        self.prev_cursor_pos = (0, 0)
        self.actions = env_settings['actions']
        self.rendering = env_settings['render']
        self.policy = PolicyGradient(
            n_actions=policy_settings['n_actions'],
            n_features=policy_settings['n_features'],
            learning_rate=policy_settings['learning_rate'],
            reward_decay=policy_settings['reward_decay'],
            output_graph=policy_settings['output_graph'],
            saving_model=policy_settings['saving_model'],
            model_name=policy_settings['policy_name'],
            output_dir=policy_settings['output_dir'],
            job_dir=policy_settings['job_dir'])
        self.uniqueStateAccessor = {state_feature: MapReverseMap() for state_feature in env_settings['state_features']}

    def reset(self,max_steps):
        self.current_game = self.Game({"max_steps": max_steps})

    def check_cursor_progression(self,cursor):
        reward_bonus = np.sum(np.subtract(self.prev_cursor_pos, cursor))
        return abs(reward_bonus)

    def play_game(self, max_steps, epc):
        rewards = []
        self.reset(max_steps)
        obs, reward, done, info = self.start()
        obs = self.feed_observation(obs)
        
        while not self.current_game.env['done']:
            action = self.policy.choose_action(obs)
            obs, reward, done, info = self.current_game.step(self.actions[action])
            obs = self.feed_observation(obs)
            self.policy.store_transition(obs, action, reward)
            rewards.append(round(reward, 2))
        if self.rendering:
            self.current_game.render()
        
        return sum(rewards), info['score']

    def feed_observation(self, obs):
        board = np.array(obs['ingredients_map']).flatten()
        slices_map = np.array(obs['slices_map']).flatten()
        cursor = np.zeros(len(board), dtype=int)
        pos = obs['cursor_position'][0] + obs['cursor_position'][1]
        cursor[pos] = 1
        slice_mode = 1 if obs['slice_mode'] else 0
        min_ingred = obs['min_each_ingredient_per_slice']
        max_ingred = obs['max_ingredients_per_slice']
        context = np.array([slice_mode, min_ingred, max_ingred])
        layout = np.concatenate((board, slices_map, cursor))
        inputs = np.concatenate((context, layout))

        return inputs

    def start(self):
        return self.current_game.init(self.init_config)


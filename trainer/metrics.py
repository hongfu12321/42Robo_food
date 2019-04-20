from collections import Counter
from matplotlib import pyplot as plt
import numpy as np

class Metrics:
    def __init__(self):
        self.plotter = plt
        self.outdir = './metrics'

    def most_common_actions(self, actions, gameID):
        action_counter = Counter(actions)
        np.save(self.outdir+'/comm_actions/'+gameID, action_counter)
    
    def statistical(self, actions, rewards, states, gameID):
        statistics = {
            'rewards': [],
            'reward_idx': [],
            'reward_distance': [],
            'efficiency': [],
            'avg_reward_distance': 0
        }
        prev_idx = 0
        for idx, (a, r, s) in enumerate(zip(actions, rewards, states)):
            if r > 2:
                statistics['reward_distance'].append(idx - prev_idx)
                statistics['rewards'].append(r)
                statistics['reward_idx'].append(idx)
                prev_idx = idx
        if(len(rewards)):
            statistics['efficiency'] = len(statistics['rewards']) / len(rewards)
        if (len(statistics['reward_distance'])):
            statistics['avg_reward_distance'] = sum(statistics['reward_distance']) / len(statistics['reward_distance'])
        np.save(self.outdir+'/statistics/'+gameID, statistics)

    def load_and_print(self, metricID, keys, n_epochs):
        for i in range(n_epochs):
            filename = self.outdir + '/' + metricID + '/' + str(i) + '.npy'
            read_d = np.load(filename).item()
            print(filename, read_d.keys())
            print (metricID +': ', str(i))
            print()

            for key in keys:
                print(key, ' => ', read_d[key])
            print()



action_table = {
    0: 'up',
    1: 'down',
    2: 'left',
    3: 'right',
    4: 'toggle'
}

epoch_maximum = 10
episode_name = '29'

m = Metrics()

a = [1,2,3,2,4,2,2,1,2,21,2,3]
m.most_common_actions(a, "0")

if __name__ == '__main__':
    m.load_and_print('statistics', ['efficiency', 'avg_reward_distance'], 6)
    m.load_and_print('comm_actions',[0, 1, 2, 3, 4], 6)

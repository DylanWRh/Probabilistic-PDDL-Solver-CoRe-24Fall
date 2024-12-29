import sys
sys.path.append('..')

from environment import BlockStackingEnv
from tqdm import tqdm

import random
import numpy as np

def gen_task_basic(n_blocks, n_samples, data_name):
    x_min, x_max = -3, 3
    y_min, y_max = -3, 3
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max+1),
        np.arange(y_min, y_max+1),
    )
    x = xx.flatten()
    y = yy.flatten()
    grid = np.stack([x, y], axis=1)
    n = len(grid)
    
    env = BlockStackingEnv(n_blocks)
    all_tasks = []
    for _ in tqdm(range(n_samples)):
        task = {'init':{}, 'target':{}}
        
        for stage in ['init', 'target']:
            list_state = [[i] for i in range(1, n_blocks+1)]
            num_step = random.randint(1, 100)
            for t in range(num_step):
                i, j = np.random.choice(n_blocks, 2, replace=False)
                if list_state[i]:
                    obj_i = list_state[i].pop(-1)
                    list_state[j].append(obj_i)
            
            coords = np.random.choice(n, n_blocks, replace=False)
            block_coords = grid[coords]
            env.set_coords(block_coords)
            env.set_list_state(list_state)
            data = env.get_coords_3d()
            label = env.vector_state
            task[stage] = {'data': data, 'label': label}
        all_tasks.append(task)
    np.savez(data_name, n_blocks=n_blocks, tasks=all_tasks)
        

if __name__ == '__main__':
    gen_task_basic(8, 3000, '../data/tasks/8blocks-3000-basic.npz')
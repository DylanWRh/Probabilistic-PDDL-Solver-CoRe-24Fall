import numpy as np
import torch
from tqdm import tqdm
from model.sgn import BlockStackingSGN
from environment import BlockStackingEnv
from data_utils.dataset import BlockStackingTaskDataset
import heapq


class BlockStackingPlanner:
    def __init__(self, env: BlockStackingEnv, init_state, goal_state, model):
        self.env = env
        self.cur_state = init_state
        self.goal_state = goal_state
        self.sgn_model = model
    
    def coord2state_sgn(self, coords):
        return self.sgn_model(coords)
    
    def set_coords(self, coords):
        self.cur_state = self.coord2state_sgn(coords)
    
    def set_coords_goal(self, coords):
        self.goal_state = self.coord2state_sgn(coords)
    
    def set_state(self, state):
        self.cur_state = state
    
    def set_goal(self, goal):
        self.goal_state = goal
    
    def get_all_actions(self):
        actions = []
        for i in range(1, self.env.num_blocks + 1):
            actions.append(('Put on table', i))
        for i in range(1, self.env.num_blocks + 1):
            for j in range(1, self.env.num_blocks + 1):
                if i != j:
                    actions.append(('Put on', i, j))
        return actions
    
    def goal_diff(self, state=None, matching_fn='L2'):
        if state is None:
            state = self.cur_state
        assert matching_fn in ['L2', 'L1', 'cross_entropy'], \
            f"Unknown matching function: {matching_fn}"
        if matching_fn == 'L2':
            return np.linalg.norm(state - self.goal_state)
        if matching_fn == 'L1':
            return np.linalg.norm(state - self.goal_state, ord=1)
        if matching_fn == 'cross_entropy':
            return -np.sum(self.goal_state * np.log(state + 1e-10))
    
    def goal_reached(self, state=None, EPS=0, matching_fn='L2'):
        if state is None:
            state = self.cur_state
        return self.goal_diff(state, matching_fn) <= EPS
    
    def action_applicable(self, action):
        if action[0] == 'Put on table':
            return self.env.put_on_table(action[1], judge_only=True)
        if action[0] == 'Put on':
            return self.env.put_A_on_B(action[1], action[2], judge_only=True)
        assert False, f'Unknown action: {action}'
    
    def run_straight(self, max_iter=100):
        pass
    
    def run(self, max_iter=100, method='straight'):
        
        if method == 'straight':
            return self.run_straight(max_iter)
        else:
            assert False, f'Unknown method: {method}'
        
        
        

if __name__ == '__main__':   
    task_dataset = BlockStackingTaskDataset('./data/tasks/8blocks-3000-basic.npz')
    model = BlockStackingSGN(8, hidden_dim=128, depth=6)
    # model.load_state_dict(torch.load('checkpoints/model_010.pth'))
    planner = BlockStackingPlanner(BlockStackingEnv(8), None, None, model)
    for task in task_dataset:
        print(task)
        planner.set_coords(task['init']['data'])
        planner.set_coords_goal(task['goal']['data'])
        
        break
    
    
    
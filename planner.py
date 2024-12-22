import numpy as np
from tqdm import tqdm
from environment import BlockStackingEnv
import heapq


class BlockStackingPlanner:
    def __init__(self, env: BlockStackingEnv, init_state, goal_state):
        self.env = env
        self.cur_state = init_state
        self.goal_state = goal_state
    
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
        
        # # TODO: Currently only supports binary states with BFS strategy
        # self.cur_state = np.clip(self.cur_state, 0, 1).round()
        # self.goal_state = np.clip(self.goal_state, 0, 1).round()
        
        # queue = [(self.cur_state.copy(), 0, [])]
        # total_steps = 0
        
        # results = None
        
        # for i in tqdm(range(max_iter)): 
        #     if not queue:
        #         break

        #     prev_state, prev_depth, prev_actions = queue.pop(0)
        #     if self.goal_reached(prev_state):
        #         results = (prev_state, prev_depth, prev_actions)
        #         break
            
        #     if total_steps < max_iter:
        #         actions = self.get_all_actions()
                
        #         self.env.set_vector_state(prev_state.copy())
        #         applicable_actions = []
        #         for action in actions:
        #             if self.action_applicable(action):
        #                 applicable_actions.append(action)
                
        #         action_score_pairs = []
        #         for action in applicable_actions:
        #             self.env.set_vector_state(prev_state.copy())
        #             self.env.execute_action(action)
        #             new_state = self.env.vector_state
        #             new_diff = self.goal_diff(new_state)
        #             action_score_pairs.append((action, new_diff))
                
        #         action_score_pairs = sorted(action_score_pairs, key=lambda x: x[1])
        #         for action, _ in action_score_pairs[:bfs_width]:
        #             self.env.set_vector_state(prev_state.copy())
        #             self.env.execute_action(action)
        #             new_state = self.env.vector_state
        #             new_depth = prev_depth + 1
        #             new_actions = prev_actions + [action]
        #             queue.append((new_state.copy(), new_depth, new_actions))
        #             total_steps += 1
                
        # if results:
        #     final_state, depth, actions = results
        #     self.env.set_vector_state(self.cur_state)
        #     for action in actions:
        #         self.env.execute_action(action)
        #     self.cur_state = final_state
            
        #     print(f'Goal reached in {depth} steps')
        #     # print(f'Final state: \n{self.env.image_state}')
        #     # print(f'Goal state: \n{self.goal_state}')
        #     # print(f'Goal reached: {self.goal_reached()}')
        #     # print(f'Goal diff: {self.goal_diff()}')
        #     return actions
        # else:
        #     print('Goal not reached')
        #     return None
        

if __name__ == '__main__':   
    env = BlockStackingEnv(8)
    
    init_image = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [8, 0, 4, 0, 0, 0, 0, 0],
        [1, 2, 3, 5, 6, 7, 0, 0],
    ])
    init_state = env.image2vector(init_image)
    
    mid_image = np.array([  
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 2, 3, 4, 5, 6, 7, 8],
    ])
    mid_state = env.image2vector(mid_image)
    
    target_image = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 8, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0],
        [7, 5, 3, 0, 0, 0, 0, 0],
        [6, 4, 1, 0, 0, 0, 0, 0],
    ])
    target_state = env.image2vector(target_image)
    
    env.set_image_state(init_image)
    planner = BlockStackingPlanner(env, init_state, mid_state)
    actions1 = planner.run()
    planner.set_goal(target_state)
    actions2 = planner.run()
    
    env.set_image_state(init_image)
    for action in actions1 + actions2:
        env.execute_action(action)
        print(action)
        print(env.image_state)
    
    
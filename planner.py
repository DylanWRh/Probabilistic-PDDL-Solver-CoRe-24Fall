import numpy as np
import torch
from tqdm import tqdm
from model.sgn import BlockStackingSGN
from environment import BlockStackingEnv
from data_utils.dataset import BlockStackingTaskDataset
import heapq
import queue


def atom2idx(atom):
    atom_type = atom[0]
    if atom_type.lower() == 'clear':
        block_idx = atom[1] - 1
        return (block_idx, -2)
    elif atom_type.lower() == 'ontable':
        block_idx = atom[1] - 1
        return (block_idx, -1)
    elif atom_type.lower() == 'on':
        block_a, block_b = atom[1] - 1, atom[2] - 1
        return (block_a, block_b)
    else:
        raise NotImplementedError()


def get_atom_prob(state, atom):
    return state[atom2idx(atom)]


def get_applicability(state, action):
    block_a, block_b = action[1], action[2]
    return get_atom_prob(state, ['Clear', block_a]) * get_atom_prob(state, ['Clear', block_b])


def get_effective_set(state, action):
    block_a, block_b = action[1], action[2]
    # removed = [['clear', idx] for idx in range(1, len(state + 1)) if idx != block_a and idx != block_b]
    return (
        [
            ['on', block_a, block_b],
        ],
        [
            ['clear', block_b],
        ]
    )


def apply_action(state, action):
    new_state = state.copy()
    positive_set, negative_set = get_effective_set(state, action)
    applicability = get_applicability(state, action)
    for positive_atom in positive_set:
        atom_idx = atom2idx(positive_atom)
        # if positive_atom[0] == 'clear':
        #     new_state[atom_idx] = np.clip(applicability * get_atom_prob(state, ['on', action[1], positive_atom[1]]) + (1 - applicability) * state[atom_idx], 0, 1)
        # else:
        #     new_state[atom_idx] = np.clip(applicability + (1 - applicability) * state[atom_idx], 0, 1)
        new_state[atom_idx] = np.clip(applicability + (1 - applicability) * state[atom_idx], 0, 1)
    for negative_atom in negative_set:
        atom_idx = atom2idx(negative_atom)
        new_state[atom_idx] = np.clip(state[atom_idx] - applicability, 0, 1)
    return new_state


def list_action(n_blocks):
    res = []
    for i in range(1, n_blocks + 1):
        for j in range(1, n_blocks + 1):
            if i != j:
                res.append(['Put on', i, j])
    return res


def evaluate_state(curr, goal):
    return np.linalg.norm(curr[..., :-1] - goal[..., :-1])


class ContinousPlanner:
    def __init__(self, goal=None, n_blocks=8, max_len=5000, eps=1e-3):
        self.goal = goal
        self.n_blocks = n_blocks
        self.action_list = list_action(n_blocks)
        self.max_len = max_len
        self.eps = eps

    def set_goal_(self, goal):
        self.goal = goal

    def naive_bfs(self, state):
        assert self.goal is not None, "goal mustn't be None for planner"
        valid_action_list = self.action_list
        state_queue = queue.Queue()
        state_queue.put((state, [], evaluate_state(state, self.goal)))
        while state_queue.qsize() < self.max_len:
            curr_state, plan, _ = state_queue.get()
            min_error = 1e10
            for action in valid_action_list:
                next_state = apply_action(curr_state, action)
                error = evaluate_state(next_state, self.goal)
                state_queue.put((next_state, plan + [action], error))
                min_error = min(min_error, error)
            if min_error < self.eps:
                break
        selected = sorted(list(state_queue.queue), key=lambda a: a[-1])[0]
        return selected[1][0]

    def __call__(self, state):
        return self.naive_bfs(state)


CP = ContinousPlanner

class TaskManager:
    def __init__(self, n_blocks, sgn=None, init_vec=None, goal_vec=None):
        self.n_blocks = n_blocks
        self.planner = CP(n_blocks=n_blocks, max_len=50000)
        self.env = BlockStackingEnv(n_blocks)
        self.sgn = sgn
        self.set_task_(init_vec, goal_vec)

    def set_sgn_(self, sgn):
        self.sgn = sgn

    def set_task_(self, init_vec, goal_vec):
        self.init_vec = init_vec
        self.goal_vec = goal_vec
        self.env.set_vector_state(init_vec)
        self.planner.set_goal_(goal_vec)

    def run_planner(self, verbose=False, max_iter=50):
        assert self.goal_vec is not None and self.init_vec is not None, "Set the task before run planner"
        iter_count = 0
        if verbose: print('init:\n', self.env.image_state, 'goal:\n', self.env.vector2image(self.goal_vec))
        while not np.allclose(self.env.vector_state, self.goal_vec):
            iter_count = iter_count + 1
            if iter_count >= max_iter:
                if verbose: print('exceed max iter, task failed')
                return 0
            gt_block_coord = self.env.get_coords_3d()
            with torch.no_grad():
                pred_vector = self.sgn(torch.from_numpy(gt_block_coord.flatten()).to(torch.float)).numpy().reshape(self.n_blocks, -1)
            # pred_vector = pred_vector.round()
            # pred_vector = self.env.vector_state
            if verbose: print('predicted vector state\n', pred_vector.round(2))
            if verbose: print('gt vector state\n', self.env.vector_state)
            action = self.planner(pred_vector)
            if not self.env.execute_action(action):
                if verbose: print('invalid action: {}'.format(action))
                if verbose: print(self.env.image_state)
                return 0
            if verbose: print('executed action {}'.format(action))
            if verbose: print(self.env.image_state)
            if verbose: print(self.env.vector2image(self.goal_vec))
        if verbose: print('task succeed')
        return 1


if __name__ == '__main__':
    init_img = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ]
    )
    goal_img = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [7, 8, 0, 0, 0, 0, 0, 0],
            [4, 5, 6, 0, 0, 0, 0, 0],
            [1, 2, 3, 0, 0, 0, 0, 0],
        ]
    )
    env = BlockStackingEnv(8)
    sgn = BlockStackingSGN(8, 128, 2)
    sgn.load_state_dict(torch.load('./checkpoints/sgn/model_100.pth'))
    manager = TaskManager(8, env.image2vector(init_img), env.image2vector(goal_img), sgn)
    manager.run_planner(verbose=True)

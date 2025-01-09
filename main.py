from data_utils.dataset import BlockStackingTaskDataset, BlockStackingDemonstration
from environment import BlockStackingEnv
from planner import TaskManager, ContinousPlanner
from model.sgn import BlockStackingSGN
import torch
from tqdm.auto import tqdm
from train_sgn import fit_sgn


def run_pretrained(data_path, model_path, n_blocks=8):
    sgn = BlockStackingSGN(n_blocks, 128, 2)
    sgn.load_state_dict(torch.load(model_path))
    task_manager = TaskManager(n_blocks, sgn)
    task_ds = BlockStackingTaskDataset(data_path)
    success_count = 0
    bar = tqdm(task_ds, desc='total_success: 0')
    for task in bar:
        init_state = task['init']['label']
        goal_state = task['goal']['label']
        task_manager.set_task_(init_state, goal_state)
        success_count += task_manager.run_planner(verbose=False)
        bar.set_description('total_success: {}'.format(success_count))
    # init_state = task_ds[0]['init']['label']
    # goal_state = task_ds[0]['goal']['label']
    # task_manager.set_task_(init_state, goal_state)
    # task_manager.run_planner(True)
    print(success_count, len(task_ds))


def run_demonstration(task_path, demostration_path, n_blocks=8, n_demo=5, device='cpu'):
    task_manager = TaskManager(n_blocks)
    task_ds = BlockStackingTaskDataset(task_path)
    success_count = 0
    bar = tqdm(task_ds, desc='total_success: 0')
    for task in bar:
        sgn = BlockStackingSGN(n_blocks, 128, 2)
        demo_ds = BlockStackingDemonstration(demostration_path, n_demo)
        fit_sgn(sgn, demo_ds, 'cpu')
        init_state = task['init']['label']
        goal_state = task['goal']['label']
        task_manager.set_task_(init_state, goal_state)
        task_manager.set_sgn_(sgn)
        success_count += task_manager.run_planner(verbose=False)
        bar.set_description('total_success: {}'.format(success_count))
    # init_state = task_ds[0]['init']['label']
    # goal_state = task_ds[0]['goal']['label']
    # task_manager.set_task_(init_state, goal_state)
    # task_manager.run_planner(True)
    print(success_count, len(task_ds))


if __name__ == '__main__':
    # run_pretrained(
    #     './data/tasks/8blocks-3000-basic-3d.npz',
    #     './checkpoints/sgn-20250109-141728/model_030.pth'
    # )
    run_demonstration(
        './data/tasks/8blocks-3000-basic-3d.npz',
        './data/tasks/8blocks-3000-demonstration-3d.npz'
    )

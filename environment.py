import numpy as np
from typing import Optional, List

class BlockStackingEnv:
    NUM_BLOCKS = 8
    
    def __init__(
        self, 
        num_blocks: int = NUM_BLOCKS, 
        coords: Optional[np.ndarray] = None,
        init_image: Optional[np.ndarray] = None, 
        init_vector: Optional[np.ndarray] = None,
        init_list: Optional[List[List]] = None,
        goal_image: Optional[np.ndarray] = None,
        goal_vector: Optional[np.ndarray] = None,
        goal_list: Optional[List[List]] = None
    ):
        '''
        Args:
            num_blocks: number of blocks in the environment
            
            coords: integer coordinates of the plane positions that can place blocks, shape [N, 2]
            
            image: image representation of the blocks, shape [N+1, N]
            vector: vector representation of the blocks, shape [N, N+2]. For each block, the last two elements are the one-hot encoding of whether the block is clear and on the table, respectively. The rest of the elements are the one-hot encoding of the block on top of it, e.g. if A is on B, then vector[A, B] = 1.
            list: list representation of the blocks
            ..Note: At most one of coords, image, vector, and list can be provided. If none is provided, the initial state is that all blocks are on the table.
        '''
        
        assert (init_image is None) + (init_vector is None) + (init_list is None) >= 2, \
            "At most one of init_image, init_vector, and init_list can be provided."
        assert (goal_image is None) + (goal_vector is None) + (goal_list is None) >= 2, \
            "At most one of goal_image, goal_vector, and goal_list can be provided."
        
        self.num_blocks = num_blocks
        
        self.set_coords(coords)
        
        if init_image is not None:
            self.set_image_state(init_image, is_goal=False)
        elif init_vector is not None:
            self.set_vector_state(init_vector, is_goal=False)
        elif init_list is not None:
            self.set_list_state(init_list, is_goal=False)
        else:
            self.set_default_state(is_goal=False)
        
        if goal_image is not None:
            self.set_image_state(goal_image, is_goal=True)
        elif goal_vector is not None:
            self.set_vector_state(goal_vector, is_goal=True)
        elif goal_list is not None:
            self.set_list_state(goal_list, is_goal=True)
        else:
            self.set_default_state(is_goal=True)
    
    #########################
    # State setting methods #
    #########################
    
    def set_coords(self, coords):
        if coords is None:
            self.coords = np.zeros((self.num_blocks, 2), dtype=int)
            self.coords[:, 0] = np.arange(1, self.num_blocks + 1)
        else:
            assert coords.shape == (self.num_blocks, 2), \
                f"Expected coords to have shape ({self.num_blocks}, 2), got {coords.shape}."
            assert len(np.unique(coords, axis=0)) == len(coords), "Coordinates must be unique."
            self.coords = coords
    
    def set_default_state(self, is_goal=False):
        list_state = [[i] for i in range(1, self.num_blocks + 1)]
        vector_state = self.list2vector(list_state)
        image_state = self.list2image(list_state)
        if is_goal:
            self.goal_list = list_state
            self.goal_image = image_state
            self.goal_vector = vector_state
        else:
            self.list_state = list_state
            self.image_state = image_state
            self.vector_state = vector_state 
    
    def set_list_state(self, list_state, is_goal=False):
        '''
        Args:
            list_state: list representation of the blocks
        '''
        if list_state is None:
            self.set_default_state(is_goal)
            return
        
        list_state = list_state
        vector_state = self.list2vector(list_state)
        image_state = self.list2image(list_state)
        if is_goal:
            self.goal_list = list_state
            self.goal_image = image_state
            self.goal_vector = vector_state
        else:
            self.list_state = list_state
            self.image_state = image_state
            self.vector_state = vector_state  
    
    def set_image_state(self, image_state, is_goal=False):
        '''
        Args:
            image_state: image representation of the blocks, shape [N+1, N]
        '''
        if image_state is None:
            self.set_default_state(is_goal)
            return

        assert image_state.shape == self.image_state.shape, \
            f"Expected image_state to have shape {self.image_state.shape}, " \
            f"got {image_state.shape}."
        image_state = image_state
        list_state = self.image2list(image_state)
        vector_state = self.image2vector(image_state)
        if is_goal:
            self.goal_list = list_state
            self.goal_image = image_state
            self.goal_vector = vector_state
        else:
            self.list_state = list_state
            self.image_state = image_state
            self.vector_state = vector_state
    
    def set_vector_state(self, vector_state=None, is_goal=False):
        '''
        Args:
            vector_state: vector representation of the blocks, shape [N, N+2]
        '''
        if vector_state is None:
            self.set_default_state(is_goal)
            return
        
        assert vector_state.shape == self.vector_state.shape, \
                f"Expected vector_state to have shape {self.vector_state.shape}, " \
                f"got {vector_state.shape}."
        vector_state = vector_state
        image_state = self.vector2image(vector_state)
        list_state = self.vector2list(vector_state)
        if is_goal:
            self.goal_list = list_state
            self.goal_image = image_state
            self.goal_vector = vector_state
        else:
            self.list_state = list_state
            self.image_state = image_state
            self.vector_state = vector_state

    ##########################
    # State transfer methods #
    ##########################

    def list2image(self, list_state: List[List[int]]) -> np.ndarray:
        '''
        Args:
            list_state: list representation of the blocks
        Returns:
            image: image representation of the blocks, shape [N+1, N]
        '''
        N = self.num_blocks
        image = np.zeros((N+1, N), dtype=int)
        for i, blocks in enumerate(list_state):
            image[ :len(blocks), i] = blocks
        image = image[::-1]
        return image
    
    def list2vector(self, list_state: List[List[int]]) -> np.ndarray:
        '''
        Args:
            list_state: list representation of the blocks
        Returns:
            vector: vector representation of the blocks, shape [N, N+2]
        '''
        N = self.num_blocks
        vector = np.zeros((N, N+2), dtype=int)
        for i, blocks in enumerate(list_state):
            for (obj_A, obj_B) in zip(blocks[1:], blocks[:-1]):
                vector[obj_A - 1, obj_B - 1] = 1
            if blocks:
                vector[blocks[-1] - 1, -2] = 1      # clear
                vector[blocks[0] - 1, -1] = 1       # on table
                
        return vector
    
    def image2list(self, image_stae: np.ndarray) -> List[List[int]]:
        '''
        Args:
            image_stae: image representation of the blocks, shape [N+1, N]
        Returns:
            list_state: list representation of the blocks
        '''
        N = self.num_blocks
        list_state = []
        for i in range(N):
            blocks = image_stae[:, i]
            blocks = blocks[blocks != 0]
            list_state.append(list(blocks)[::-1])
        return list_state
    
    def vector2list(self, vector_state: np.ndarray) -> List[List[int]]:
        '''
        Args:
            vector: vector representation of the blocks, shape [N, N+2]
        Returns:
            list_state: list representation of the blocks
        '''
        N = self.num_blocks
        list_state = [[] for _ in range(N)]
        
        block2col = dict()
        used_blocks = set()
        
        col_num = 0
        ground_blocks = np.where(vector_state[:, -1] == 1)[0] + 1
        for block in ground_blocks:
            list_state[col_num].append(block)
            block2col[block] = col_num
            used_blocks.add(block)
            col_num += 1
        
        search_list = list(ground_blocks)
        while search_list:
            block = search_list.pop(0)
            objs_on_block = np.where(vector_state[:, block - 1] == 1)[0] + 1
            objs_on_block = set(objs_on_block) - used_blocks            
            if not objs_on_block:
                continue
            obj_on_block = objs_on_block.pop(-1)
            
            block2col[obj_on_block] = block2col[block]
            list_state[block2col[block]].append(obj_on_block)
            used_blocks.add(obj_on_block)
            
            search_list.append(obj_on_block)
        return list_state
    
    def vector2image(self, vector_state: np.ndarray) -> np.ndarray:
        image_state = self.list2image(self.vector2list(vector_state))
        return image_state
    
    def image2vector(self, image_state: np.ndarray) -> np.ndarray:
        vector_state = self.list2vector(self.image2list(image_state))
        return vector_state
    
    ########################
    # Info getting methods #
    ########################
    
    def __str__(self):
        string = f"BlockStackingEnv with {self.num_blocks} blocks\n"
        string += "Current state:\n"
        string += f"\tCoords:\n{self.coords.tolist()}\n"
        string += f"\tList:\n{self.list_state}\n"
        string += f"\tImage:\n{self.image_state}\n"
        string += f"\tVector:\n{self.vector_state}\n"
        return string
    
    def get_language_state(self, is_goal=False):       
        N = self.num_blocks
        language = f"BlockStackingEnv with {self.num_blocks} blocks"
        if is_goal:
            language += " (Goal state):\n"
        else:
            language += " (Current state):\n"
        
        language += "On Table: ["
        for i in range(1, N+1):
            if self.is_on_table(i, is_goal):
                language += f"{i}, "
        if language[-2:] == ", ":
            language = language[:-2]
        language += "]\n"
        
        language += "Clear: ["
        for i in range(1, N+1):
            if self.is_clear(i, is_goal):
                language += f"{i}, "
        if language[-2:] == ", ":
            language = language[:-2]
        language += "]\n"
        
        language += "On State: ["
        for i in range(1, N+1):
            for j in range(1, N+1):
                if self.is_on(i, j, is_goal):
                    language += f"({i} is on {j}), "
        if language[-2:] == ", ":
            language = language[:-2]
        language += "]\n"
        
        return language
    
    def get_coords_3d(self):
        coords_3d = np.zeros((self.num_blocks, 3), dtype=int)
        for i, blocks in enumerate(self.list_state):
            for j, block in enumerate(blocks):
                coords_3d[block - 1, :2] = self.coords[i]
                coords_3d[block - 1, 2] = j
        return coords_3d
    
    #############################
    # Condition judging methods #
    #############################
    
    def is_clear(self, obj, is_goal=False):
        if is_goal:
            return self.goal_vector[obj - 1, -2] == 1
        return self.vector_state[obj - 1, -2] == 1
    
    def is_on_table(self, obj, is_goal=False):
        if is_goal:
            return self.goal_vector[obj - 1, -1] == 1
        return self.vector_state[obj - 1, -1] == 1
    
    def is_on(self, obj_A, obj_B, is_goal=False):
        ''' Judge whether obj_A is on obj_B
        '''
        if is_goal:
            return self.goal_vector[obj_A - 1, obj_B - 1] == 1
        return self.vector_state[obj_A - 1, obj_B - 1] == 1
    
    ############################
    # Action executing methods #
    ############################
    
    def put_on_table(self, obj_A, judge_only=False):
        preconds = [
            not self.is_on_table(obj_A),
            self.is_clear(obj_A)
        ]
        if not all(preconds):
            return False
        if judge_only:
            return True
        
        list_state = self.list_state
        i1, i2 = -1, -1
        for i, blocks in enumerate(list_state):
            if obj_A in blocks:
                i1 = i
            if i2 == -1 and not blocks:
                i2 = i
        list_state[i1].remove(obj_A)
        list_state[i2].append(obj_A)
        self.set_list_state(list_state)
        return True
    
    def put_A_on_B(self, obj_A, obj_B, judge_only=False):
        ''' A should be clear, B should be clear
        '''
        preconds = [
            self.is_clear(obj_A),
            self.is_clear(obj_B),
            not self.is_on(obj_A, obj_B),
            not obj_A == obj_B
        ]
        if not all(preconds):
            return False
        if judge_only:
            return True
        
        i1, i2 = -1, -1
        list_state = self.list_state
        for i, blocks in enumerate(list_state):
            if obj_A in blocks:
                i1 = i
            if obj_B in blocks:
                i2 = i
        list_state[i1].remove(obj_A)
        list_state[i2].append(obj_A)
        self.set_list_state(list_state)
        return True
    
    def execute_action(self, action):
        if action[0] == 'Put on table':
            return self.put_on_table(action[1])
        elif action[0] == 'Put on':
            return self.put_A_on_B(action[1], action[2])
        else:
            assert False, f'Unknown action: {action}'


if __name__ == '__main__':
    env = BlockStackingEnv()
    
    ### Test Initial state
    print('\n----------------- Test Initial state -----------------')
    print(env)

    ### Test state setting
    print('\n----------------- Test Coord Setting -----------------')
    coords = np.array([
        [1, 1], [1, 2], [2, 1], [4, 3],
        [4, 5], [7, -1], [8, 1], [9, 2]
    ])
    env.set_coords(coords)
    print(env)
    
    print('\n----------------- Test Non-Unique Coord Setting -----------------')
    try:
        coords = np.array([
            [1, 1], [1, 2], [2, 1], [1, 1],
            [4, 5], [7, -1], [8, 1], [9, 2]
        ])
        env.set_coords(coords)
    except AssertionError as e:
        print('\n----------------- Non-Unique Coord Test Passed -----------------')
    
    print('\n----------------- Test Image Setting -----------------')
    image = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 5, 0, 0, 0, 0, 0],
                      [0, 6, 8, 0, 0, 0, 0, 0],
                      [1, 2, 3, 4, 0, 0, 7, 0]] )
    env.set_image_state(image)
    print(env)
    
    print('\n----------------- Test Vector Setting -----------------')
    vector = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    ])
    env.set_vector_state(vector)  
    print(env)
    
    print('\n----------------- Test List Setting -----------------')
    list_state  = [
        [1, 2, 5],
        [7, 6],
        [],
        [4],
        [8],
        [],
        [],
        []
    ]
    env.set_list_state(list_state)
    print(env)
    
    ### Test language state translation
    print('\n----------------- Test Language State -----------------')
    print(env.get_language_state())
    print(env.get_language_state(is_goal=True))
        
    ### Test move on table action
    print('\n----------------- Test Move on table action False case -----------------')
    print(f"Try to move 1 on table: {env.put_on_table(1)}")
    print(f"Try to move 2 on table: {env.put_on_table(2)}")
    
    print('\n----------------- Test Move on table action True case -----------------')
    print(f"Try to move 5 on table: {env.put_on_table(5)}")
    print(env)
    print(env.get_language_state())
    
    ### Test put on action 
    print('\n----------------- Test Put on action False case -----------------')
    print(f"Try to move 1 on 2: {env.put_A_on_B(1, 2)}")
    print(f"Try to move 7 on 8: {env.put_A_on_B(7, 8)}")
    
    print('\n----------------- Test Put on action True case 1 -----------------')
    print(f"Try to move 8 on 6: {env.put_A_on_B(8, 6)}")
    print(env)
    print(env.get_language_state())
    
    print('\n----------------- Test Put on action True case 2 -----------------')
    print(f"Try to move 2 on 8: {env.put_A_on_B(2, 8)}")
    print(env)
    print(env.get_language_state())
    
    ### Test execute action
    print('\n----------------- Test Execute action Unknown case -----------------')
    try:
        print(f"Try to execute action ['Unknown', 1, 2]: {env.execute_action(['Unknown', 1, 2])}")
    except AssertionError as e:
        print('\n----------------- Unknown action Test Passed -----------------')
    
    print('\n----------------- Test Execute action False case -----------------')
    print(f"Try to execute action ['Put on table', 7]: {env.execute_action(['Put on table', 7])}")
    print(f"Try to execute action ['Put on table', 4]: {env.execute_action(['Put on table', 4])}")
    print(f"Try to execute action ['Put on', 8, 1]: {env.execute_action(['Put on', 8, 1])}")
    print(f"Try to execute action ['Put on', 1, 1]: {env.execute_action(['Put on', 1, 1])}")
    print(f"Try to execute action ['Put on', 2, 8]: {env.execute_action(['Put on', 2, 8])}")
    
    print('\n----------------- Test Execute action True case 1 -----------------')
    print(f"Try to execute action ['Put on table', 2]: {env.execute_action(['Put on table', 2])}")
    print(env)
    print(env.get_language_state())
    
    print('\n----------------- Test Execute action True case 2 -----------------')
    print(f"Try to execute action ['Put on', 4, 5]: {env.execute_action(['Put on', 4, 5])}")
    print(env)
    print(env.get_language_state())
    
    print('\n----------------- Test Execute action True case 3 -----------------')
    print(f"Try to execute action ['Put on', 8, 2]: {env.execute_action(['Put on', 8, 2])}")
    print(env)
    print(env.get_language_state())
    
    print('\n----------------- Test 3D Coordinates -----------------')
    print(env)
    print(env.get_coords_3d())
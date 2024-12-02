import numpy as np
from typing import Optional

class BlockStackingEnv:
    NUM_BLOCKS = 8
    
    def __init__(
        self, 
        num_blocks: int = NUM_BLOCKS, 
        init_coords: Optional[np.ndarray] = None,
        init_image: Optional[np.ndarray] = None, 
        init_vector: Optional[np.ndarray] = None,
        goal_coords: Optional[np.ndarray] = None,
        goal_image: Optional[np.ndarray] = None,
        goal_vector: Optional[np.ndarray] = None
    ):
        '''
        Args:
            num_blocks: number of blocks in the environment
            init_coords: initial integer coordinates of the blocks, shape [N, 3]. Table plane is at z = 0.
            init_matrix: initial image of the blocks, shape [N, N+1]
            init_vector: initial vector representation of the blocks, shape [N, N+2]. For each block, the last two elements are the one-hot encoding of whether the block is clear and on the table, respectively. The rest of the elements are the one-hot encoding of the block on top of it.
            ..Note: At most one of init_coords, init_image, and init_vector can be provided. If none is provided, the initial state is that all blocks are on the table.
            goal_coords: goal integer coordinates of the blocks, shape [N, 3]. Table plane is at z = 0.
            goal_image: goal image of the blocks, shape [N, N+1]
            goal_vector: goal vector representation of the blocks, shape [N, N+2]. For each block, the last two elements are the one-hot encoding of whether the block is clear and on the table, respectively. The rest of the elements are the one-hot encoding of the block on top of it.
            ..Note: At most one of goal_coords, goal_image, and goal_vector can be provided. If none is provided, the goal state is that all blocks are on the table.
        '''
        
        assert (init_coords is None) + (init_image is None) + (init_vector is None) >= 2, \
            "At most one of init_coords, init_image, and init_vector can be provided."
        assert (goal_coords is None) + (goal_image is None) + (goal_vector is None) >= 2, \
            "At most one of goal_coords, goal_image, and goal_vector can be provided."
        
        self.num_blocks = num_blocks
        
        if init_coords is not None:
            self.set_coords_state(init_coords, is_goal=False)
        elif init_image is not None:
            self.set_image_state(init_image, is_goal=False)
        elif init_vector is not None:
            self.set_vector_state(init_vector, is_goal=False)
        else:
            self.set_default_state(is_goal=False)
        
        if goal_coords is not None:
            self.set_coords_state(goal_coords, is_goal=True)
        elif goal_image is not None:
            self.set_image_state(goal_image, is_goal=True)
        elif goal_vector is not None:
            self.set_vector_state(goal_vector, is_goal=True)
        else:
            self.set_default_state(is_goal=True)
    
    
    def set_default_state(self, is_goal=False):
        vector = np.zeros((self.num_blocks, self.num_blocks + 2), dtype=int)
        vector[:, -1] = 1     # all blocks are on the table
        vector[:, -2] = 1     # all blocks are clear
        image = self.vector_to_image(vector)
        if is_goal:
            self.goal_image = image
            self.goal_vector = vector
        else:
            self.image_state = image
            self.vector_state = vector
    
    def __str__(self):
        string = f"BlockStackingEnv with {self.num_blocks} blocks\n"
        string += "Current state:\n"
        string += f"\tImage:\n{self.image_state}\n"
        string += f"\tVector:\n{self.vector_state}\n"
        return string
    
    def set_image_state(self, image_state, is_goal=False):
        '''
        Args:
            image_state: image representation of the blocks, shape [N, N+1]
        '''
        if image_state is not None:
            assert image_state.shape == self.image_state.shape, \
                f"Expected image_state to have shape {self.image_state.shape}, " \
                f"got {image_state.shape}."
            image_state = image_state
            vector_state = self.image_to_vector(image_state)
        else:
            self.set_default_state(is_goal)
            return
        
        if is_goal:
            self.goal_image = image_state
            self.goal_vector = vector_state
        else:
            self.image_state = image_state
            self.vector_state = vector_state
    
    def set_vector_state(self, vector_state=None, is_goal=False):
        '''
        Args:
            vector_state: vector representation of the blocks, shape [N, N+2]
        '''
        if vector_state is not None:
            assert vector_state.shape == self.vector_state.shape, \
                f"Expected vector_state to have shape {self.vector_state.shape}, " \
                f"got {vector_state.shape}."
            vector_state = vector_state
            image_state = self.vector_to_image(vector_state)
        else:
            self.set_default_state(is_goal)
            return
            
        if is_goal:
            self.goal_image = image_state
            self.goal_vector = vector_state
        else:
            self.image_state = image_state
            self.vector_state = vector_state
    
    def set_coords_state(self, coords_state, is_goal=False):
        '''
        Args:
            coords_state: integer coordinates of the blocks, shape [N, 3]
        '''
        if coords_state is not None:
            assert coords_state.shape == (self.NUM_BLOCKS, 3), \
                f"Expected coords_state to have shape " \
                f"({self.NUM_BLOCKS}, 3), got {coords_state.shape}."
            image_state = self.coords_to_image(coords_state)
            vector_state = self.coords_to_vector(coords_state)
        else:
            self.set_default_state(is_goal)
            return
        
        if is_goal:
            self.goal_image = image_state
            self.goal_vector = vector_state
        else:
            self.image_state = image_state
            self.vector_state = vector_state
    
    
    def coords_to_image(self, coords: np.ndarray) -> np.ndarray:
        '''
        Args:
            coords: integer coordinates of the blocks, shape [N, 3]
        Returns:
            image: image representation of the blocks, shape [N, N+1]
        '''
        
        N = self.num_blocks
        
        # Step 1. Sort the blocks by z-coordinate
        coords = coords.astype(int)
        coords[coords[:, 2] < 0] = 0    # set blocks below the table to the table
        sorted_idx = coords[:, 2].argsort()
        coords = coords[sorted_idx]
        coords_unique = np.unique(coords, axis=0)
        assert len(coords_unique) == N, \
            f"Expected {N} unique coordinates, got {len(coords_unique)}."
        
        # Step 2. Create the image
        image = np.zeros((N, N + 1), dtype=int)
        num_cols = 0
        xy2col = {}
        col2height = {}
        
        for i, (x, y, _) in zip(sorted_idx, coords):
            if (x, y) not in xy2col:
                xy2col[(x, y)] = num_cols
                col2height[num_cols] = 0
                num_cols += 1
            col = xy2col[(x, y)]
            height = col2height[col]
            image[height, col] = i + 1
            col2height[col] += 1

        image = image[::-1]
        return image
    
    def coords_to_vector(self, coords: np.ndarray) -> np.ndarray:
        '''
        Args:
            coords: integer coordinates of the blocks, shape [N, 3]
        Returns:
            vector: vector representation of the blocks, shape [N, N+2]
        '''
        return self.image_to_vector(self.coords_to_image(coords))
    
    def image_to_vector(self, image: np.ndarray) -> np.ndarray:
        '''
        Args:
            image: image representation of the blocks, shape [N, N+1]
        Returns:
            vector: vector representation of the blocks, shape [N, N+2]
        '''
        
        N = self.num_blocks
        image = image[::-1].T
        vector = np.zeros((N, N+2), dtype=int)

        for i in range(N):
            for j in range(N):
                if image[i, j] != 0:
                    cur_block = image[i, j] - 1
                    if j == 0:
                        vector[cur_block, -1] = 1
                    next_block = image[i, j+1] - 1
                    if next_block == -1:
                        vector[cur_block, -2] = 1
                    else:
                        vector[cur_block, next_block] = 1
        return vector
    
    def vector_to_image(self, vector: np.ndarray) -> np.ndarray:
        '''
        Args:
            vector: vector representation of the blocks, shape [N, N+2]
        Returns:
            image: image representation of the blocks, shape [N, N+1]
        '''
        N = self.num_blocks
        image = np.zeros((N, N+1), dtype=int)
        
        num_cols = 0
        vector_graph = vector[:N, :N]
        relations = np.where(vector_graph == 1)

        columns = [[i] for i in range(N)]
        idx2col = {i: i for i in range(N)}

        for down, up in zip(*relations):
            upcols = columns[idx2col[up]]
            columns[idx2col[up]] = []
            idx2col[up] = idx2col[down]
            
            ii = idx2col[up]
            while idx2col[ii] != ii:
                ii = idx2col[ii]
            columns[idx2col[ii]].extend(upcols)

        for col in columns:
            if col:
                image[num_cols, :len(col)] = col
                image[num_cols, :len(col)] += 1
                num_cols += 1
        return image.T[::-1]
    
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
            return self.goal_vector[obj_B - 1, obj_A - 1] == 1
        return self.vector_state[obj_B - 1, obj_A - 1] == 1
    
    def set_clear(self, obj, is_clear=True, is_goal=False):
        if is_goal:
            self.goal_vector[obj - 1, -2] = is_clear
        else:
            self.vector_state[obj - 1, -2] = is_clear
        self.set_vector_state(self.vector_state, is_goal)
    
    def set_on_table(self, obj, is_on_table=True, is_goal=False):
        if is_goal:
            self.goal_vector[obj - 1, -1] = is_on_table
        else:
            self.vector_state[obj - 1, -1] = is_on_table
        self.set_vector_state(self.vector_state, is_goal)
    
    def set_on(self, obj_A, obj_B, is_goal=False):
        ''' Set obj_A on obj_B
        '''
        N = self.num_blocks
        if is_goal:
            self.goal_vector[:N, obj_A - 1] = 0
            self.goal_vector[obj_B - 1, obj_A - 1] = 1
        else:
            self.vector_state[:N, obj_A - 1] = 0
            self.vector_state[obj_B - 1, obj_A - 1] = 1
        self.set_vector_state(self.vector_state, is_goal)
    
    def put_A_on_B_(self, obj_A, obj_B):   
        ''' A should be on table and clear, B should be clear
        '''
        if obj_A == obj_B:
            return False
        if self.is_on(obj_A, obj_B):
            return False
        if not self.is_on_table(obj_A):
            return False
        if not self.is_clear(obj_A):
            return False
        if not self.is_clear(obj_B):
            return False
        
        N = self.num_blocks
        # B becomes not clear 
        self.set_clear(obj_B, is_clear=False)
        # A becomes not on table
        self.set_on_table(obj_A, is_on_table=False)
        # A is not on any block other than B
        self.set_on(obj_A, obj_B)
        return True 

    def put_on_table(self, obj_A):
        if self.is_on_table(obj_A):
            return False
        if not self.is_clear(obj_A):
            return False
        
        N = self.num_blocks
        # A becomes on table    
        self.set_on_table(obj_A)
        # A is not on any block
        A_is_on = np.where(self.vector_state[:N, obj_A - 1] == 1)[0]
        for obj in A_is_on:
            self.vector_state[obj, obj_A - 1] = 0
            self.vector_state[obj, -2] = 1
        self.set_vector_state(self.vector_state)
        return True
    
    def put_A_on_B(self, obj_A, obj_B):
        ''' A should be clear, yet not necessarily on table;
        B should be clear
        '''
        self.before_state = self.vector_state.copy()
        if not self.is_on_table(obj_A):
            put_A_on_table = self.put_on_table(obj_A)
            if not put_A_on_table:
                self.vector_state = self.before_state
                return False
        return self.put_A_on_B_(obj_A, obj_B)
        
        
if __name__ == '__main__':
    env = BlockStackingEnv()
    
    ### Test Initial state
    print(env)

    ### Test state setting
    coords = np.array([
        [1, 1, 0], [1, 1, 1], [1, 2, 0], [1, 2, 1],
        [1, 2, 2], [2, 10, 0], [2, 2, 10], [2, 3, -5]
    ])
    env.set_coords_state(coords)
    print(env)
    
    ### Test language state translation
    print(env.get_language_state())
    print(env.get_language_state(is_goal=True))
    
    ### Test put on action with A on table
    print(f"Try to move 4 on 2: {env.put_A_on_B_(4, 2)}")
    print(f"Try to move 5 on 2: {env.put_A_on_B_(5, 2)}")
    print(f"Try to move 6 on 2: {env.put_A_on_B_(6, 2)}")
    print(env)
    print(env.get_language_state())
    
    ### Test move on table action
    print(f"Try to move 1 on table: {env.put_on_table(1)}")
    print(f"Try to move 2 on table: {env.put_on_table(2)}")
    print(f"Try to move 6 on table: {env.put_on_table(6)}")
    print(env)
    print(env.get_language_state())
    
    # ### Test put on action with A not on table
    print(f"Try to move 4 on 2: {env.put_A_on_B(4, 2)}")
    print(f"Try to move 5 on 2: {env.put_A_on_B(5, 2)}")
    print(f"Try to move 4 on 1: {env.put_A_on_B(6, 2)}")
    print(env)
    print(env.get_language_state())
            
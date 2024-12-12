import matplotlib.pyplot as plt
import numpy as np


def render_coords(coords: np.ndarray, img_path='./rendered/{}.png'.format(np.random.randint(2 << 14))):
    """render the block state with matplotlib

    Args:
        coords (np.ndarray): array of shape (3, N) denoting the coordinates of blocks
        img_path (str, optional): export path. Defaults to './rendered/{randint}.png'
    """
    grid = np.zeros((8, 8, 8))
    colors = np.array([
        'red',
        'orange',
        'yellow',
        'green',
        'blue',
        'purple',
        'gray',
        'black'
    ], dtype=str)
    grid[*coords] = 1
    colorgrid = np.empty_like(grid, dtype=object)
    colorgrid[*coords] = colors
    
    
    ax = plt.figure().add_subplot(projection='3d')
    
    ax.voxels(grid, edgecolor='k')
    
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_zlim(0, 8)
    
    # Set axis labels for clarity
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.voxels(grid, facecolors=colorgrid, edgecolor='k')

    # plt.savefig(img_path)

if __name__ == '__main__':
    with open('data/8blocks-500_test.npz', 'rb') as f:
        data = np.load(f)
        coords = data['coords']
        labels = data['labels']
    
    np.random.seed(42)
    t = np.random.randint(coords.shape[0])
    coords = coords[t]
    labels = labels[t]
    
    from environment import BlockStackingEnv
    env = BlockStackingEnv(num_blocks=8)
    env.set_vector_state(labels)
    
    coords_unique = np.unique(coords[..., :2], axis=0)
    
    while coords_unique.shape[0] < 8:
        new_coord = np.random.randint(-3, 4, size=(1, 2))
        if not np.any(np.all(coords_unique == new_coord, axis=0)):
            coords_unique = np.vstack((coords_unique, new_coord))
    
    env.set_coords(coords_unique)
        
    render_coords(env.get_coords_3d().T)
    
    print(env.put_A_on_B(3, 5))
    
    render_coords(env.get_coords_3d().T)
    
    
    plt.show()
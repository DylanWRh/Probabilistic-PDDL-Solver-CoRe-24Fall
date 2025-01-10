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

    plt.savefig(img_path)

if __name__ == '__main__':
    render_coords(
        np.asarray([
            [1, 2, 0],
            [3, 2, 0],
            [1, 2, 1],
            [4, 7, 0],
            [4, 5, 0],
            [4, 5, 1],
            [3, 6, 0],
            [0, 3, 0]
        ]).T,
        './rendered/simple_example.png'
    )

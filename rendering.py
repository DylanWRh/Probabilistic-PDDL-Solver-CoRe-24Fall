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
    # ax.voxels(grid, edgecolor='k')
    ax.voxels(grid, facecolors=colorgrid, edgecolor='k')

    plt.savefig(img_path)
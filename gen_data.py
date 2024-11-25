import numpy as np
import os


def gen_data(n_blocks=8):
    img = -np.ones([n_blocks, n_blocks], dtype=int)
    vector = np.zeros([n_blocks, n_blocks + 2], dtype=int)
    permutation = np.random.permutation(n_blocks)
    # permutation = np.arange(n_blocks)
    n_cols = np.random.randint(1, n_blocks)
    split_points = np.sort(np.random.choice(range(n_blocks), n_cols - 1, replace=False))
    selected_cols = np.random.permutation(n_blocks)[:n_cols]
    n_blocks_per_col = np.diff([0] + (split_points + 1).tolist() + [n_blocks])
    rol_idxs = np.repeat(np.cumsum([0] + n_blocks_per_col.tolist())[:-1], n_blocks_per_col) - np.arange(n_blocks) + n_blocks - 1
    col_idxs = np.repeat(selected_cols, n_blocks_per_col)
    # print('[debug]: generated permutation: {}, generated n_cols: {}, generated splitpoints: {}, n_block_per_col: {}, generated column index: {}, generated rol index: {}'.format(permutation, n_cols, split_points, n_blocks_per_col, col_idxs, rol_idxs))
    img[rol_idxs, col_idxs] = permutation[range(n_blocks)]
    # print('[debug]: image:\n {}'.format(img))
    vector[permutation[split_points], -2] = 1
    vector[permutation[-1], -2] = 1
    vector[permutation[np.where(rol_idxs == n_blocks - 1)[0]], -1] = 1
    for i in range(n_blocks - 1):
        if col_idxs[i] == col_idxs[i + 1]:
            vector[permutation[i]][permutation[i + 1]] = 1
    # print('[debug]: vector:\n {}'.format(vector))
    return img, vector


def vector2language(vector: np.ndarray, spliter=', '):
    assert len(vector.shape) == 2 and vector.shape[0] == vector.shape[1] - 2
    N, D = vector.shape
    # print('[debug]: clear objs: {}'.format(np.where(vector[:, -2])))
    language = spliter.join(['clear({})'.format(idx) for idx in (np.where(vector[:, -2])[0])])
    language += '\n'
    # print('[debug]: language: {}'.format(language))
    language += spliter.join(['On({}, {})'.format(j, i) for i, j in zip(*np.where(vector[:, :-2]))])
    language += '\n'
    language += spliter.join(['OnTable({})'.format(i) for i in np.where(vector[:, -1])[0]])
    # print('[debug]: language: {}'.format(language))
    return language


if __name__ == '__main__':
    root = './data'
    n_sample = 1000
    n_block = 8
    datas = [gen_data(n_block) for _ in range(n_sample)]
    images = np.stack([data[0] for data in datas], axis=0)
    vectors = np.stack([data[1] for data in datas], axis=0)
    np.savez(os.path.join(root, '{}blocks-{}.npz'.format(n_block, n_sample)), images=images, vectors=vectors)

import numpy as np
import scipy.sparse
from tqdm import tqdm


def norma(feat):
    minn = feat.min()
    return (feat - minn) / (feat.max() - minn)


def np_group_by(key, values, sorted=False):
    is_single_value = isinstance(values, np.ndarray)
    if is_single_value:
        values = [values]
    if not sorted:
        sort_idx = argsort_axis0(key)
        key = key[sort_idx]
        values = [v[sort_idx] for v in values]

    unique_key, return_index = np.unique(key, return_index=True, axis=0)
    split_idx = return_index[1:]
    groups = [np.split(v, split_idx) for v in values]
    if is_single_value:
        groups = groups[0]
    return unique_key, groups


def argsort_axis0(key):
    if key.ndim == 1:
        return np.argsort(key)
    unique_key, new_value_to_sort = np.unique(key, return_inverse=1, axis=0)
    sort_idx = np.argsort(new_value_to_sort)
    return sort_idx


def graph_conv_by_affinity(feat, affinity, momenta=0.6, order=1, itern=10, visn=0):
    if isinstance(affinity, np.ndarray):
        affinity = scipy.sparse.coo_matrix(affinity)

    sorted_idx = np.argsort(affinity.row)
    row = affinity.row[sorted_idx]
    col = affinity.col[sorted_idx]
    data = affinity.data[sorted_idx]
    assert data.min() >= 0 and data.max() <= 1

    indirect_rcds = []
    for order_idx in range(1, order):
        indirect_rcds.clear()
        row_to_cd = np.array([None] * affinity.shape[0])
        unique_row, (split_col, split_data) = np_group_by(row, [col, data])
        # (n, [col(neighbor*int), data(neighbor*float)])
        _cds = np.zeros((affinity.shape[0],), dtype=object)
        _cds[:] = list(zip(split_col, split_data))
        row_to_cd[unique_row] = _cds
        order_rate = 1
        for row_idx, (direct_col, direct_data) in enumerate(row_to_cd):
            indirect_cds = row_to_cd[direct_col]
            for direct_idx, (indirect_col, indirect_data) in enumerate(indirect_cds):
                weight = direct_idx
                indirect_rcds.append(
                    (
                        np.ones_like(indirect_col) * row_idx,
                        indirect_col,
                        indirect_data * direct_data[direct_idx] * order_rate,
                    )
                )

        indirect_row = np.concatenate([rcd[0] for rcd in indirect_rcds])
        indirect_col = np.concatenate([rcd[1] for rcd in indirect_rcds])
        indirect_data = np.concatenate([rcd[2] for rcd in indirect_rcds])

        row = np.append(row, indirect_row)
        col = np.append(col, indirect_col)
        data = np.append(data, indirect_data)

        # remove replica
        unique_edge, groups = np_group_by(np.array([row, col]).T, data)

        row = unique_edge[:, 0]
        col = unique_edge[:, 1]
        data = np.array([gp.max() for gp in groups])

    unique_row, counts = np.unique(row, return_counts=1)
    compact_n = max(counts)
    compact_col_idx = np.concatenate(list(map(np.arange, counts))) % compact_n
    compact = np.zeros((affinity.shape[0], compact_n), affinity.dtype)

    col_idx_to_compact_shape = -np.ones_like(compact, np.int32)
    col_idx_to_compact_shape[row, compact_col_idx] = col

    new_feat = feat.copy()
    for idx in range(itern):
        # compute by batchs for save memory
        batch = 200000 * 20 * 90 // compact_n // feat.shape[-1]
        new_feat_ = new_feat.copy()
        for batch_idx in tqdm(range(int(np.ceil(affinity.shape[0] / batch)))):
            # slice_on_row = boxx.sliceInt[batch_idx * batch : batch_idx * batch + batch]
            slice_on_row = slice(batch_idx * batch, batch_idx * batch + batch)
            # feat_ = feat[slicee]
            slice_on_coo = (batch_idx * batch <= row) & (
                row < batch_idx * batch + batch
            )
            col_idx_to_compact_shape_ = col_idx_to_compact_shape[slice_on_row]
            feat_in_compact_shape_ = new_feat[col_idx_to_compact_shape_]

            compact[row[slice_on_coo], compact_col_idx[slice_on_coo]] = data[
                slice_on_coo
            ]
            compact_ = compact[slice_on_row]
            weight = compact_ / compact_.sum(-1, keepdims=True)
            aggregation = (weight[..., None] * feat_in_compact_shape_).sum(-2)
            new_feat_[slice_on_row] = new_feat[slice_on_row] * momenta + aggregation * (
                1 - momenta
            )
        new_feat = new_feat_

    return new_feat

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch


def read_binvox_coords(f, integer_division=True, dtype=torch.float32):
    """
    Read a binvox file and return the indices of all nonzero voxels.

    This matches the behavior of binvox_rw.read_as_coord_array
    (https://github.com/dimatura/binvox-rw-py/blob/public/binvox_rw.py#L153)
    but this implementation uses torch rather than numpy, and is more efficient
    due to improved vectorization.

    I think that binvox_rw.read_as_coord_array actually has a bug; when converting
    linear indices into three-dimensional indices, they use floating-point
    division instead of integer division. We can reproduce their incorrect
    implementation by passing integer_division=False.

    Args:
      f (str): A file pointer to the binvox file to read
      integer_division (bool): If False, then match the buggy implementation from binvox_rw
      dtype: Datatype of the output tensor. Use float64 to match binvox_rw

    Returns:
      coords (tensor): A tensor of shape (N, 3) where N is the number of nonzero voxels,
           and coords[i] = (x, y, z) gives the index of the ith nonzero voxel. If the
           voxel grid has shape (V, V, V) then we have 0 <= x, y, z < V.
    """
    size, translation, scale = _read_binvox_header(f)
    storage = torch.ByteStorage.from_buffer(f.read())
    data = torch.tensor([], dtype=torch.uint8)
    data.set_(source=storage)
    vals, counts = data[::2], data[1::2]
    idxs = _compute_idxs_v2(vals, counts)
    if not integer_division:
        idxs = idxs.to(dtype)
    x_idxs = idxs / (size * size)
    zy_idxs = idxs % (size * size)
    z_idxs = zy_idxs / size
    y_idxs = zy_idxs % size
    coords = torch.stack([x_idxs, y_idxs, z_idxs], dim=1)
    return coords.to(dtype)


def _compute_idxs_v1(vals, counts):
    """ Naive version of index computation with loops """
    idxs = []
    cur = 0
    for i in range(vals.shape[0]):
        val, count = vals[i].item(), counts[i].item()
        if val == 1:
            idxs.append(torch.arange(cur, cur + count))
        cur += count
    idxs = torch.cat(idxs, dim=0)
    return idxs


def _compute_idxs_v2(vals, counts):
    """ Fast vectorized version of index computation """
    # Consider an example where:
    # vals   = [0, 1, 0, 1, 1]
    # counts = [2, 3, 3, 2, 1]
    #
    # These values of counts and vals mean that the dense binary grid is:
    # [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    #
    # So the nonzero indices we want to return are:
    # [2, 3, 4, 8, 9, 10]

    # After the cumsum we will have:
    # end_idxs = [2, 5, 8, 10, 11]
    end_idxs = counts.cumsum(dim=0)

    # After masking and computing start_idx we have:
    # end_idxs   = [5, 10, 11]
    # counts     = [3,  2,  1]
    # start_idxs = [2,  8, 10]
    mask = vals == 1
    end_idxs = end_idxs[mask]
    counts = counts[mask].to(end_idxs)
    start_idxs = end_idxs - counts

    # We initialize delta as:
    # [2, 1, 1, 1, 1, 1]
    delta = torch.ones(counts.sum().item(), dtype=torch.int64)
    delta[0] = start_idxs[0]

    # We compute pos = [3, 5], val = [3, 0]; then delta is
    # [2, 1, 1, 4, 1, 1]
    pos = counts.cumsum(dim=0)[:-1]
    val = start_idxs[1:] - end_idxs[:-1]
    delta[pos] += val

    # A final cumsum gives the idx we want: [2, 3, 4, 8, 9, 10]
    idxs = delta.cumsum(dim=0)
    return idxs


def _read_binvox_header(f):
    # First line of the header should be "#binvox 1"
    line = f.readline().strip()
    if line != b"#binvox 1":
        raise ValueError("Invalid header (line 1)")

    # Second line of the header should be "dim [int] [int] [int]"
    # and all three int should be the same
    line = f.readline().strip()
    if not line.startswith(b"dim "):
        raise ValueError("Invalid header (line 2)")
    dims = line.split(b" ")
    try:
        dims = [int(d) for d in dims[1:]]
    except ValueError:
        raise ValueError("Invalid header (line 2)")
    if len(dims) != 3 or dims[0] != dims[1] or dims[0] != dims[2]:
        raise ValueError("Invalid header (line 2)")
    size = dims[0]

    # Third line of the header should be "translate [float] [float] [float]"
    line = f.readline().strip()
    if not line.startswith(b"translate "):
        raise ValueError("Invalid header (line 3)")
    translation = line.split(b" ")
    if len(translation) != 4:
        raise ValueError("Invalid header (line 3)")
    try:
        translation = tuple(float(t) for t in translation[1:])
    except ValueError:
        raise ValueError("Invalid header (line 3)")

    # Fourth line of the header should be "scale [float]"
    line = f.readline().strip()
    if not line.startswith(b"scale "):
        raise ValueError("Invalid header (line 4)")
    line = line.split(b" ")
    if not len(line) == 2:
        raise ValueError("Invalid header (line 4)")
    scale = float(line[1])

    # Fifth line of the header should be "data"
    line = f.readline().strip()
    if not line == b"data":
        raise ValueError("Invalid header (line 5)")

    return size, translation, scale

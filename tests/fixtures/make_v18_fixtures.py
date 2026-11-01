"""Write the v18 fixtures used by tests/v18_layout.rs.

libver=("v108", "v108") is the layout netCDF-4 files usually have: version 2
object headers, version 3 attribute messages (with a name character set
byte), dense attribute storage from 8 attributes on, and version 1 B-tree
chunk indexes.

    python tests/fixtures/make_v18_fixtures.py
"""

from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).parent

with h5py.File(HERE / "v18_layout.h5", "w", libver=("v108", "v108")) as f:
    # Compact attributes, written as version 3 attribute messages.
    small = f.create_dataset("small", data=np.arange(4, dtype=np.int16))
    small.attrs["units"] = "m"
    small.attrs["scale_factor"] = np.float32(0.5)

    # 12 attributes: more than the compact limit of 8, so stored densely
    # in a fractal heap with a name-index v2 B-tree.
    dense = f.create_dataset("dense", data=np.zeros(3, dtype=np.float32))
    for i in range(12):
        dense.attrs[f"attr_{i:02d}"] = np.int32(i)
    dense.attrs["long_name"] = "dense attribute storage"

    # 20 x 20 chunks of 1 x 1 in a version 1 B-tree: 400 entries, more
    # than one leaf holds (2K = 64), so the tree has two levels.
    grid = np.arange(20 * 20, dtype=np.int32).reshape(20, 20)
    f.create_dataset("grid", data=grid, chunks=(1, 1))

# libhdf5's chunk list for "grid" (H5Dget_chunk_info), the oracle for the
# chunk index walk: row, col, filter mask, byte offset, size.
with h5py.File(HERE / "v18_layout.h5", "r") as f:
    d = f["grid"].id
    with open(HERE / "v18_layout.grid.csv", "w") as out:
        out.write("row,col,filter_mask,offset,size\n")
        for i in range(d.get_num_chunks()):
            c = d.get_chunk_info(i)
            out.write(
                f"{c.chunk_offset[0]},{c.chunk_offset[1]},"
                f"{c.filter_mask},{c.byte_offset},{c.size}\n"
            )

from landlab import RasterModelGrid
import numpy as np

grid = RasterModelGrid((3, 3))
# Link 0 connects node 0 to node 1.
# Points Right (Away from 0, Towards 1).
dirs = grid.link_dirs_at_node
print(f"Node 0, Link 0 dir: {dirs[0, 0]}") # Expect 1 (Away)
print(f"Node 1, Link 2 dir: {dirs[1, 2]}") # Expect -1 (Towards)

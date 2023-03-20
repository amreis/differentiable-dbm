import numpy as np

# I want to iterate from a given point in a grid towards the
# edges, in circles of equidistant points ("levels"), until a
# condition is met.
from queue import PriorityQueue
from itertools import chain

dbm = np.c_[np.zeros((10, 3)), np.ones((10, 3)), np.full((10, 4), 2)]

dbm[2:4, 5:8] = 4

neighboring_class = {}
border_points = []

for (i, j), cl in np.ndenumerate(dbm):
    # 4-neighborhood, then 8-neighborhood
    for dx, dy in (
        (0, 1),
        (-1, 0),
        (0, -1),
        (1, 0),
        (1, 1),
        (-1, 1),
        (-1, -1),
        (1, -1),
    ):
        if not (0 <= dx + i < 10 and 0 <= dy + j < 10):
            continue
        if (neigh_cl := dbm[dx + i, dy + j]) != cl:
            neighboring_class[(i, j)] = neigh_cl
            break

to_expand = list(neighboring_class.keys())

while to_expand:
    i, j = to_expand.pop(0)

    for dx, dy in ((0, 1), (-1, 0), (0, -1), (1, 0)):
        if not (0 <= dx + i < 10 and 0 <= dy + j < 10):
            continue
        new_point = (i + dx, j + dy)
        if new_point in neighboring_class:
            continue
        neighboring_class[new_point] = neighboring_class[i, j]
        to_expand.append(new_point)

neighbors_dbm = np.zeros_like(dbm) * np.nan
for point, cl in neighboring_class.items():
    neighbors_dbm[point] = cl

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"aspect": 1})
ax1.imshow(dbm, cmap="tab10", origin="lower", interpolation="none")
ax2.imshow(neighbors_dbm, cmap="tab10", origin="lower", interpolation="none")
plt.show()

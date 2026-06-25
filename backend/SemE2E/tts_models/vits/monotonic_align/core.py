"""Pure Python fallback for the VITS monotonic alignment extension.

The original implementation ships as ``core.pyx`` and is faster when compiled,
but local Windows environments often do not have Cython/MSVC ready.  This module
keeps the same mutating ``maximum_path_c`` interface so model initialization can
run without the compiled extension.
"""

from __future__ import annotations


def _maximum_path_each(path, value, t_y: int, t_x: int, max_neg_val: float = -1.0e9) -> None:
    index = t_x - 1
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = max_neg_val if x == y else value[y - 1, x]
            if x == 0:
                v_prev = 0.0 if y == 0 else max_neg_val
            else:
                v_prev = value[y - 1, x - 1]
            value[y, x] += max(v_prev, v_cur)

    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index -= 1


def maximum_path_c(paths, values, t_ys, t_xs) -> None:
    for i in range(paths.shape[0]):
        _maximum_path_each(paths[i], values[i], int(t_ys[i]), int(t_xs[i]))

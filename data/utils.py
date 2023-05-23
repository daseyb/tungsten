import numpy as np
import matplotlib.pyplot as plt
import os 

def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def ray_points(o, d, tmin, tmax, n):
    ts = np.linspace(tmin, tmax, n)
    ps = np.array(o)[None, ...] + ts[..., None] * np.array(d)[None, ...]
    return ps, ts

def plot_realization(realization, ax, xx, yy):
    realization = -realization
    max_v = np.max(realization)
    min_v = np.min(realization)

    abs_range = np.maximum(np.abs(max_v), np.abs(min_v))
    levels = np.linspace(-abs_range, abs_range, 11)

    cs = ax.contour(xx, yy, realization, alpha=0.5, levels = levels, cmap="coolwarm")
    if(len(cs.collections) > 5):
        cs.collections[5].set_linewidth(3)
        cs.collections[5].set_color("black")
        cs.collections[5].set_alpha(1.0)

        ax.contourf(xx, yy, realization, alpha=0.5,  levels = levels, cmap="coolwarm")
        ax.set_aspect("equal")

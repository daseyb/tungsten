import numpy as np
import matplotlib.pyplot as plt
import os 
import scipy.interpolate as interp

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

def plot_realization(realization, ax, xx, yy, clw=3, alpha=1, zorder=-10, level_set=True, levels=11, max_range=None):
    with plt.style.context("default", True):
        realization = -realization
        max_v = np.max(realization)
        min_v = np.min(realization)

        abs_range = np.maximum(np.abs(max_v), np.abs(min_v))

        if max_range is not None:
            abs_range = np.minimum(abs_range, max_range)

        if levels is not None:
            levels = np.linspace(-abs_range, abs_range, levels if level_set else 10)
            cs = ax.contour(xx, yy, realization, alpha=0.5*alpha, levels = levels, cmap="coolwarm", zorder=zorder)
            if(len(cs.collections) > 5):
                ax.contourf(xx, yy, realization, alpha=0.5*alpha,  levels = levels, cmap="coolwarm", zorder=zorder)
        else:
            ax.pcolormesh(xx,yy,realization,vmin=-abs_range, vmax=abs_range, cmap="coolwarm", zorder=zorder)

        if(level_set):
            ax.contour(xx, yy, realization, alpha=1*alpha, levels = [0], linewidths=clw, colors="black", zorder=zorder)

        ax.set_aspect("equal")
        return abs_range


def plot_posterior(file, ax_occupancy, ax_surface_density=None, num_reals=200, shared_file_id="grid-samples-fs", target_file_id = "grid-samples-cond"):
    target_file = file if target_file_id is None else file.replace(shared_file_id, "grid-samples-cond")
    realizations = np.fromfile(target_file, dtype=np.float64)
    res = int(np.sqrt(realizations.shape[0]/num_reals))
    realizations= np.reshape(realizations, (num_reals, res, res))[:500,...]
    realizations = realizations.transpose(0, 2, 1)

    cond_ps_file = file.replace(shared_file_id, "cond-ps-cond")
    if os.path.exists(cond_ps_file):
        cond_ps = np.reshape(np.fromfile(cond_ps_file, dtype=np.float64), (-1, 3))
        cond_ns = np.reshape(np.fromfile(file.replace(shared_file_id, "cond-ns-cond"), dtype=np.float64), (-1, 3))
        cond_ds = np.reshape(np.fromfile(file.replace(shared_file_id, "cond-ds-cond"), dtype=np.uint8), (-1, 1))
        cond_vs = np.reshape(np.fromfile(file.replace(shared_file_id, "cond-vs-cond"), dtype=np.float64), (-1, 1))
    else:
        cond_ps = None

    xs = np.linspace(-1, 1, realizations.shape[1])
    ys = np.linspace(-1, 1, realizations.shape[2])
    xx, yy = np.meshgrid(xs,ys)

    grid_interp = interp.RegularGridInterpolator((xs, ys), realizations.transpose(2, 1, 0))
    hd_xs = np.linspace(-1, 1, 500)
    hd_ys = np.linspace(-1, 1, 500)
    hd_xx, hd_yy = np.meshgrid(hd_xs,hd_ys)

    interp_reals = np.atleast_2d(grid_interp((hd_xx, hd_yy)))

    if ax_occupancy is not None:
        occupancy = np.mean(interp_reals < 0, axis=2)
        #ax_occupancy.contourf(hd_xx, hd_yy, occupancy, vmin=0, vmax=1, alpha=0.5, levels = np.linspace(0, 1, 21), cmap="coolwarm", zorder=-10)
        ax_occupancy.pcolormesh(hd_xx, hd_yy, occupancy, vmin=0, vmax=1, alpha=0.5, cmap="coolwarm", zorder=-10)
        #ax_occupancy.set_title("Occupancy ($P(f(x) < 0)$)")
        ax_occupancy.set_xticks([])
        ax_occupancy.set_yticks([])
        ax_occupancy.set_aspect("equal")

    if ax_surface_density is not None:
        surface_density = np.mean(np.abs(interp_reals) < 0.01, axis=2)
        cs = ax_surface_density.contourf(hd_xx, hd_yy, surface_density, vmin=0, vmax=1, alpha=0.5, levels = np.linspace(0, 1, 21), cmap="coolwarm", zorder=-10)
        #ax_surface_density.set_title("Surface density ($p(f(x) = 0)$)")
        ax_surface_density.set_xticks([])
        ax_surface_density.set_yticks([])
        ax_surface_density.set_aspect("equal")

    if cond_ps is not None:
        is_fo = (cond_ds == 0).flatten()
        if ax_occupancy:
            ax_occupancy.scatter(cond_ps[is_fo,0], cond_ps[is_fo,1], color="red", s=10, zorder=10, alpha=0.5)
        
        if ax_surface_density:
            ax_surface_density.scatter(cond_ps[is_fo,0], cond_ps[is_fo,1], color="red", s=10, zorder=10, alpha=0.5)

        is_so = ~is_fo
        if np.any(is_so):
            if ax_occupancy:
                ax_occupancy.quiver(
                    cond_ps[is_so,0], cond_ps[is_so,1], 
                    cond_ns[is_so, 0] * cond_vs[is_so, 0], cond_ns[is_so, 1] * cond_vs[is_so, 0], 
                    color="green", zorder=10, alpha=1)
            if ax_surface_density:
                ax_surface_density.quiver(
                    cond_ps[is_so,0], cond_ps[is_so,1], 
                    cond_ns[is_so, 0] * cond_vs[is_so, 0], cond_ns[is_so, 1] * cond_vs[is_so, 0], 
                    color="green", zorder=10, alpha=1)
                
def plot_prior(file, ax_real, num_reals=1, level_set=True, levels=11, max_range=None):
    realizations = np.fromfile(file, dtype=np.float64)
    res = int(np.sqrt(realizations.shape[0]/num_reals))
    realizations= np.reshape(realizations, (num_reals, res, res))
    realizations = realizations.transpose(0, 2, 1)
    val_range = 0
    for real in realizations:
        val_range = max(val_range, plot_realization(real, ax_real, np.linspace(-1,1,res), np.linspace(-1,1,res), 
                                                    alpha=1/num_reals, zorder=-10, 
                                                    level_set=level_set, levels=levels, max_range=max_range))
    ax_real.set_xticks([])
    ax_real.set_yticks([])
    return val_range

def plot_kernel(file, ax_kernel):
    kernel = np.fromfile(file.replace("32-grid-samples-fs", "kernel"), dtype=np.float64)
    kernel_dx = np.fromfile(file.replace("32-grid-samples-fs", "kernel-dx"), dtype=np.float64)
    kernel_dxdy = np.fromfile(file.replace("32-grid-samples-fs", "kernel-dxdy"), dtype=np.float64)

    ts = np.linspace(0, 2, kernel.shape[0])
    ax_kernel.grid()
    ax_kernel.plot(ts, kernel, label="$k(0,t)$")
    ax_kernel.plot(ts, kernel_dx, label="$k_y(0,t)$")
    ax_kernel.plot(ts, kernel_dxdy/kernel_dxdy[0], label="$k_{x,y}(0,t)$")
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D

_cmap = plt.cm.jet


def function_value_3d(f, range):
    x = np.arange(-range, range, 0.01)
    y = np.arange(-range, range, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = f.f_opt([X, Y])
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title(f.name)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")

    ax.plot_surface(X, Y, Z)
    return ax


def function_value_transition(f_store, title="function value transition"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(f_store)
    ax.set_title(title)

    return ax


def w_value_2d_heatmap(f, w_store, _t_max, title="w value"):
    w_store = np.array(w_store)
    w_star = f.w_star
    grid_x_min = min(w_store.T[0].min(), w_star[0]) - 1
    grid_x_max = max(w_store.T[0].max(), w_star[0]) + 1
    grid_y_min = min(w_store.T[1].min(), w_star[1]) - 1
    grid_y_max = max(w_store.T[1].max(), w_star[1]) + 1
    xvals = np.arange(grid_x_min, grid_x_max, 0.1)
    yvals = np.arange(grid_y_min, grid_y_max, 0.1)
    X, Y = np.meshgrid(xvals, yvals)
    Z = f.f_opt([X, Y])

    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    axes.pcolor(X, Y, Z, cmap=plt.cm.rainbow)
    # wの軌跡
    axes.plot(w_store.T[0], w_store.T[1], c="k", alpha=0.2, linewidth=1)
    c = np.linspace(0, _t_max, len(w_store))
    axes.scatter(w_store.T[0], w_store.T[1], c=c, cmap=plt.cm.hot, linewidths=0.01, alpha=1, s=10)

    # 始点(黄色)、終点（緑）、真値（赤）
    axes.plot(*w_store[0], 'ks', markersize=5, label="start")
    axes.plot(*w_store[-1], 'gs', markersize=5, label="finish")
    axes.plot(*w_star, 'r*', markersize=8, label="true value")
    axes.set_title(title)

    # カラーバーの設定
    axpos = axes.get_position()
    cbar_ax = fig.add_axes([0.9, axpos.y0, 0.03, axpos.height])
    norm = colors.Normalize(vmin=Z.min(), vmax=Z.max())
    mappable = ScalarMappable(cmap=plt.cm.rainbow, norm=norm)
    mappable._A = []
    fig.colorbar(mappable, cax=cbar_ax)

    # 余白の調整
    plt.subplots_adjust(right=0.85)
    plt.subplots_adjust(wspace=0.1)

    plt.show()

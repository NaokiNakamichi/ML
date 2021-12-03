import numpy as np
import os
import datetime

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
    axes.pcolor(X, Y, Z, cmap=plt.cm.rainbow, shading='auto')
    # wの軌跡
    axes.plot(w_store.T[0], w_store.T[1], c="k", alpha=0.2, linewidth=1)
    c = np.linspace(0, _t_max, w_store.shape[0])
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


def w_value_2d_k_candidates_contour(f, core_store, _t_max, selected_index, title="w value", levels=None,
                                    trajectory=True):
    # 等高線を引くためのグリッド作成
    w_star = f.w_star
    # 等高線を引くためのグリッド作成
    grid_x_max, grid_y_max = np.amax(core_store, axis=(0, 1)) + 1
    grid_x_min, grid_y_min = np.amin(core_store, axis=(0, 1)) - 1
    xvals = np.arange(grid_x_min, grid_x_max, 0.01)
    yvals = np.arange(grid_y_min, grid_y_max, 0.01)
    X, Y = np.meshgrid(xvals, yvals)
    # グリッドにおける関数値
    Z = f.f_opt([X, Y])

    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    if not levels:
        levels = [0, 1, 10, 100, 400]
    # axes.pcolor(X, Y, Z, cmap=plt.cm.rainbow,shading='auto')
    ctrx = axes.contour(X, Y, Z,
                        levels=levels,  # 等高線の間隔
                        linewidths=1,  # 等高線の幅
                        colors="black"
                        )
    axes.clabel(ctrx)
    cmap = plt.get_cmap("tab10")
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # wの軌跡
    if trajectory:
        for k, w_store in enumerate(core_store):
            axes.plot(w_store.T[0], w_store.T[1], alpha=0.3, linewidth=3, color=cmap(k))
            pos = (np.array(w_store[:-1]) + np.array(w_store[1:])) / 2
            dir = np.array(w_store[:-1]) - np.array(w_store[1:])

            long = np.quantile((dir ** 2).sum(axis=1), 0.995)

            for i, (i_pos, i_dir) in enumerate(zip(pos, dir)):
                if long < (i_dir ** 2).sum():
                    axes.annotate("", xytext=(i_pos), xy=(i_pos + 0.01 * i_dir),
                                  arrowprops=dict(arrowstyle="<-", facecolor=cmap(k), edgecolor=cmap(k)),
                                  size=15 - 10 * (i / dir.shape[0] - 1))

    for k, w_store in enumerate(core_store):
        axes.plot(*w_store[-1], "o", markersize=10, label=f"process {k}", color=cmap(k))

    # 諸々の点、開始点とか最適値とか
    axes.plot(*core_store[selected_index][-1], "8", markersize=10, label=f"selected", color="k")
    axes.plot(*w_star, 'k*', markersize=12, label="optimal")
    axes.plot(*core_store[0][0], 'ks', markersize=5, label="w_init")
    axes.set_title(title)
    plt.legend()
    plt.show()


def multiple_w_value_2d_k_candidates_contour(f, k_list, k_list_core_store, _t_max, k_selected_index,
                                             title="w trajectory", folder_title="",
                                             trajectory=True, levels=None, saving_png=False):
    if saving_png:
        new_dir_path_recursive = folder_title
        for i, core_store in enumerate(k_list_core_store):
            plt.figure(figsize=(6, 6))
            w_star = f.w_star
            grid_x_max, grid_y_max = np.amax(core_store, axis=(0, 1))
            grid_x_min, grid_y_min = np.amin(core_store, axis=(0, 1))
            grid_x_max, grid_y_max = max(grid_x_max, w_star[0]) + 1, max(grid_y_max, w_star[1]) + 1
            grid_x_min, grid_y_min = min(grid_x_min, w_star[1]) - 1, min(grid_y_min, w_star[1]) - 1
            xvals = np.arange(grid_x_min, grid_x_max, ((grid_x_max - grid_x_min) / 500))
            yvals = np.arange(grid_y_min, grid_y_max, ((grid_y_max - grid_y_min) / 500))
            X, Y = np.meshgrid(xvals, yvals)
            Z = f.f_opt([X, Y])
            if not levels:
                levels = [0, 1, 10, 100, 400, 1000, 10000, 100000, 1000000]
            ctrx = plt.contour(X, Y, Z,
                               levels=levels,  # 等高線の間隔
                               linewidths=1,  # 等高線の幅
                               colors="black"
                               )
            plt.clabel(ctrx)
            cmap = plt.get_cmap("tab10")
            # wの軌跡
            if trajectory:
                for k, w_store in enumerate(core_store):
                    plt.plot(w_store.T[0], w_store.T[1], alpha=0.3, linewidth=3, color=cmap(k))
                    pos = (np.array(w_store[:-1]) + np.array(w_store[1:])) / 2
                    dire = np.array(w_store[:-1]) - np.array(w_store[1:])

                    long = np.quantile((dire ** 2).sum(axis=1), 0.995)

                    for i_annotate, (i_pos, i_dir) in enumerate(zip(pos, dire)):
                        if long < (i_dir ** 2).sum():
                            plt.annotate("", xytext=(i_pos), xy=(i_pos + 0.01 * i_dir),
                                         arrowprops=dict(arrowstyle="<-", facecolor=cmap(k), edgecolor=cmap(k)),
                                         size=15 - 10 * (i_annotate / dire.shape[0] - 1))

            for k, w_store in enumerate(core_store):
                plt.plot(*w_store[-1], "o", markersize=8, label=f"process {k}", color=cmap(k))

            plt.plot(*core_store[k_selected_index[i]][-1], "x", markersize=10, label=f"selected", color="r")
            plt.plot(*w_star, 'r*', markersize=12, label="optimal")
            plt.plot(*core_store[0][0], 'rs', markersize=5, label="w_init")
            plt.title(f"k = {k_list[i]} {title}")
            plt.legend()
            plt.savefig(f"{new_dir_path_recursive}/k_{k_list[i]}{title}.png")
            plt.show()


    else:
        fig, axes = plt.subplots(len(k_list_core_store), 1, figsize=(6, 36))
        for i, core_store in enumerate(k_list_core_store):
            w_star = f.w_star
            grid_x_max, grid_y_max = np.amax(core_store, axis=(0, 1))
            grid_x_min, grid_y_min = np.amin(core_store, axis=(0, 1))
            grid_x_max, grid_y_max = max(grid_x_max, w_star[0]) + 1, max(grid_y_max, w_star[1]) + 1
            grid_x_min, grid_y_min = min(grid_x_min, w_star[1]) - 1, min(grid_y_min, w_star[1]) - 1
            xvals = np.arange(grid_x_min, grid_x_max, ((grid_x_max - grid_x_min) / 500))
            yvals = np.arange(grid_y_min, grid_y_max, ((grid_y_max - grid_y_min) / 500))
            X, Y = np.meshgrid(xvals, yvals)
            Z = f.f_opt([X, Y])

            # plt.pcolor(X, Y, Z, cmap=plt.cm.rainbow,shading='auto')
            if not levels:
                levels = [0, 1, 10, 100, 400, 1000, 10000, 100000, 1000000]
            ctrx = axes[i].contour(X, Y, Z,
                                   levels=levels,  # 等高線の間隔
                                   linewidths=1,  # 等高線の幅
                                   colors="black"
                                   )
            axes[i].clabel(ctrx)
            cmap = plt.get_cmap("tab10")
            # wの軌跡
            if trajectory:
                for k, w_store in enumerate(core_store):
                    axes[i].plot(w_store.T[0], w_store.T[1], alpha=0.3, linewidth=3, color=cmap(k))
                    pos = (np.array(w_store[:-1]) + np.array(w_store[1:])) / 2
                    dire = np.array(w_store[:-1]) - np.array(w_store[1:])

                    long = np.quantile((dire ** 2).sum(axis=1), 0.995)

                    for i_annotate, (i_pos, i_dir) in enumerate(zip(pos, dire)):
                        if long < (i_dir ** 2).sum():
                            axes[i].annotate("", xytext=(i_pos), xy=(i_pos + 0.01 * i_dir),
                                             arrowprops=dict(arrowstyle="<-", facecolor=cmap(k), edgecolor=cmap(k)),
                                             size=15 - 10 * (i_annotate / dire.shape[0] - 1))

            for k, w_store in enumerate(core_store):
                axes[i].plot(*w_store[-1], "o", markersize=8, label=f"process {k}", color=cmap(k))

            axes[i].plot(*core_store[k_selected_index[i]][-1], "x", markersize=10, label=f"selected", color="r")
            axes[i].plot(*w_star, 'r*', markersize=12, label="optimal")
            axes[i].plot(*core_store[0][0], 'rs', markersize=5, label="w_init")
            axes[i].set_title(f"k = {k_list[i]} {title}")
            axes[i].legend()
        plt.show()


def box_plot_k(result, k_list, k_string, title, folder_title="", saving_png=False):
    fdic = {
        "size": 20,
    }

    columns = k_string
    fig = plt.figure(figsize=(10.0, 8.0))
    ax1 = fig.add_subplot(111)

    if len(result.shape) == 2:
        if result.shape[1] == len(k_list):

            ax1.boxplot(result)
        else:
            ax1.boxplot(result[:, k_list])
    else:
        raise ValueError("変数resultの形を確認してください")

    ax1.set_xticklabels(columns, fontsize=20)
    ax1.set_title(f'{title}', fontsize=10)
    ax1.set_xlabel('k', fontdict=fdic)
    if saving_png:
        new_dir_path_recursive = folder_title

        plt.savefig(f"{new_dir_path_recursive}/{title}.png")

    plt.show()


def transition(result, title, k_list_string, xlim=None, ylim=None):
    fig = plt.figure(figsize=(10.0, 8.0))
    ax1 = fig.add_subplot(111)
    ax1.set_title(title)
    for i in result:
        ax1.plot(i)
    ax1.set_xlabel("step")
    if xlim is not None:
        ax1.set_xlim(*xlim)
    if ylim is not None:
        ax1.set_ylim(*ylim)
    ax1.legend(labels=[f'k = {k}' for k in k_list_string])
    plt.show()

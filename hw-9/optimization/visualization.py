'''
Author: Aiden Li
Date: 2022-05-03 17:52:10
LastEditors: Aiden Li (i@aidenli.net)
LastEditTime: 2022-05-04 23:19:19
Description: Visualization functions.
'''
import matplotlib.pyplot as plt
import numpy as np

def contour_plot(fn, x_range=[-2.5, 2.5], y_range=[-2.5, 2.5], x_points=100, y_points=100, cmap='jet'):
    xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], x_points), np.linspace(y_range[0], y_range[1], y_points))
    zz = fn([xx, yy])
    
    fig, ax = plt.subplots(1, 1)
    bar = ax.contourf(xx, yy, zz, cmap=cmap, vmin=zz.min(), vmax=zz.max())
    
    fig.colorbar(bar)
    ax.set_title("Contour plot of $f(x, y)$")
    return fig
    
def contour_plot_with_tours(fn, x_trajs, traj_labels, x_range=[-2.5, 2.5], y_range=[-2.5, 2.5], x_points=100, y_points=100, cmap='jet'):
    xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], x_points), np.linspace(y_range[0], y_range[1], y_points))
    zz = fn([xx, yy])
    
    fig, ax = plt.subplots(1, 1)
    bar = ax.contourf(xx, yy, zz, cmap=cmap, vmin=zz.min(), vmax=zz.max())
    
    fig.colorbar(bar)
    ax.set_title("Contour plot of $f(x, y)$ and Descent Process")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    
    for i, x_traj in enumerate(x_trajs):
        x_traj = np.array([x[0] for x in x_trajs[i]])
        y_traj = np.array([x[1] for x in x_trajs[i]])
        plt.plot(x_traj, y_traj, label=traj_labels[i])
        plt.scatter(x_traj, y_traj, s=4.0, c=-fn((np.array(x_traj), np.array(y_traj))))
    
    plt.legend()
    return fig
    
def val_descent(val_trajs, traj_labels):
    fig, ax = plt.subplots(1, 1)
    ax.set_title("$f(x, y)$")
    plt.xlabel("Iteration")
    plt.ylabel("$f(x, y)$")
    
    for i in range(len(val_trajs)):
        plt.plot(np.arange(len(val_trajs[i])), val_trajs[i], label=traj_labels[i])
        
    plt.legend()
    return fig
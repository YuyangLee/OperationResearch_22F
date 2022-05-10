'''
Author: Aiden Li
Date: 2022-05-03 17:01:48
LastEditors: Aiden Li (i@aidenli.net)
LastEditTime: 2022-05-10 15:18:13
Description: Optimization of convex functions with GD and SD.
'''

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import trange

import descent_utils
import visualization

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--epsilon", default=1e-4, type=float)
    parser.add_argument("--max_iter", default=256, type=int)

    return parser.parse_args()

def descent(args, init_x, f, grad_f, hessian_f, descent_dir_fn):
    x_traj = [ init_x ]
    val_traj = [ f(init_x) ]
    
    for i in trange(args.max_iter):
        grad, d_x = descent_dir_fn(x_traj[-1], f, grad_f, hessian_f, None if i % len(init_x) == 0 else x_traj[-2])
        step_size = descent_utils.get_step_size(x_traj[-1], d_x, f, grad_f, hessian_f)
        
        x_traj.append(x_traj[-1] + step_size * d_x)
        val_traj.append(f(x_traj[-1]))
        if np.linalg.norm(grad, 2) < args.epsilon:
            break
        
    return x_traj, val_traj

if __name__ == '__main__':
    args = get_args()
    
    sns.set(rc={'figure.figsize': (12.0, 8.0)})
    np.random.seed(args.random_seed)

    f = lambda x: (1 - x[0])**2 + 2 * (x[0]**2 - x[1])**2
    grad_f = lambda x: np.array([2 * (4 * x[0]**3 - 4 * x[0] * x[1] + x[0] - 1), 4 * x[1] - 4 * x[0]**2])
    hessian_f = lambda x: np.asarray([
        [24 * x[0]**2 - 8 * x[1] + 2, -8 * x[0]],
        [-8 * x[0], 4]
    ])
    
    # init_x = np.random.randn(2)
    init_x = np.zeros(2)
    
    con_plot = visualization.contour_plot(f, [-1.5, 1.5], [-1.5, 1.5], 1000, 1000, 'RdBu')
    plt.savefig('hw-9/visualization/function_contour_plot.png')
    plt.show()
    
    methods_dict = {
        "1-norm": descent_utils.descent_1n,
        "2-norm": descent_utils.descent_2n,
        "inf-norm": descent_utils.descent_in,
        "fr": descent_utils.descent_fr,
        "pr": descent_utils.descent_pr
    }

    labels = []
    x_trajs = []
    y_trajs = []
    val_trajs = []
    
    for method, descent_fn in methods_dict.items():
        x_traj, val_traj = descent(
            args,
            init_x,
            f, grad_f, hessian_f,
            descent_fn
        )
        
        labels.append(method)
        x_trajs.append(x_traj)
        val_trajs.append(val_traj)
        
        print(f"{method}:\t(x, y) = ({ x_traj[-1][0] }, { x_traj[-1][1] })\tf(x, y) = {val_traj[-1]}")

    min_x = np.Inf
    min_y = np.Inf
    max_x = - np.Inf
    max_y = - np.Inf
    for x_traj in x_trajs:
        traj_min_x, traj_min_y = np.min(x_traj, axis=0)
        traj_max_x, traj_max_y = np.max(x_traj, axis=0)
        min_x = min(min_x, traj_min_x)
        min_y = min(min_y, traj_min_y)
        max_x = max(max_x, traj_max_x)
        max_y = max(max_y, traj_max_y)
        
    con_trajs_plot = visualization.contour_plot_with_tours(f, x_trajs, labels, [min_x - 0.1, max_x + 0.1], [min_y - 0.1, max_y + 0.1], 1000, 1000, 'RdBu')
    plt.savefig('hw-9/visualization/descent_trajectories.png')
    plt.show()
    
    val_trajs_plot = visualization.val_descent(val_trajs, labels)
    plt.savefig('hw-9/visualization/value_descent.png')
    plt.show()
        
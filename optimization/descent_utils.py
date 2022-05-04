'''
Author: Aiden Li
Date: 2022-05-03 17:58:29
LastEditors: Aiden Li (i@aidenli.net)
LastEditTime: 2022-05-04 22:31:30
Description: Choosing the descent direction.
'''
import numpy as np

def get_step_size(x, grad, fn, grad_fn, hessian_fn, points=1000, mode='even'):
    if mode == 'even':
        step_sizes = np.linspace(0, 1, points)
        dxs = grad[:, np.newaxis] @ step_sizes[np.newaxis, :]
        xs = x[:, np.newaxis].repeat(points, axis=1) + grad[:, np.newaxis] @ step_sizes[np.newaxis, :]
        min_step_idx = np.argmin(fn(xs))
        return step_sizes[min_step_idx]
    elif mode == 'fixed':
        return 0.05
    else:
        raise NotImplementedError()
    
def descent_1n(x, fn, grad_fn, hessian_fn, last_x=None):
    grad = grad_fn(x)
    descent_dim = np.argmax(np.abs(grad))
    
    sd = np.zeros_like(x)
    sd[descent_dim] = - grad[descent_dim]
    
    return grad, sd
    
def descent_2n(x, fn, grad_fn, hessian_fn, last_x=None):
    grad = grad_fn(x)
    Hf = hessian_fn(x)
    inv_Hf = np.linalg.inv(Hf)
    return grad, - inv_Hf @ grad

def descent_in(x, fn, grad_fn, hessian_fn, last_x=None):
    grad = grad_fn(x)
    d = np.sign(grad)
    return grad, grad * d

def descent_fr(x, fn, grad_fn, hessian_fn, last_x=None):
    grad = grad_fn(x)
    if last_x is None:
        alpha = 0.
        last_grad = 0.
    else:
        last_grad = grad_fn(last_x)
        alpha = np.linalg.norm(grad, 2) / np.linalg.norm(last_grad, 2)
    
    return grad, - grad + alpha * last_grad

def descent_pr(x, fn, grad_fn, hessian_fn, last_x=None):
    grad = grad_fn(x)
    if last_x is None:
        alpha = 0.
        last_grad = 0.
    else:
        last_grad = grad_fn(last_x)
        alpha = np.dot(grad, grad - last_grad) / np.linalg.norm(last_grad, 2)
    
    return grad, - grad + alpha * last_grad

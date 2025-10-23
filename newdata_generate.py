#%%
import jax
import jax.numpy as jnp
from jax import random
from time import time
import optax
from jax import value_and_grad
from functools import partial
from jax import jit
from tqdm import tqdm
from typing import Any
from flax import struct
from flax.serialization import to_state_dict, from_state_dict
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.interpolate import interpn
import h5py
from scipy.io import loadmat
import argparse
from Tecplot_mesh import tecplot_Mesh
from tqdm import tqdm
#%%
class Model(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)

class PINNbase:
    def __init__(self,c):
        c.get_outdirs()
        self.c=c

class PINN(PINNbase):
    def test(self):
        all_params = {"domain":{}, "data":{}, "network":{}, "problem":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        global_key = random.PRNGKey(42)
        all_params["network"] = self.c.network.init_params(**self.c.network_init_kwargs)
        all_params["problem"] = self.c.problem.init_params(**self.c.problem_init_kwargs)
        optimiser = self.c.optimization_init_kwargs["optimiser"](self.c.optimization_init_kwargs["learning_rate"])
        grids, all_params = self.c.domain.sampler(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        time_n = np.unique(train_data['pos'][:,0])
        #all_params = self.c.problem.constraints(all_params)
        #valid_data = self.c.problem.exact_solution(all_params)
        model_fn = c.network.network_fn
        return all_params, model_fn, train_data, time_n
    
def equ_func1(all_params, g_batch, cotangent, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)
    out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
    return out, out_t

def equ_func2(all_params, g_batch, cotangent1, cotangent2, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)
    def u_tt(batch):
        return jax.jvp(u_t,(batch,), (cotangent1, ))[1]
    out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent2,))
    return out_x, out_xx

def equ_func3(all_params, g_batch, cotangent1, cotangent2, cotangent3, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)
    def u_tt(batch):
        return jax.jvp(u_t,(batch,), (cotangent1, ))[1]
    def u_ttt(batch):
        return jax.jvp(u_tt,(batch,), (cotangent2, ))[1]
    out_xx, out_xxx = jax.jvp(u_ttt, (g_batch,), (cotangent3,))
    return out_xx, out_xxx

def Derivatives(dynamic_params, all_params, g_batch, model_fns):
    keys = ['u_ref', 'v_ref', 'w_ref', 'p_ref', 'T_ref']

    all_params["network"]["layers"] = dynamic_params
    out_xx, out_xxx = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),
                                                     jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),
                                                     jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)), model_fns)
    out_xy, out_xxy = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),
                                                     jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),
                                                     jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)), model_fns)
    out_xz, out_xxz = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),
                                                     jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),
                                                     jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)), model_fns)
    
    _, out_xyy = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)), model_fns)
    out_yy, out_yyy = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),
                                                     jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),
                                                     jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)), model_fns)
    out_yz, out_yyz = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),
                                                     jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),
                                                     jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)), model_fns)
    
    _, out_xzz = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)), model_fns)
    _, out_yzz = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),
                                                jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)), model_fns)
    out_zz, out_zzz = equ_func3(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),
                                                     jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),
                                                     jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)), model_fns)
    
    out_x, out_xt = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),
                                                   jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_y, out_yt = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),
                                                   jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_z, out_zt = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),
                                                   jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out = model_fns(all_params, g_batch)
    uvwp = np.concatenate([out[:,k:(k+1)]*all_params["data"][keys[k]] for k in range(len(keys))],1)
    uvwp[:,-2] = 1.185*uvwp[:,-2]

    uxs = np.concatenate([out_x[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,1] for k in range(len(keys))],1)
    uys = np.concatenate([out_y[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,2] for k in range(len(keys))],1)
    uzs = np.concatenate([out_z[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,3] for k in range(len(keys))],1)
    uxts = np.concatenate([out_xt[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,1]/all_params["domain"]["in_max"][0,0] for k in range(len(keys))],1)
    uyts = np.concatenate([out_yt[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,2]/all_params["domain"]["in_max"][0,0] for k in range(len(keys))],1)
    uzts = np.concatenate([out_zt[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,3]/all_params["domain"]["in_max"][0,0] for k in range(len(keys))],1)
    uxxs = np.concatenate([out_xx[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,1]/all_params["domain"]["in_max"][0,1] for k in range(len(keys))],1)
    uxys = np.concatenate([out_xy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,1]/all_params["domain"]["in_max"][0,2] for k in range(len(keys))],1)
    uxzs = np.concatenate([out_xz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,1]/all_params["domain"]["in_max"][0,3] for k in range(len(keys))],1)
    uyys = np.concatenate([out_yy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,2]/all_params["domain"]["in_max"][0,2] for k in range(len(keys))],1)
    uyzs = np.concatenate([out_yz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,2]/all_params["domain"]["in_max"][0,3] for k in range(len(keys))],1)
    uzzs = np.concatenate([out_zz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,3]/all_params["domain"]["in_max"][0,3] for k in range(len(keys))],1)
    uxxxs = np.concatenate([out_xxx[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,1]/all_params["domain"]["in_max"][0,1]/all_params["domain"]["in_max"][0,1] for k in range(len(keys))],1)
    uxxys = np.concatenate([out_xxy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,1]/all_params["domain"]["in_max"][0,1]/all_params["domain"]["in_max"][0,2] for k in range(len(keys))],1)
    uxxzs = np.concatenate([out_xxz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,1]/all_params["domain"]["in_max"][0,1]/all_params["domain"]["in_max"][0,3] for k in range(len(keys))],1)
    uxyys = np.concatenate([out_xyy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,1]/all_params["domain"]["in_max"][0,2]/all_params["domain"]["in_max"][0,2] for k in range(len(keys))],1)
    uyyys = np.concatenate([out_yyy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,2]/all_params["domain"]["in_max"][0,2]/all_params["domain"]["in_max"][0,2] for k in range(len(keys))],1)
    uyyzs = np.concatenate([out_yyz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,2]/all_params["domain"]["in_max"][0,2]/all_params["domain"]["in_max"][0,3] for k in range(len(keys))],1)
    uxzzs = np.concatenate([out_xzz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,1]/all_params["domain"]["in_max"][0,3]/all_params["domain"]["in_max"][0,3] for k in range(len(keys))],1)
    uyzzs = np.concatenate([out_yzz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,2]/all_params["domain"]["in_max"][0,3]/all_params["domain"]["in_max"][0,3] for k in range(len(keys))],1)
    uzzzs = np.concatenate([out_zzz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,3]/all_params["domain"]["in_max"][0,3]/all_params["domain"]["in_max"][0,3] for k in range(len(keys))],1)


    Ty = uyts[:,2:3]-uzts[:,1:2] + uvwp[:,0:1]*(uxys[:,2:3] - uxzs[:,1:2]) + uvwp[:,1:2]*(uyys[:,2:3] - uyzs[:,1:2]) + uvwp[:,2:3]*(uyzs[:,2:3] - uzzs[:,1:2]) \
         - (uys[:,2:3] - uzs[:,1:2])*uxs[:,0:1] - (uzs[:,0:1] - uxs[:,2:3])*uys[:,0:1] - (uxs[:,1:2] - uys[:,0:1])*uzs[:,0:1] \
         - all_params["data"]["viscosity"]*(uxxys[:,2:3] + uyyys[:,2:3] + uyzzs[:,2:3] - uxxzs[:,1:2] - uyyzs[:,1:2] - uzzzs[:,1:2])
    
    Tx = (uys[:,2:3] - uzs[:,1:2])*uxs[:,1:2] + (uzs[:,0:1] - uxs[:,2:3])*uys[:,1:2] + (uxs[:,1:2] - uys[:,0:1])*uzs[:,1:2] \
         + all_params["data"]["viscosity"]*(uxxzs[:,0:1] + uyyzs[:,0:1] + uzzzs[:,0:1] - uxxxs[:,2:3] - uxyys[:,2:3] - uxzzs[:,2:3]) \
         - uzts[:,0:1] + uxts[:,2:3] - uvwp[:,0:1]*(uxzs[:,0:1] - uxxs[:,2:3]) - uvwp[:,1:2]*(uyzs[:,0:1] - uxys[:,2:3]) - uvwp[:,2:3]*(uzzs[:,0:1] - uxzs[:,2:3])
    return uvwp, uxs[:,4:5], uys[:,4:5], Tx, Ty

def Tecplotfile_gen(path, name, all_params, domain_range, output_shape, order, timestep, is_ground, is_mean, train_data, time_n, model_fn):
    
    # Load the parameters
    pos_ref = all_params["domain"]["in_max"].flatten()
    dynamic_params = all_params["network"].pop("layers")
    counts = np.unique(train_data['pos'][:,0], return_counts=True)[1]
    eval_grid = train_data['pos'][np.sum(counts[:timestep]):np.sum(counts[:timestep+1]),:]
    eval_grid_e = [eval_grid.copy()*pos_ref[i] for i in range(4)]
    # Evaluate the derivatives
    uvwp, Txo, Tyo, Tx, Ty = zip(*[Derivatives(dynamic_params, all_params, eval_grid[i:i+10000], model_fn)
                                        for i in range(0, eval_grid.shape[0], 10000)])
    
    # Concatenate the results
    uvwp = np.concatenate(uvwp, axis=0)
    Txo = np.concatenate(Txo, axis=0)
    Tyo = np.concatenate(Tyo, axis=0)
    Tx = np.concatenate(Tx, axis=0)
    Ty = np.concatenate(Ty, axis=0)
    uvwp[:,3] = uvwp[:,3] - np.mean(uvwp[:,3])
    timestep_new = timestep + 170
    if os.path.isdir(path + '/' + name):
        pass
    else:
        print('check')
        os.mkdir(path + 'npydata/lv6_T')
    np.save(path + 'npdata/lv6_T/' + f'/flow0_{timestep_new:03d}' + '.npy', np.concatenate([eval_grid_e, train_data['vel'], Tx, Ty], axis=1))
#%%
if __name__ == "__main__":
    from domain import *
    from trackdata import *
    from network import *
    from constants import *
    from problem import *
    from txt_reader import *
    import os
    parser = argparse.ArgumentParser(description='PINN')
    parser.add_argument('-f', '--foldername', type=str, help='foldername', default='HIT')
    parser.add_argument('-c', '--config', type=str, help='configuration', default='eval_config')
    args = parser.parse_args()

    # Get evaluation configuration
    cur_dir = os.getcwd()
    config_txt = cur_dir + '/' + args.config + '.txt'
    data = parse_tree_structured_txt(config_txt)

    # Get model constants
    with open(os.path.dirname(cur_dir)+ '/' + data['path'] + args.foldername +'/summary/constants.pickle','rb') as f:
        constants = pickle.load(f)
    values = list(constants.values())

    c = Constants(run = values[0],
                domain_init_kwargs = values[1],
                data_init_kwargs = values[2],
                network_init_kwargs = values[3],
                problem_init_kwargs = values[4],
                optimization_init_kwargs = values[5],
                equation_init_kwargs = values[6],)
    run = PINN(c)

    # Get model parameters
    checkpoint_list = sorted(glob(run.c.model_out_dir+'/*.pkl'), key=lambda x: int(x.split('_')[-1].split('.')[0]))

    with open(checkpoint_list[-1],"rb") as f:
        model_params = pickle.load(f)
    all_params, model_fn, train_data, time_n = run.test()
    model = Model(all_params["network"]["layers"], model_fn)
    all_params["network"]["layers"] = from_state_dict(model, model_params).params
    domain_range = data['tecplot_init_kwargs']['domain_range']
    output_shape = data['tecplot_init_kwargs']['out_shape']
    order = data['tecplot_init_kwargs']['order']
    timesteps = data['tecplot_init_kwargs']['timestep']
    is_ground = data['tecplot_init_kwargs']['is_ground']
    path = data['tecplot_init_kwargs']['path']
    is_mean = data['tecplot_init_kwargs']['is_mean']
    path = os.path.dirname(cur_dir) + '/' + path
    pos_ref = all_params["domain"]["in_max"].flatten()
    for timestep in timesteps:
        Tecplotfile_gen(path, args.foldername, all_params, domain_range, output_shape, order, timestep, is_ground, is_mean, train_data, time_n, model_fn)
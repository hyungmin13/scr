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

class Model(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)

class PINNbase:
    def __init__(self,c):
        c.get_outdirs()
        c.save_constants_file()
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
        valid_data = self.c.problem.exact_solution(all_params)
        model_fn = c.network.network_fn
        return all_params, model_fn, train_data, valid_data, grids

def equ_func(all_params, g_batch, cotangent, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)
    def u_tt(batch):
        return jax.jvp(u_t,(batch,), (cotangent, ))[1]
    out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
    return out_x, out_xx

def equ_func2(all_params, g_batch, cotangent, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)
    out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
    return out, out_t

def u_loss(dynamic_params, all_params, p_batch, v_batch, model_fns):
    keys = ['u_ref', 'v_ref', 'w_ref', 'u_ref']
    all_params['network']['layers'] = dynamic_params
    p_out = model_fns(all_params, p_batch)
    loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - v_batch[:,0:1]
    loss_u = jnp.mean(loss_u**2)
    return loss_u

def v_loss(dynamic_params, all_params, p_batch, v_batch, model_fns):
    keys = ['u_ref', 'v_ref', 'w_ref', 'u_ref']
    all_params['network']['layers'] = dynamic_params
    p_out = model_fns(all_params, p_batch)
    loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - v_batch[:,1:2]
    loss_v = jnp.mean(loss_v**2)
    return loss_v

def w_loss(dynamic_params, all_params, p_batch, v_batch, model_fns):
    keys = ['u_ref', 'v_ref', 'w_ref', 'u_ref']
    all_params['network']['layers'] = dynamic_params
    p_out = model_fns(all_params, p_batch)
    loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - v_batch[:,2:3]
    loss_w = jnp.mean(loss_w**2)
    return loss_w

def con(dynamic_params, all_params, g_batch, model_fns):
    def first_order(all_params, g_batch, cotangent, model_fns):
        def u_t(batch):
            return model_fns(all_params, batch)
        out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
        return out, out_t

    all_params["network"]["layers"] = dynamic_params

    out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    _, out_x = first_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    _, out_y = first_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    _, out_z = first_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)

    ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
    vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
    wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
    loss_con = ux + vy + wz
    loss_con = jnp.mean(loss_con**2)
    return loss_con

def NS1(dynamic_params, all_params, g_batch, model_fns):
    def first_order(all_params, g_batch, cotangent, model_fns):
        def u_t(batch):
            return model_fns(all_params, batch)
        out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
        return out, out_t

    def second_order(all_params, g_batch, cotangent, model_fns):
        def u_t(batch):
            return model_fns(all_params, batch)
        def u_tt(batch):
            return jax.jvp(u_t,(batch,), (cotangent, ))[1]
        out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
        return out_x, out_xx
    all_params["network"]["layers"] = dynamic_params

    out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_x, out_xx = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_y, out_yy = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_z, out_zz = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)
    u = all_params["data"]['u_ref']*out[:,0:1]
    v = all_params["data"]['v_ref']*out[:,1:2]
    w = all_params["data"]['w_ref']*out[:,2:3]
    ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
    ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
    uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
    uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
    px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]
    uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
    uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
    uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
    loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)-2.22*10**(-1)/(3*0.43685**2)*u
    loss_NS1 = jnp.mean(loss_NS1**2)
    return loss_NS1

def NS2(dynamic_params, all_params, g_batch, model_fns):
    def first_order(all_params, g_batch, cotangent, model_fns):
        def u_t(batch):
            return model_fns(all_params, batch)
        out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
        return out, out_t

    def second_order(all_params, g_batch, cotangent, model_fns):
        def u_t(batch):
            return model_fns(all_params, batch)
        def u_tt(batch):
            return jax.jvp(u_t,(batch,), (cotangent, ))[1]
        out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
        return out_x, out_xx
    all_params["network"]["layers"] = dynamic_params

    out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_x, out_xx = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_y, out_yy = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_z, out_zz = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)
    u = all_params["data"]['u_ref']*out[:,0:1]
    v = all_params["data"]['v_ref']*out[:,1:2]
    w = all_params["data"]['w_ref']*out[:,2:3]
    vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
    vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
    vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
    vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
    py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]
    vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
    vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
    vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
    loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)-2.22*10**(-1)/(3*0.43685**2)*v
    loss_NS2 = jnp.mean(loss_NS2**2)
    return loss_NS2

def NS3(dynamic_params, all_params, g_batch, model_fns):
    def first_order(all_params, g_batch, cotangent, model_fns):
        def u_t(batch):
            return model_fns(all_params, batch)
        out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
        return out, out_t

    def second_order(all_params, g_batch, cotangent, model_fns):
        def u_t(batch):
            return model_fns(all_params, batch)
        def u_tt(batch):
            return jax.jvp(u_t,(batch,), (cotangent, ))[1]
        out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
        return out_x, out_xx
    all_params["network"]["layers"] = dynamic_params

    out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_x, out_xx = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_y, out_yy = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_z, out_zz = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)
    u = all_params["data"]['u_ref']*out[:,0:1]
    v = all_params["data"]['v_ref']*out[:,1:2]
    w = all_params["data"]['w_ref']*out[:,2:3]
    wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]
    wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
    wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
    wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
    pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]
    wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2
    wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2
    wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2
    loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz)-2.22*10**(-1)/(3*0.43685**2)*w
    loss_NS3 = jnp.mean(loss_NS3**2)
    return loss_NS3

def total_loss(dynamic_params, all_params, g_batch, particles, particle_vel, model_fns):
        def first_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
            return out, out_t

        def second_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            def u_tt(batch):
                return jax.jvp(u_t,(batch,), (cotangent, ))[1]
            out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
            return out_x, out_xx
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
        out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_x, out_xx = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_y, out_yy = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_z, out_zz = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)

        p_out = model_fns(all_params, particles)                                                                           

        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]

        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]

        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)-2.22*10**(-1)/(3*0.43685**2)*u
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)-2.22*10**(-1)/(3*0.43685**2)*v
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz)-2.22*10**(-1)/(3*0.43685**2)*w
        loss_NS3 = jnp.mean(loss_NS3**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 
        return total_loss

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
    domain_range = data['tecplot_init_kwargs']['domain_range']
    output_shape = data['tecplot_init_kwargs']['out_shape']
    order = data['tecplot_init_kwargs']['order']
    timesteps = data['tecplot_init_kwargs']['timestep']
    is_ground = data['tecplot_init_kwargs']['is_ground']
    path = data['tecplot_init_kwargs']['path']
    is_mean = data['tecplot_init_kwargs']['is_mean']
    path = os.path.dirname(cur_dir) + '/' + path
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
    all_params, model_fn, train_data, valid_data, grids = run.test()
    checkpoint_list = sorted(glob(run.c.model_out_dir+'/*.pkl'), key=lambda x: int(x.split('_')[-1].split('.')[0]))

    global_key = random.PRNGKey(42)
    key, network_key = random.split(global_key)
    key, batch_key = random.split(key)
    num_keysplit = 10
    keys = random.split(batch_key, num = num_keysplit)
    keys_split = [random.split(keys[i], num = c.optimization_init_kwargs["n_steps"]) for i in range(num_keysplit)]
    keys_iter = [iter(keys_split[i]) for i in range(num_keysplit)]
    keys_next = [next(keys_iter[i]) for i in range(num_keysplit)]
    p_batch = random.choice(keys_next[0],train_data['pos'],shape=(c.optimization_init_kwargs["p_batch"],))
    v_batch = random.choice(keys_next[0],train_data['vel'],shape=(c.optimization_init_kwargs["p_batch"],))
    g_batch = jnp.stack([random.choice(keys_next[k+1], 
                                        grids['eqns'][arg], 
                                        shape=(c.optimization_init_kwargs["e_batch"],)) 
                            for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
    b_batches = []
    m = 0
    for b_key in all_params["domain"]["bound_keys"]:
        b_batch = jnp.stack([random.choice(keys_next[k+5], 
                                        grids[b_key][arg], 
                                        shape=(c.optimization_init_kwargs["e_batch"],)) 
                            for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
        b_batches.append(b_batch)
    intra_list = []
    inter_list = []
    total_grad_list = []
    for i in tqdm(range(330000)):
        keys_next = [next(keys_iter[i]) for i in range(num_keysplit)]
        
        if (i<100) or (300000<i<300100):
            p_batch = random.choice(keys_next[0],train_data['pos'],shape=(c.optimization_init_kwargs["p_batch"],))
            v_batch = random.choice(keys_next[0],train_data['vel'],shape=(c.optimization_init_kwargs["p_batch"],))
            g_batch = jnp.stack([random.choice(keys_next[k+1], 
                                            grids['eqns'][arg], 
                                            shape=(c.optimization_init_kwargs["e_batch"],)) 
                                for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
            b_batches = []
            for b_key in all_params["domain"]["bound_keys"]:
                b_batch = jnp.stack([random.choice(keys_next[k+5], 
                                                grids[b_key][arg], 
                                                shape=(c.optimization_init_kwargs["e_batch"],)) 
                                    for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
                b_batches.append(b_batch)
            with open(checkpoint_list[m],"rb") as f:
                model_params = pickle.load(f)
            model = Model(all_params["network"]["layers"], model_fn)
            all_params["network"]["layers"] = from_state_dict(model, model_params).params
            dynamic_params = all_params["network"].pop("layers")
            _, u_loss_grad = value_and_grad(u_loss,argnums=0)(dynamic_params, all_params, p_batch, v_batch, model_fn)
            _, v_loss_grad = value_and_grad(v_loss,argnums=0)(dynamic_params, all_params, p_batch, v_batch, model_fn)
            _, w_loss_grad = value_and_grad(w_loss,argnums=0)(dynamic_params, all_params, p_batch, v_batch, model_fn)
            _, con_grad = value_and_grad(con,argnums=0)(dynamic_params, all_params, g_batch, model_fn)
            _, NS1_grad = value_and_grad(NS1,argnums=0)(dynamic_params, all_params, g_batch, model_fn)
            _, NS2_grad = value_and_grad(NS2,argnums=0)(dynamic_params, all_params, g_batch, model_fn)
            _, NS3_grad = value_and_grad(NS3,argnums=0)(dynamic_params, all_params, g_batch, model_fn)
            _, total_grad = value_and_grad(total_loss,argnums=0)(dynamic_params, all_params, g_batch, p_batch, v_batch, model_fn)
            #print(len(u_loss_grad[0]))
            
            #print(u_loss_grad[1][0].shape)
            u_loss_grad = jnp.concatenate([jnp.ravel(u_loss_grad[i][j]) for i in range(len(u_loss_grad)) for j in range(len(u_loss_grad[i]))])
            v_loss_grad = jnp.concatenate([jnp.ravel(v_loss_grad[i][j]) for i in range(len(v_loss_grad)) for j in range(len(v_loss_grad[i]))])
            w_loss_grad = jnp.concatenate([jnp.ravel(w_loss_grad[i][j]) for i in range(len(w_loss_grad)) for j in range(len(w_loss_grad[i]))])
            con_grad = jnp.concatenate([jnp.ravel(con_grad[i][j]) for i in range(len(con_grad)) for j in range(len(con_grad[i]))])
            NS1_grad = jnp.concatenate([jnp.ravel(NS1_grad[i][j]) for i in range(len(NS1_grad)) for j in range(len(NS1_grad[i]))])
            NS2_grad = jnp.concatenate([jnp.ravel(NS2_grad[i][j]) for i in range(len(NS2_grad)) for j in range(len(NS2_grad[i]))])
            NS3_grad = jnp.concatenate([jnp.ravel(NS3_grad[i][j]) for i in range(len(NS3_grad)) for j in range(len(NS3_grad[i]))])
            total_grad = jnp.concatenate([jnp.ravel(total_grad[i][j]) for i in range(len(total_grad)) for j in range(len(total_grad[i]))])
            intra_score = 2*(jnp.linalg.norm((u_loss_grad/jnp.linalg.norm(u_loss_grad)+v_loss_grad/jnp.linalg.norm(v_loss_grad)+
                                             w_loss_grad/jnp.linalg.norm(w_loss_grad)+con_grad/jnp.linalg.norm(con_grad)+
                                             NS1_grad/jnp.linalg.norm(NS1_grad)+NS2_grad/jnp.linalg.norm(NS2_grad)+
                                             NS3_grad/jnp.linalg.norm(NS3_grad))/7)**2)-1
            print('check')
            if i > 0:
                inter_score = 2*(jnp.linalg.norm((total_grad/jnp.linalg.norm(total_grad)+total_grad_temp/jnp.linalg.norm(total_grad_temp))/2)**2)-1
                inter_list.append(inter_score)
                intra_list.append(intra_score)
            total_grad_temp = total_grad.copy()
            
            
            m = m + 1
    intra_scores = np.array(intra_list)
    inter_scores = np.array(inter_list)
    if os.isdir(os.path.dirname(cur_dir)+ '/' + data['path'] + args.foldername + '/alignment')==False:
        os.mkdir(os.path.dirname(cur_dir)+ '/' + data['path'] + args.foldername + '/alignment')
    else:
        pass
    scores = np.concatenate([intra_scores.reshape(-1,1), inter_scores.reshape(-1,1)],axis=1)
    np.save(os.path.dirname(cur_dir)+ '/' + data['path'] + args.foldername + '/alignment' + '/alignment_scores.npy',scores)

        

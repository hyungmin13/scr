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
from soap_jax import soap
class Model(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)

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

def PINN_loss(dynamic_params, all_params, g_batch, particles, particle_vel, model_fns):
    all_params["network"]["layers"] = dynamic_params
    weights = all_params["problem"]["loss_weights"]
    out, out_t = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_x, out_xx = equ_func(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_y, out_yy = equ_func(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_z, out_zz = equ_func(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)

    p_out = model_fns(all_params, particles)

    #out, out_t, out_x, out_xx, out_y, out_yy, out_z, out_zz = eqn_func(g_batch)
    u = all_params["data"]['u_ref']*out[:,0:1]
    v = all_params["data"]['v_ref']*out[:,1:2]
    w = all_params["data"]['w_ref']*out[:,2:3]

    ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["data"]["domain_range"]["t"][1]
    vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["data"]["domain_range"]["t"][1]
    wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["data"]["domain_range"]["t"][1]

    ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["data"]["domain_range"]["x"][1]
    vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["data"]["domain_range"]["x"][1]
    wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["data"]["domain_range"]["x"][1]
    px = all_params["data"]['u_ref']*out_x[:,3:4]/all_params["data"]["domain_range"]["x"][1]

    uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["data"]["domain_range"]["y"][1]
    vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["data"]["domain_range"]["y"][1]
    wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["data"]["domain_range"]["y"][1]
    py = all_params["data"]['u_ref']*out_y[:,3:4]/all_params["data"]["domain_range"]["y"][1]

    uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["data"]["domain_range"]["z"][1]
    vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["data"]["domain_range"]["z"][1]
    wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["data"]["domain_range"]["z"][1]
    pz = all_params["data"]['u_ref']*out_z[:,3:4]/all_params["data"]["domain_range"]["z"][1]

    uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["data"]["domain_range"]["x"][1]**2
    vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["data"]["domain_range"]["x"][1]**2
    wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["data"]["domain_range"]["x"][1]**2

    uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["data"]["domain_range"]["y"][1]**2
    vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["data"]["domain_range"]["y"][1]**2
    wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["data"]["domain_range"]["y"][1]**2

    uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["data"]["domain_range"]["z"][1]**2
    vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["data"]["domain_range"]["z"][1]**2
    wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["data"]["domain_range"]["z"][1]**2
       

    loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
    loss_u = jnp.mean(loss_u**2)

    loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
    loss_v = jnp.mean(loss_v**2)

    loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
    loss_w = jnp.mean(loss_w**2)
    
    
    loss_con = ux + vy + wz
    loss_con = jnp.mean(loss_con**2)
    loss_NS1 = ut + u*ux + v*uy + w*uz + px -\
               all_params["data"]["viscosity"]*(uxx+uyy+uzz)-2.22*10**(-1)/(3*0.43685**2)*u
    loss_NS1 = jnp.mean(loss_NS1**2)
    loss_NS2 = vt + u*vx + v*vy + w*vz + py -\
               all_params["data"]["viscosity"]*(vxx+vyy+vzz)-2.22*10**(-1)/(3*0.43685**2)*v
    loss_NS2 = jnp.mean(loss_NS2**2)
    loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz)-2.22*10**(-1)/(3*0.43685**2)*w

    loss_NS3 = jnp.mean(loss_NS3**2)
    total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                 weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                 weights[6]*loss_NS3 
    return total_loss

@partial(jax.jit, static_argnums=(1, 4, 8))
def PINN_update(model_states, optimiser_fn, dynamic_params, static_params, static_keys, grids, particles, particle_vel, model_fn):
    static_leaves, treedef = static_keys
    leaves = [d if s is None else s for d, s in zip(static_params, static_leaves)]
    all_params = jax.tree_util.tree_unflatten(treedef, leaves)
    lossval, grads = value_and_grad(PINN_loss, argnums=0)(dynamic_params, all_params, grids, particles, particle_vel, model_fn)
    updates, model_states = optimiser_fn(grads, model_states, dynamic_params)
    dynamic_params = optax.apply_updates(dynamic_params, updates)
    return lossval, model_states, dynamic_params
class PINNbase:
    def __init__(self,c):
        c.get_outdirs()
        c.save_constants_file()
        self.c=c
class PINN(PINNbase):
    def train(self):
        all_params = {"domain":{}, "data":{}, "network":{}, "problem":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        global_key = random.PRNGKey(42)
        key, network_key = random.split(global_key)
        all_params["network"] = self.c.network.init_params(**self.c.network_init_kwargs)
        all_params["problem"] = self.c.problem.init_params(**self.c.problem_init_kwargs)
        learn_rate = optax.exponential_decay(self.c.optimization_init_kwargs["learning_rate"],16000,0.9)
        optimiser = self.c.optimization_init_kwargs["optimiser"](learning_rate=learn_rate, b1=0.95, b2=0.95,
                                                                 weight_decay=0.01, precondition_frequency=5)
        grids, all_params = self.c.domain.sampler(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        all_params = self.c.problem.constraints(all_params)
        valid_data = self.c.problem.exact_solution(all_params)
        model_states = optimiser.init(all_params["network"]["layers"])
        optimiser_fn = optimiser.update
        model_fn = c.network.network_fn
        dynamic_params = all_params["network"].pop("layers")
        key, batch_key = random.split(key)
        g_batch_key1, g_batch_key2, g_batch_key3, g_batch_key4, p_batch_key, b_batch_key, e_batch_key = random.split(batch_key, num = 7)
        g_batch_keys1 = random.split(g_batch_key1, num = self.c.optimization_init_kwargs["n_steps"])
        g_batch_keys2 = random.split(g_batch_key2, num = self.c.optimization_init_kwargs["n_steps"])
        g_batch_keys3 = random.split(g_batch_key3, num = self.c.optimization_init_kwargs["n_steps"])
        g_batch_keys4 = random.split(g_batch_key4, num = self.c.optimization_init_kwargs["n_steps"])

        p_batch_keys = random.split(p_batch_key, num = self.c.optimization_init_kwargs["n_steps"])
        b_batch_keys = random.split(b_batch_key, num = self.c.optimization_init_kwargs["n_steps"])
        e_batch_keys = random.split(e_batch_key, num = self.c.optimization_init_kwargs["n_steps"]//2000)

        p_batch_keys = iter(p_batch_keys)
        g_batch_keys1 = iter(g_batch_keys1)
        g_batch_keys2 = iter(g_batch_keys2)
        g_batch_keys3 = iter(g_batch_keys3)
        g_batch_keys4 = iter(g_batch_keys4)
        b_batch_keys = iter(b_batch_keys)
        e_batch_keys = iter(e_batch_keys)
        leaves, treedef = jax.tree_util.tree_flatten(all_params)
        ab = tuple(x if isinstance(x,(np.ndarray, jnp.ndarray)) else None for x in leaves)
        ac = tuple(None if isinstance(x,(np.ndarray, jnp.ndarray)) else x for x in leaves)
        ad = (ac, treedef)

        p_key = next(p_batch_keys)
        g_key1 = next(g_batch_keys1)
        g_key2 = next(g_batch_keys2)
        g_key3 = next(g_batch_keys3)
        g_key4 = next(g_batch_keys4)

        p_batch = random.choice(p_key,train_data['pos'],shape=(self.c.optimization_init_kwargs["p_batch"],))
        v_batch = random.choice(p_key,train_data['vel'],shape=(self.c.optimization_init_kwargs["p_batch"],))

        g_batch = jnp.stack([random.choice(g_key1,grids['eqns']['t'],shape=(self.c.optimization_init_kwargs["e_batch"],)),
                            random.choice(g_key2,grids['eqns']['x'],shape=(self.c.optimization_init_kwargs["e_batch"],)),
                            random.choice(g_key3,grids['eqns']['y'],shape=(self.c.optimization_init_kwargs["e_batch"],)),
                            random.choice(g_key4,grids['eqns']['z'],shape=(self.c.optimization_init_kwargs["e_batch"],))],axis=1)

        update = PINN_update.lower(model_states, optimiser_fn, dynamic_params, ab, ad, g_batch, p_batch, v_batch, model_fn).compile()
        
        for i in range(self.c.optimization_init_kwargs["n_steps"]):
            template = ("iteration {}, loss_val {}")
            p_key = next(p_batch_keys)
            g_key1 = next(g_batch_keys1)
            g_key2 = next(g_batch_keys2)
            g_key3 = next(g_batch_keys3)
            g_key4 = next(g_batch_keys4)

            p_batch = random.choice(p_key,train_data['pos'],shape=(self.c.optimization_init_kwargs["p_batch"],))
            v_batch = random.choice(p_key,train_data['vel'],shape=(self.c.optimization_init_kwargs["p_batch"],))
            g_batch = jnp.stack([random.choice(g_key1,grids['eqns']['t'],shape=(self.c.optimization_init_kwargs["e_batch"],)),
                                random.choice(g_key2,grids['eqns']['x'],shape=(self.c.optimization_init_kwargs["e_batch"],)),
                                random.choice(g_key3,grids['eqns']['y'],shape=(self.c.optimization_init_kwargs["e_batch"],)),
                                random.choice(g_key4,grids['eqns']['z'],shape=(self.c.optimization_init_kwargs["e_batch"],))],axis=1)            
            lossval, model_states, dynamic_params = update(model_states, dynamic_params, ab, g_batch, p_batch, v_batch)
        
        
            self.report(i, model_states, dynamic_params, all_params, p_batch, v_batch, valid_data, e_batch_keys, model_fn)

    def test(self):
        all_params = {"domain":{}, "data":{}, "network":{}, "problem":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        global_key = random.PRNGKey(42)
        key, network_key = random.split(global_key)
        all_params["network"] = self.c.network.init_params(**self.c.network_init_kwargs)
        all_params["problem"] = self.c.problem.init_params(**self.c.problem_init_kwargs)
        optimiser = self.c.optimization_init_kwargs["optimiser"](self.c.optimization_init_kwargs["learning_rate"])
        grids, all_params = self.c.domain.sampler(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        all_params = self.c.problem.constraints(all_params)
        valid_data = self.c.problem.exact_solution(all_params)
        model_states = optimiser.init(all_params["network"]["layers"])
        optimiser_fn = optimiser.update
        model_fn = c.network.network_fn
        return all_params, model_fn, train_data, valid_data
    def report(self, i, model_states, dynamic_params, all_params, p_batch, v_batch, valid_data, e_batch_key, model_fns):
        model_save = (i % 20000 == 0)
        if model_save:
            all_params["network"]["layers"] = dynamic_params
            e_key = next(e_batch_key)
            e_batch_pos = random.choice(e_key, valid_data['pos'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            e_batch_vel = random.choice(e_key, valid_data['vel'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            v_pred2 = model_fns(all_params, e_batch_pos)
            p_new = 1.185*all_params["data"]["u_ref"]*v_pred2[:,3:]-(jnp.mean(1.185*all_params["data"]["u_ref"]*v_pred2[:,3:] - e_batch_vel[:,3:]))
            u_error = jnp.sqrt(jnp.mean((all_params["data"]["u_ref"]*v_pred2[:,0:1] - e_batch_vel[:,0:1])**2)/jnp.mean(e_batch_vel[:,0:1]**2))
            v_error = jnp.sqrt(jnp.mean((all_params["data"]["v_ref"]*v_pred2[:,1:2] - e_batch_vel[:,1:2])**2)/jnp.mean(e_batch_vel[:,1:2]**2))
            w_error = jnp.sqrt(jnp.mean((all_params["data"]["w_ref"]*v_pred2[:,2:3] - e_batch_vel[:,2:3])**2)/jnp.mean(e_batch_vel[:,2:3]**2))
            p_error = jnp.sqrt(jnp.mean((p_new - e_batch_vel[:,3:])**2)/jnp.mean(e_batch_vel[:,3:4]**2))
            v_pred = model_fns(all_params, p_batch)
            u_loss = jnp.mean((all_params["data"]["u_ref"]*v_pred[:,0:1] - v_batch[:,0:1])**2)
            v_loss = jnp.mean((all_params["data"]["v_ref"]*v_pred[:,1:2] - v_batch[:,1:2])**2)
            w_loss = jnp.mean((all_params["data"]["w_ref"]*v_pred[:,2:3] - v_batch[:,2:3])**2)
            model = Model(all_params["network"]["layers"], model_fns)
            serialised_model = to_state_dict(model)
            with open(self.c.model_out_dir + "saved_dic_"+str(i)+".pkl","wb") as f:
                pickle.dump(serialised_model,f)
            
            print(u_loss, v_loss, w_loss, u_error, v_error, w_error, p_error)

        return

#%%
if __name__=="__main__":
    from PINN_domain import *
    from PINN_trackdata import *
    from PINN_network import *
    from PINN_constants import *
    from PINN_problem import *
    from PINN_txt_reader import *
    import argparse
    
    parser = argparse.ArgumentParser(description='TBL_PINN')
    parser.add_argument('-n', '--name', type=str, help='run name', default='HIT_k1')
    parser.add_argument('-p', '--path', type=str, help='summary location', default='/ETFS/HIT/results/')
    args = parser.parse_args()
    cur_dir = os.getcwd()
    input_txt = os.path.dirname(cur_dir) + args.path + args.name + '/summary/' + args.name + '.txt' 
    data = parse_tree_structured_txt(input_txt)
    c = Constants(**data)
    print(c)
    print(c['network_init_kwargs']['key'])
    run = PINN(c)
    run.train()
    #all_params, model_fn, train_data, valid_data = run.test()


# %%

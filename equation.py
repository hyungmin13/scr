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

class Equation:
    @staticmethod
    def init_params(*args):
        raise NotImplementedError

class Boundless_flow(Equation):
    def __init__(self, all_params):
        self.all_params = all_params
    
    @staticmethod
    def Loss(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
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
    
    @staticmethod
    def Loss_report(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
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
        return total_loss, loss_u, loss_v, loss_w, loss_con, loss_NS1, loss_NS2, loss_NS3

class Boundless_flow_vmap(Equation):
    def __init__(self, all_params):
        self.all_params = all_params
    
    @staticmethod
    def Loss(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
        def jvp1(params, X, e, model_fns):
            return jax.vmap(lambda x: jax.jvp(lambda y: model_fns(params, y),
                                            (x,), (e,)))(X)
        def jvp2(params, X, e1, e2, model_fns):
            def g(y): return jax.jvp(lambda z: model_fns(params, z), (y,), (e1,))[1]
            return jax.vmap(lambda x: jax.jvp(g, (x,), (e2,)))(X)
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
        out, out_t = jvp1(all_params, g_batch, jnp.array([1.0, 0.0, 0.0, 0.0]),model_fns)
        out_x, out_xx = jvp2(all_params, g_batch, jnp.array([0.0, 1.0, 0.0, 0.0]),
                                                    jnp.array([0.0, 1.0, 0.0, 0.0]),model_fns)
        out_y, out_yy = jvp2(all_params, g_batch, jnp.array([0.0, 0.0, 1.0, 0.0]),
                                                    jnp.array([0.0, 0.0, 1.0, 0.0]),model_fns)
        out_z, out_zz = jvp2(all_params, g_batch, jnp.array([0.0, 0.0, 0.0, 1.0]),
                                                    jnp.array([0.0, 0.0, 0.0, 1.0]),model_fns)

        p_out, p_out_t = jvp1(all_params, particles, jnp.array([1.0, 0.0, 0.0, 0.0]),model_fns)

        u = all_params["data"]['u_ref']*out[:,0,0:1]
        v = all_params["data"]['v_ref']*out[:,0,1:2]
        w = all_params["data"]['w_ref']*out[:,0,2:3]
        p = all_params["data"]['p_ref']*out[:,0,3:4]

        ut = all_params["data"]['u_ref']*out_t[:,0,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,0,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,0,2:3]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,0,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,0,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,0,3:4]/all_params["domain"]["domain_range"]["x"][1]

        uy = all_params["data"]['u_ref']*out_y[:,0,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,0,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,0,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,0,3:4]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,0,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,0,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,0,3:4]/all_params["domain"]["domain_range"]["z"][1]

        uxx = all_params["data"]['u_ref']*out_xx[:,0,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,0,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,0,2:3]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,0,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,0,2:3]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,0,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,0,2:3]/all_params["domain"]["domain_range"]["z"][1]**2

        loss_u = all_params["data"]['u_ref']*p_out[:,0,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,0,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,0,2:3] - particle_vel[:,2:3]
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
    
    @staticmethod
    def Loss_report(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
        def jvp1(params, X, e, model_fns):
            return jax.vmap(lambda x: jax.jvp(lambda y: model_fns(params, y),
                                            (x,), (e,)))(X)
        def jvp2(params, X, e1, e2, model_fns):
            def g(y): return jax.jvp(lambda z: model_fns(params, z), (y,), (e1,))[1]
            return jax.vmap(lambda x: jax.jvp(g, (x,), (e2,)))(X)
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
        out, out_t = jvp1(all_params, g_batch, jnp.array([1.0, 0.0, 0.0, 0.0]),model_fns)
        out_x, out_xx = jvp2(all_params, g_batch, jnp.array([0.0, 1.0, 0.0, 0.0]),
                                                    jnp.array([0.0, 1.0, 0.0, 0.0]),model_fns)
        out_y, out_yy = jvp2(all_params, g_batch, jnp.array([0.0, 0.0, 1.0, 0.0]),
                                                    jnp.array([0.0, 0.0, 1.0, 0.0]),model_fns)
        out_z, out_zz = jvp2(all_params, g_batch, jnp.array([0.0, 0.0, 0.0, 1.0]),
                                                    jnp.array([0.0, 0.0, 0.0, 1.0]),model_fns)

        p_out, p_out_t = jvp1(all_params, particles, jnp.array([1.0, 0.0, 0.0, 0.0]),model_fns)

        u = all_params["data"]['u_ref']*out[:,0,0:1]
        v = all_params["data"]['v_ref']*out[:,0,1:2]
        w = all_params["data"]['w_ref']*out[:,0,2:3]
        p = all_params["data"]['p_ref']*out[:,0,3:4]

        ut = all_params["data"]['u_ref']*out_t[:,0,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,0,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,0,2:3]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,0,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,0,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,0,3:4]/all_params["domain"]["domain_range"]["x"][1]

        uy = all_params["data"]['u_ref']*out_y[:,0,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,0,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,0,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,0,3:4]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,0,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,0,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,0,3:4]/all_params["domain"]["domain_range"]["z"][1]

        uxx = all_params["data"]['u_ref']*out_xx[:,0,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,0,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,0,2:3]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,0,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,0,2:3]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,0,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,0,2:3]/all_params["domain"]["domain_range"]["z"][1]**2

        loss_u = all_params["data"]['u_ref']*p_out[:,0,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,0,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,0,2:3] - particle_vel[:,2:3]
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
        return total_loss, loss_u, loss_v, loss_w, loss_con, loss_NS1, loss_NS2, loss_NS3

class Boundless_flow_xgrad(Equation):
    def __init__(self, all_params):
        self.all_params = all_params
    
    @staticmethod
    def Loss(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
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
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz)
        loss_NS3 = jnp.mean(loss_NS3**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 
        return total_loss
    
    @staticmethod
    def Loss_report(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
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
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz)
        loss_NS3 = jnp.mean(loss_NS3**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 
        return total_loss, loss_u, loss_v, loss_w, loss_con, loss_NS1, loss_NS2, loss_NS3


class Boundary_layer(Equation):
    def __init__(self, all_params):
        self.all_params = all_params
    
    @staticmethod
    def Loss(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
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
        b_out = model_fns(all_params, boundaries[0])
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

        loss_u_b = all_params["data"]['u_ref']*b_out[:,0:1]
        loss_u_b = jnp.mean(loss_u_b**2)

        loss_v_b = all_params["data"]['v_ref']*b_out[:,1:2]
        loss_v_b = jnp.mean(loss_v_b**2)

        loss_w_b = all_params["data"]['w_ref']*b_out[:,2:3]
        loss_w_b = jnp.mean(loss_w_b**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz)
        loss_NS3 = jnp.mean(loss_NS3**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*(loss_u_b + loss_v_b + loss_w_b)
        return total_loss
    @staticmethod
    def Loss_report(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
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
        b_out = model_fns(all_params, boundaries[0])

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

        loss_u_b = all_params["data"]['u_ref']*b_out[:,0:1]
        loss_u_b = jnp.mean(loss_u_b**2)

        loss_v_b = all_params["data"]['v_ref']*b_out[:,1:2]
        loss_v_b = jnp.mean(loss_v_b**2)

        loss_w_b = all_params["data"]['w_ref']*b_out[:,2:3]
        loss_w_b = jnp.mean(loss_w_b**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz)
        loss_NS3 = jnp.mean(loss_NS3**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*(loss_u_b + loss_v_b + loss_w_b)
        return total_loss, loss_u, loss_v, loss_w, loss_con, loss_NS1, loss_NS2, loss_NS3

class Boundary_layer_energy(Equation):
    def __init__(self, all_params):
        self.all_params = all_params
    
    @staticmethod
    def Loss(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
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
        b_out = model_fns(all_params, boundaries[0])
        b_out2 = model_fns(all_params, boundaries[1])
        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]
        T = all_params["data"]['T_ref']*out[:,4:5]

        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]
        Tt = all_params["data"]['T_ref']*out_t[:,4:5]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]
        Tx = all_params["data"]['T_ref']*out_x[:,4:5]/all_params["domain"]["domain_range"]["x"][1]

        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]
        Ty = all_params["data"]['T_ref']*out_y[:,4:5]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]
        Tz = all_params["data"]['T_ref']*out_z[:,4:5]/all_params["domain"]["domain_range"]["z"][1]

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2
        Txx = all_params["data"]['T_ref']*out_xx[:,4:5]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2
        Tyy = all_params["data"]['T_ref']*out_yy[:,4:5]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2
        Tzz = all_params["data"]['T_ref']*out_zz[:,4:5]/all_params["domain"]["domain_range"]["z"][1]**2

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_u_b = all_params["data"]['u_ref']*b_out[:,0:1]
        loss_u_b = jnp.mean(loss_u_b**2)

        loss_v_b = all_params["data"]['v_ref']*b_out[:,1:2]
        loss_v_b = jnp.mean(loss_v_b**2)

        loss_w_b = all_params["data"]['w_ref']*b_out[:,2:3]
        loss_w_b = jnp.mean(loss_w_b**2)

        loss_T_b = all_params["data"]['T_ref']*b_out2[:,4:5] - boundaries[2]
        loss_T_b = jnp.mean(loss_T_b**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz)
        loss_NS3 = jnp.mean(loss_NS3**2)
        loss_ENR = Tt + u*Tx + v*Ty + w*Tz - all_params["data"]["viscosity"]*(Txx+Tyy+Tzz)/0.25
        loss_ENR = jnp.mean(loss_ENR**2)
        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*(loss_u_b + loss_v_b + loss_w_b + loss_T_b) + weights[8]*loss_ENR
        return total_loss
    @staticmethod
    def Loss_report(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
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
        b_out = model_fns(all_params, boundaries[0])
        b_out2 = model_fns(all_params, boundaries[1])
        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]
        T = all_params["data"]['T_ref']*out[:,4:5]

        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]
        Tt = all_params["data"]['T_ref']*out_t[:,4:5]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]
        Tx = all_params["data"]['T_ref']*out_x[:,4:5]/all_params["domain"]["domain_range"]["x"][1]

        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]
        Ty = all_params["data"]['T_ref']*out_y[:,4:5]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]
        Tz = all_params["data"]['T_ref']*out_z[:,4:5]/all_params["domain"]["domain_range"]["z"][1]

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2
        Txx = all_params["data"]['T_ref']*out_xx[:,4:5]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2
        Tyy = all_params["data"]['T_ref']*out_yy[:,4:5]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2
        Tzz = all_params["data"]['T_ref']*out_zz[:,4:5]/all_params["domain"]["domain_range"]["z"][1]**2

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_u_b = all_params["data"]['u_ref']*b_out[:,0:1]
        loss_u_b = jnp.mean(loss_u_b**2)

        loss_v_b = all_params["data"]['v_ref']*b_out[:,1:2]
        loss_v_b = jnp.mean(loss_v_b**2)

        loss_w_b = all_params["data"]['w_ref']*b_out[:,2:3]
        loss_w_b = jnp.mean(loss_w_b**2)

        loss_T_b = all_params["data"]['T_ref']*b_out2[:,4:5] - boundaries[2]
        loss_T_b = jnp.mean(loss_T_b**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz)
        loss_NS3 = jnp.mean(loss_NS3**2)
        loss_ENR = Tt + u*Tx + v*Ty + w*Tz - all_params["data"]["viscosity"]*(Txx+Tyy+Tzz)/0.25
        loss_ENR = jnp.mean(loss_ENR**2)
        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*(loss_u_b + loss_v_b + loss_w_b + loss_T_b) + weights[8]*loss_ENR
        return total_loss, loss_u, loss_v, loss_w, loss_con, loss_NS1, loss_NS2, loss_NS3, loss_ENR

class DUCT_flow(Equation):
    def __init__(self, all_params):
        self.all_params = all_params
    
    @staticmethod
    def Loss(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
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
        b_out1 = model_fns(all_params, boundaries[0])
        b_out2 = model_fns(all_params, boundaries[1])
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

        loss_u_b1 = all_params["data"]['u_ref']*b_out1[:,0:1]
        loss_u_b1 = jnp.mean(loss_u_b1**2)

        loss_v_b1 = all_params["data"]['v_ref']*b_out1[:,1:2]
        loss_v_b1 = jnp.mean(loss_v_b1**2)

        loss_w_b1 = all_params["data"]['w_ref']*b_out1[:,2:3]
        loss_w_b1 = jnp.mean(loss_w_b1**2)

        loss_u_b2 = all_params["data"]['u_ref']*b_out2[:,0:1]
        loss_u_b2 = jnp.mean(loss_u_b2**2)

        loss_v_b2 = all_params["data"]['v_ref']*b_out2[:,1:2]
        loss_v_b2 = jnp.mean(loss_v_b2**2)

        loss_w_b2 = all_params["data"]['w_ref']*b_out2[:,2:3]
        loss_w_b2 = jnp.mean(loss_w_b2**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz)
        loss_NS3 = jnp.mean(loss_NS3**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*(loss_u_b1 + loss_v_b1 + loss_w_b1 + loss_u_b2 + loss_v_b2 + loss_w_b2)
        return total_loss
    @staticmethod
    def Loss_report(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
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
        b_out1 = model_fns(all_params, boundaries[0])
        b_out2 = model_fns(all_params, boundaries[1])
        
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

        loss_u_b1 = all_params["data"]['u_ref']*b_out1[:,0:1]
        loss_u_b1 = jnp.mean(loss_u_b1**2)

        loss_v_b1 = all_params["data"]['v_ref']*b_out1[:,1:2]
        loss_v_b1 = jnp.mean(loss_v_b1**2)

        loss_w_b1 = all_params["data"]['w_ref']*b_out1[:,2:3]
        loss_w_b1 = jnp.mean(loss_w_b1**2)

        loss_u_b2 = all_params["data"]['u_ref']*b_out2[:,0:1]
        loss_u_b2 = jnp.mean(loss_u_b2**2)

        loss_v_b2 = all_params["data"]['v_ref']*b_out2[:,1:2]
        loss_v_b2 = jnp.mean(loss_v_b2**2)

        loss_w_b2 = all_params["data"]['w_ref']*b_out2[:,2:3]
        loss_w_b2 = jnp.mean(loss_w_b2**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz)
        loss_NS3 = jnp.mean(loss_NS3**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*(loss_u_b1 + loss_v_b1 + loss_w_b1 + loss_u_b2 + loss_v_b2 + loss_w_b2)
        return total_loss, loss_u, loss_v, loss_w, loss_con, loss_NS1, loss_NS2, loss_NS3

class Energy(Equation):
    def __init__(self, all_params):
        self.all_params = all_params
    
    @staticmethod
    def Loss(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
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
        b_out1 = model_fns(all_params, boundaries[0])                                                                                  
        b_out2 = model_fns(all_params, boundaries[1])                                                                                  

        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]
        T = all_params["data"]['T_ref']*out[:,4:5]

        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]
        Tt = all_params["data"]['T_ref']*out_t[:,4:5]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]
        Tx = all_params["data"]['T_ref']*out_x[:,4:5]/all_params["domain"]["domain_range"]["x"][1]

        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]
        Ty = all_params["data"]['T_ref']*out_y[:,4:5]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]
        Tz = all_params["data"]['T_ref']*out_z[:,4:5]/all_params["domain"]["domain_range"]["z"][1]

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2
        Txx = all_params["data"]['T_ref']*out_xx[:,4:5]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2
        Tyy = all_params["data"]['T_ref']*out_yy[:,4:5]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2
        Tzz = all_params["data"]['T_ref']*out_zz[:,4:5]/all_params["domain"]["domain_range"]["z"][1]**2

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_T_bu = all_params["data"]['T_ref']*b_out1[:,4:5] + all_params["data"]['T_ref']
        loss_T_bu = jnp.mean(loss_T_bu**2)
        
        loss_T_bb = all_params["data"]['T_ref']*b_out2[:,4:5] - all_params["data"]['T_ref']
        loss_T_bb = jnp.mean(loss_T_bb**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz) - T

        loss_NS3 = jnp.mean(loss_NS3**2)

        loss_ENR = Tt + u*Tx + v*Ty + w*Tz - all_params["data"]["viscosity"]*(Txx+Tyy+Tzz)/7
        loss_ENR = jnp.mean(loss_ENR**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*loss_ENR + weights[8]*(loss_T_bu + loss_T_bb)
        return total_loss
    @staticmethod
    def Loss_report(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
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
        b_out1 = model_fns(all_params, boundaries[0])                                                                                  
        b_out2 = model_fns(all_params, boundaries[1])                                                                                  

        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]
        T = all_params["data"]['T_ref']*out[:,4:5]
        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]
        Tt = all_params["data"]['T_ref']*out_t[:,4:5]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]
        Tx = all_params["data"]['T_ref']*out_x[:,4:5]/all_params["domain"]["domain_range"]["x"][1]

        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]
        Ty = all_params["data"]['T_ref']*out_y[:,4:5]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]
        Tz = all_params["data"]['T_ref']*out_z[:,4:5]/all_params["domain"]["domain_range"]["z"][1]

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2
        Txx = all_params["data"]['T_ref']*out_xx[:,4:5]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2
        Tyy = all_params["data"]['T_ref']*out_yy[:,4:5]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2
        Tzz = all_params["data"]['T_ref']*out_zz[:,4:5]/all_params["domain"]["domain_range"]["z"][1]**2

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_T_bu = all_params["data"]['T_ref']*b_out1[:,4:5] + all_params["data"]['T_ref']
        loss_T_bu = jnp.mean(loss_T_bu**2)
        
        loss_T_bb = all_params["data"]['T_ref']*b_out2[:,4:5] - all_params["data"]['T_ref']
        loss_T_bb = jnp.mean(loss_T_bb**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz) - T

        loss_NS3 = jnp.mean(loss_NS3**2)

        loss_ENR = Tt + u*Tx + v*Ty + w*Tz - all_params["data"]["viscosity"]*(Txx+Tyy+Tzz)/7
        loss_ENR = jnp.mean(loss_ENR**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*loss_ENR + weights[8]*(loss_T_bu + loss_T_bb)
        return total_loss, loss_u, loss_v, loss_w, loss_con, loss_NS1, loss_NS2, loss_NS3, loss_ENR, loss_T_bu, loss_T_bb
class Energy_hard_profile(Equation):
    def __init__(self, all_params):
        self.all_params = all_params
    
    @staticmethod
    def Loss(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
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
        
        def temperature_profile(z):
            denom = 2-2*jnp.exp(-0.5*30)
            return jnp.where(z > 0.5,
                            (jnp.exp(-z * 30) - jnp.exp(-0.5 * 30)) / denom,
                            (jnp.exp(-0.5 * 30) - jnp.exp(30 * (z - 1))) / denom
                            )
        def temperature_profile_z(z):
            denom = 2-2*jnp.exp(-0.5*30)
            return jnp.where(z > 0.5,
                            -30*jnp.exp(-z * 30) / denom,
                            -30*jnp.exp(30 * (z - 1)) / denom
                            )
        def temperature_profile_zz(z):
            denom = 2-2*jnp.exp(-0.5*30)
            return jnp.where(z > 0.5,
                            900*jnp.exp(-z * 30) / denom,
                            -900*jnp.exp(30 * (z - 1)) / denom
                            )
   
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
        out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_x, out_xx = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_y, out_yy = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_z, out_zz = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)

        p_out = model_fns(all_params, particles)
        b_out1 = model_fns(all_params, boundaries[0])                                                                                  
        b_out2 = model_fns(all_params, boundaries[1])                                                                                  
        temp_mean = temperature_profile(g_batch[:,3:4])
        temp_mean_z = temperature_profile_z(g_batch[:,3:4])
        temp_mean_zz = temperature_profile_zz(g_batch[:,3:4])
        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]
        T_ = all_params["data"]['T_ref']*out[:,4:5]
        T = (g_batch[:,3:4]**2-g_batch[:,3:4])*T_ + temp_mean
        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]
        Tt = (g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_t[:,4:5]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]
        Tx = (g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_x[:,4:5]/all_params["domain"]["domain_range"]["x"][1]

        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]
        Ty = (g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_y[:,4:5]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]
        Tz = (2*g_batch[:,3:4]-1)*T_+(g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_z[:,4:5]/all_params["domain"]["domain_range"]["z"][1] + temp_mean_z

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2
        Txx = (g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_xx[:,4:5]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2
        Tyy = (g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_yy[:,4:5]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2
        Tzz = 2*T_+(4*g_batch[:,3:4]-2)*all_params["data"]['T_ref']*out_z[:,4:5]/all_params["domain"]["domain_range"]["z"][1]\
              +(g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_zz[:,4:5]/all_params["domain"]["domain_range"]["z"][1]**2 + temp_mean_zz

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_T_bu = all_params["data"]['T_ref']*b_out1[:,4:5] + all_params["data"]['T_ref']
        loss_T_bu = jnp.mean(loss_T_bu**2)
        
        loss_T_bb = all_params["data"]['T_ref']*b_out2[:,4:5] - all_params["data"]['T_ref']
        loss_T_bb = jnp.mean(loss_T_bb**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz) - T

        loss_NS3 = jnp.mean(loss_NS3**2)

        loss_ENR = Tt + u*Tx + v*Ty + w*Tz - all_params["data"]["viscosity"]*(Txx+Tyy+Tzz)/7
        loss_ENR = jnp.mean(loss_ENR**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*loss_ENR + weights[8]*(loss_T_bu + loss_T_bb)
        return total_loss
    @staticmethod
    def Loss_report(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
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
        def temperature_profile(z):
            denom = 2-2*jnp.exp(-0.5*30)
            return jnp.where(z > 0.5,
                            (jnp.exp(-z * 30) - jnp.exp(-0.5 * 30)) / denom,
                            (jnp.exp(-0.5 * 30) - jnp.exp(30 * (z - 1))) / denom
                            )
        def temperature_profile_z(z):
            denom = 2-2*jnp.exp(-0.5*30)
            return jnp.where(z > 0.5,
                            -30*jnp.exp(-z * 30) / denom,
                            -30*jnp.exp(30 * (z - 1)) / denom
                            )
        def temperature_profile_zz(z):
            denom = 2-2*jnp.exp(-0.5*30)
            return jnp.where(z > 0.5,
                            900*jnp.exp(-z * 30) / denom,
                            -900*jnp.exp(30 * (z - 1)) / denom
                            )
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
        out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_x, out_xx = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_y, out_yy = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_z, out_zz = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)

        p_out = model_fns(all_params, particles)
        b_out1 = model_fns(all_params, boundaries[0])                                                                                  
        b_out2 = model_fns(all_params, boundaries[1])                                                                                  
        temp_mean = temperature_profile(g_batch[:,3:4])
        temp_mean_z = temperature_profile_z(g_batch[:,3:4])
        temp_mean_zz = temperature_profile_zz(g_batch[:,3:4])
        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]
        T_ = all_params["data"]['T_ref']*out[:,4:5]
        T = (g_batch[:,3:4]**2-g_batch[:,3:4])*T_ + temp_mean
        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]
        Tt = (g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_t[:,4:5]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]
        Tx = (g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_x[:,4:5]/all_params["domain"]["domain_range"]["x"][1]

        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]
        Ty = (g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_y[:,4:5]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]
        Tz = (2*g_batch[:,3:4]-1)*T_+(g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_z[:,4:5]/all_params["domain"]["domain_range"]["z"][1] + temp_mean_z

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2
        Txx = (g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_xx[:,4:5]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2
        Tyy = (g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_yy[:,4:5]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2
        Tzz = 2*T_+(4*g_batch[:,3:4]-2)*all_params["data"]['T_ref']*out_z[:,4:5]/all_params["domain"]["domain_range"]["z"][1]\
              +(g_batch[:,3:4]**2-g_batch[:,3:4])*all_params["data"]['T_ref']*out_zz[:,4:5]/all_params["domain"]["domain_range"]["z"][1]**2 + temp_mean_zz


        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_T_bu = all_params["data"]['T_ref']*b_out1[:,4:5] + all_params["data"]['T_ref']
        loss_T_bu = jnp.mean(loss_T_bu**2)
        
        loss_T_bb = all_params["data"]['T_ref']*b_out2[:,4:5] - all_params["data"]['T_ref']
        loss_T_bb = jnp.mean(loss_T_bb**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz) - T

        loss_NS3 = jnp.mean(loss_NS3**2)

        loss_ENR = Tt + u*Tx + v*Ty + w*Tz - all_params["data"]["viscosity"]*(Txx+Tyy+Tzz)/7
        loss_ENR = jnp.mean(loss_ENR**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*loss_ENR + weights[8]*(loss_T_bu + loss_T_bb)
        return total_loss, loss_u, loss_v, loss_w, loss_con, loss_NS1, loss_NS2, loss_NS3, loss_ENR, loss_T_bu, loss_T_bb


class Energy_bd(Equation):
    def __init__(self, all_params):
        self.all_params = all_params
    
    @staticmethod
    def Loss(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
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

        b_out1, b_out1_x = first_order(all_params, boundaries[0], jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(boundaries[0].shape[0],1)),model_fns)
        b_out2, b_out2_x = first_order(all_params, boundaries[1], jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(boundaries[1].shape[0],1)),model_fns)
        b_out3, b_out3_y = first_order(all_params, boundaries[2], jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(boundaries[2].shape[0],1)),model_fns)
        b_out4, b_out4_y = first_order(all_params, boundaries[3], jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(boundaries[3].shape[0],1)),model_fns)
        b_out5 = model_fns(all_params, boundaries[4])  
        b_out6 = model_fns(all_params, boundaries[5])

        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]
        T = all_params["data"]['T_ref']*out[:,4:5]

        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]
        Tt = all_params["data"]['T_ref']*out_t[:,4:5]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]
        Tx = all_params["data"]['T_ref']*out_x[:,4:5]/all_params["domain"]["domain_range"]["x"][1]

        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]
        Ty = all_params["data"]['T_ref']*out_y[:,4:5]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]
        Tz = all_params["data"]['T_ref']*out_z[:,4:5]/all_params["domain"]["domain_range"]["z"][1]

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2
        Txx = all_params["data"]['T_ref']*out_xx[:,4:5]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2
        Tyy = all_params["data"]['T_ref']*out_yy[:,4:5]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2
        Tzz = all_params["data"]['T_ref']*out_zz[:,4:5]/all_params["domain"]["domain_range"]["z"][1]**2

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_u_zu = all_params["data"]['u_ref']*b_out5[:,0:1]
        loss_u_zu = jnp.mean(loss_u_zu**2)
        loss_u_zl = all_params["data"]['u_ref']*b_out6[:,0:1]
        loss_u_zl = jnp.mean(loss_u_zl**2)
        loss_v_zu = all_params["data"]['v_ref']*b_out5[:,1:2]
        loss_v_zu = jnp.mean(loss_v_zu**2)
        loss_v_zl = all_params["data"]['v_ref']*b_out6[:,1:2]
        loss_v_zl = jnp.mean(loss_v_zl**2)
        loss_w_zu = all_params["data"]['w_ref']*b_out5[:,2:3]
        loss_w_zu = jnp.mean(loss_w_zu**2)
        loss_w_zl = all_params["data"]['w_ref']*b_out6[:,2:3]
        loss_w_zl = jnp.mean(loss_w_zl**2)
        loss_T_zu = all_params["data"]['T_ref']*b_out5[:,4:5] + all_params["data"]['T_ref']
        loss_T_zu = jnp.mean(loss_T_zu**2)
        loss_T_zb = all_params["data"]['T_ref']*b_out6[:,4:5] - all_params["data"]['T_ref']
        loss_T_zb = jnp.mean(loss_T_zb**2)

        loss_u_xu = all_params["data"]['u_ref']*b_out1[:,0:1]
        loss_u_xu = jnp.mean(loss_u_xu**2)
        loss_u_xl = all_params["data"]['u_ref']*b_out2[:,0:1]
        loss_u_xl = jnp.mean(loss_u_xl**2)
        loss_v_xu = all_params["data"]['v_ref']*b_out1[:,1:2]
        loss_v_xu = jnp.mean(loss_v_xu**2)
        loss_v_xl = all_params["data"]['v_ref']*b_out2[:,1:2]
        loss_v_xl = jnp.mean(loss_v_xl**2)
        loss_w_xu = all_params["data"]['w_ref']*b_out1[:,2:3]
        loss_w_xu = jnp.mean(loss_w_xu**2)
        loss_w_xl = all_params["data"]['w_ref']*b_out2[:,2:3]
        loss_w_xl = jnp.mean(loss_w_xl**2)
        loss_Tx_xu = all_params["data"]['T_ref']*b_out1_x[:,4:5]/all_params["domain"]["domain_range"]['x'][1]
        loss_Tx_xu = jnp.mean(loss_Tx_xu**2)
        loss_Tx_xl = all_params["data"]['T_ref']*b_out2_x[:,4:5]/all_params["domain"]["domain_range"]['x'][1]
        loss_Tx_xl = jnp.mean(loss_Tx_xl**2)

        loss_u_yu = all_params["data"]['u_ref']*b_out3[:,0:1]
        loss_u_yu = jnp.mean(loss_u_yu**2)
        loss_u_yl = all_params["data"]['u_ref']*b_out4[:,0:1]
        loss_u_yl = jnp.mean(loss_u_yl**2)
        loss_v_yu = all_params["data"]['v_ref']*b_out3[:,1:2]
        loss_v_yu = jnp.mean(loss_v_yu**2)
        loss_v_yl = all_params["data"]['v_ref']*b_out4[:,1:2]
        loss_v_yl = jnp.mean(loss_v_yl**2)
        loss_w_yu = all_params["data"]['w_ref']*b_out3[:,2:3]
        loss_w_yu = jnp.mean(loss_w_yu**2)
        loss_w_yl = all_params["data"]['w_ref']*b_out4[:,2:3]
        loss_w_yl = jnp.mean(loss_w_yl**2)
        loss_Ty_yu = all_params["data"]['T_ref']*b_out3_y[:,4:5]/all_params["domain"]["domain_range"]['y'][1]
        loss_Ty_yu = jnp.mean(loss_Ty_yu**2)
        loss_Ty_yl = all_params["data"]['T_ref']*b_out4_y[:,4:5]/all_params["domain"]["domain_range"]['y'][1]
        loss_Ty_yl = jnp.mean(loss_Ty_yl**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz) - T

        loss_NS3 = jnp.mean(loss_NS3**2)

        loss_ENR = Tt + u*Tx + v*Ty + w*Tz - all_params["data"]["viscosity"]*(Txx+Tyy+Tzz)/7
        loss_ENR = jnp.mean(loss_ENR**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*loss_ENR + weights[8]*(loss_T_zu + loss_T_zb) + \
                    weights[9]*(loss_u_xu + loss_u_xl + loss_v_xu + loss_v_xl + loss_w_xu + loss_w_xl + loss_u_yu + 
                                loss_u_yl + loss_v_yu + loss_v_yl + loss_w_yu + loss_w_yl) + \
                    weights[10]*(loss_Tx_xu + loss_Tx_xl + loss_Ty_yu + loss_Ty_yl)
        return total_loss
    @staticmethod
    def Loss_report(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
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
        b_out1, b_out1_x = first_order(all_params, boundaries[0], jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(boundaries[0].shape[0],1)),model_fns)
        b_out2, b_out2_x = first_order(all_params, boundaries[1], jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(boundaries[1].shape[0],1)),model_fns)
        b_out3, b_out3_y = first_order(all_params, boundaries[2], jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(boundaries[2].shape[0],1)),model_fns)
        b_out4, b_out4_y = first_order(all_params, boundaries[3], jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(boundaries[3].shape[0],1)),model_fns)
        b_out5 = model_fns(all_params, boundaries[4])  
        b_out6 = model_fns(all_params, boundaries[5])

        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]
        T = all_params["data"]['T_ref']*out[:,4:5]

        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]
        Tt = all_params["data"]['T_ref']*out_t[:,4:5]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]
        Tx = all_params["data"]['T_ref']*out_x[:,4:5]/all_params["domain"]["domain_range"]["x"][1]

        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]
        Ty = all_params["data"]['T_ref']*out_y[:,4:5]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]
        Tz = all_params["data"]['T_ref']*out_z[:,4:5]/all_params["domain"]["domain_range"]["z"][1]

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2
        Txx = all_params["data"]['T_ref']*out_xx[:,4:5]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2
        Tyy = all_params["data"]['T_ref']*out_yy[:,4:5]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2
        Tzz = all_params["data"]['T_ref']*out_zz[:,4:5]/all_params["domain"]["domain_range"]["z"][1]**2

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_u_zu = all_params["data"]['u_ref']*b_out5[:,0:1]
        loss_u_zu = jnp.mean(loss_u_zu**2)
        loss_u_zl = all_params["data"]['u_ref']*b_out6[:,0:1]
        loss_u_zl = jnp.mean(loss_u_zl**2)
        loss_v_zu = all_params["data"]['v_ref']*b_out5[:,1:2]
        loss_v_zu = jnp.mean(loss_v_zu**2)
        loss_v_zl = all_params["data"]['v_ref']*b_out6[:,1:2]
        loss_v_zl = jnp.mean(loss_v_zl**2)
        loss_w_zu = all_params["data"]['w_ref']*b_out5[:,2:3]
        loss_w_zu = jnp.mean(loss_w_zu**2)
        loss_w_zl = all_params["data"]['w_ref']*b_out6[:,2:3]
        loss_w_zl = jnp.mean(loss_w_zl**2)
        loss_T_zu = all_params["data"]['T_ref']*b_out5[:,4:5] + all_params["data"]['T_ref']
        loss_T_zu = jnp.mean(loss_T_zu**2)
        loss_T_zb = all_params["data"]['T_ref']*b_out6[:,4:5] - all_params["data"]['T_ref']
        loss_T_zb = jnp.mean(loss_T_zb**2)

        loss_u_xu = all_params["data"]['u_ref']*b_out1[:,0:1]
        loss_u_xu = jnp.mean(loss_u_xu**2)
        loss_u_xl = all_params["data"]['u_ref']*b_out2[:,0:1]
        loss_u_xl = jnp.mean(loss_u_xl**2)
        loss_v_xu = all_params["data"]['v_ref']*b_out1[:,1:2]
        loss_v_xu = jnp.mean(loss_v_xu**2)
        loss_v_xl = all_params["data"]['v_ref']*b_out2[:,1:2]
        loss_v_xl = jnp.mean(loss_v_xl**2)
        loss_w_xu = all_params["data"]['w_ref']*b_out1[:,2:3]
        loss_w_xu = jnp.mean(loss_w_xu**2)
        loss_w_xl = all_params["data"]['w_ref']*b_out2[:,2:3]
        loss_w_xl = jnp.mean(loss_w_xl**2)
        loss_Tx_xu = all_params["data"]['T_ref']*b_out1_x[:,4:5]/all_params["domain"]["domain_range"]['x'][1]
        loss_Tx_xu = jnp.mean(loss_Tx_xu**2)
        loss_Tx_xl = all_params["data"]['T_ref']*b_out2_x[:,4:5]/all_params["domain"]["domain_range"]['x'][1]
        loss_Tx_xl = jnp.mean(loss_Tx_xl**2)

        loss_u_yu = all_params["data"]['u_ref']*b_out3[:,0:1]
        loss_u_yu = jnp.mean(loss_u_yu**2)
        loss_u_yl = all_params["data"]['u_ref']*b_out4[:,0:1]
        loss_u_yl = jnp.mean(loss_u_yl**2)
        loss_v_yu = all_params["data"]['v_ref']*b_out3[:,1:2]
        loss_v_yu = jnp.mean(loss_v_yu**2)
        loss_v_yl = all_params["data"]['v_ref']*b_out4[:,1:2]
        loss_v_yl = jnp.mean(loss_v_yl**2)
        loss_w_yu = all_params["data"]['w_ref']*b_out3[:,2:3]
        loss_w_yu = jnp.mean(loss_w_yu**2)
        loss_w_yl = all_params["data"]['w_ref']*b_out4[:,2:3]
        loss_w_yl = jnp.mean(loss_w_yl**2)
        loss_Ty_yu = all_params["data"]['T_ref']*b_out3_y[:,4:5]/all_params["domain"]["domain_range"]['y'][1]
        loss_Ty_yu = jnp.mean(loss_Ty_yu**2)
        loss_Ty_yl = all_params["data"]['T_ref']*b_out4_y[:,4:5]/all_params["domain"]["domain_range"]['y'][1]
        loss_Ty_yl = jnp.mean(loss_Ty_yl**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz) - T

        loss_NS3 = jnp.mean(loss_NS3**2)

        loss_ENR = Tt + u*Tx + v*Ty + w*Tz - all_params["data"]["viscosity"]*(Txx+Tyy+Tzz)/7
        loss_ENR = jnp.mean(loss_ENR**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*loss_ENR + weights[8]*(loss_T_zu + loss_T_zb) + \
                    weights[9]*(loss_u_xu + loss_u_xl + loss_v_xu + loss_v_xl + loss_w_xu + loss_w_xl + loss_u_yu + 
                                loss_u_yl + loss_v_yu + loss_v_yl + loss_w_yu + loss_w_yl) + \
                    weights[10]*(loss_Tx_xu + loss_Tx_xl + loss_Ty_yu + loss_Ty_yl)
        return total_loss, loss_u, loss_v, loss_w, loss_con, loss_NS1, loss_NS2, loss_NS3, loss_ENR, loss_T_zu, loss_T_zb

class Energy_vor(Equation):
    def __init__(self, all_params):
        self.all_params = all_params
    
    @staticmethod
    def Loss(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
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
        keys = ['u_ref', 'v_ref', 'w_ref', 'p_ref', 'T_ref']
        keys2 = ['t', 'x', 'y', 'z']
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
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
        out, out_t = equ_func1(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)

        uvwp = jnp.concatenate([out[:,k:(k+1)]*all_params["data"][keys[k]] for k in range(len(keys))],1)

        uts = jnp.concatenate([out_t[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
        uxs = jnp.concatenate([out_x[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1] for k in range(len(keys))],1)
        uys = jnp.concatenate([out_y[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uzs = jnp.concatenate([out_z[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uxts = jnp.concatenate([out_xt[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
        uyts = jnp.concatenate([out_yt[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
        uzts = jnp.concatenate([out_zt[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
        uxxs = jnp.concatenate([out_xx[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1] for k in range(len(keys))],1)
        uxys = jnp.concatenate([out_xy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uxzs = jnp.concatenate([out_xz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uyys = jnp.concatenate([out_yy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uyzs = jnp.concatenate([out_yz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uzzs = jnp.concatenate([out_zz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uxxxs = jnp.concatenate([out_xxx[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1] for k in range(len(keys))],1)
        uxxys = jnp.concatenate([out_xxy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uxxzs = jnp.concatenate([out_xxz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uxyys = jnp.concatenate([out_xyy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uyyys = jnp.concatenate([out_yyy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uyyzs = jnp.concatenate([out_yyz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uxzzs = jnp.concatenate([out_xzz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uyzzs = jnp.concatenate([out_yzz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uzzzs = jnp.concatenate([out_zzz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)


        p_out = model_fns(all_params, particles)

        b_out1 = model_fns(all_params, boundaries[0])  
        b_out2 = model_fns(all_params, boundaries[1])

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_T_zu = all_params["data"]['T_ref']*b_out1[:,4:5] + all_params["data"]['T_ref']
        loss_T_zu = jnp.mean(loss_T_zu**2)
        loss_T_zb = all_params["data"]['T_ref']*b_out2[:,4:5] - all_params["data"]['T_ref']
        loss_T_zb = jnp.mean(loss_T_zb**2)

        loss_con = uxs[:,0:1] + uys[:,1:2] + uzs[:,2:3]
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = uts[:,0:1] + uvwp[:,0:1]*uxs[:,0:1] + uvwp[:,1:2]*uys[:,0:1] + uvwp[:,2:3]*uzs[:,0:1] + uxs[:,3:4] - all_params["data"]["viscosity"]*(uxxs[:,0:1]+uyys[:,0:1]+uzzs[:,0:1])
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = uts[:,1:2] + uvwp[:,0:1]*uxs[:,1:2] + uvwp[:,1:2]*uys[:,1:2] + uvwp[:,2:3]*uzs[:,1:2] + uys[:,3:4] - all_params["data"]["viscosity"]*(uxxs[:,1:2]+uyys[:,1:2]+uzzs[:,1:2])
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = uts[:,2:3] + uvwp[:,0:1]*uxs[:,2:3] + uvwp[:,1:2]*uys[:,2:3] + uvwp[:,2:3]*uzs[:,2:3] + uzs[:,3:4] - all_params["data"]["viscosity"]*(uxxs[:,2:3]+uyys[:,2:3]+uzzs[:,2:3]) - uvwp[:,4:5]
        loss_NS3 = jnp.mean(loss_NS3**2)

        loss_vor1 = uyts[:,2:3]-uzts[:,1:2] + uvwp[:,0:1]*(uxys[:,2:3] - uxzs[:,1:2]) + uvwp[:,1:2]*(uyys[:,2:3] - uyzs[:,1:2]) + uvwp[:,2:3]*(uyzs[:,2:3] - uzzs[:,1:2]) \
         - (uys[:,2:3] - uzs[:,1:2])*uxs[:,0:1] - (uzs[:,0:1] - uxs[:,2:3])*uys[:,0:1] - (uxs[:,1:2] - uys[:,0:1])*uzs[:,0:1] \
         - all_params["data"]["viscosity"]*(uxxys[:,2:3] + uyyys[:,2:3] + uyzzs[:,2:3] - uxxzs[:,1:2] - uyyzs[:,1:2] - uzzzs[:,1:2]) - uys[:,4:5]
        loss_vor1 = jnp.mean(loss_vor1**2)

        loss_vor2 = (uys[:,2:3] - uzs[:,1:2])*uxs[:,1:2] + (uzs[:,0:1] - uxs[:,2:3])*uys[:,1:2] + (uxs[:,1:2] - uys[:,0:1])*uzs[:,1:2] \
         + all_params["data"]["viscosity"]*(uxxzs[:,0:1] + uyyzs[:,0:1] + uzzzs[:,0:1] - uxxxs[:,2:3] - uxyys[:,2:3] - uxzzs[:,2:3]) \
         - uzts[:,0:1] + uxts[:,2:3] - uvwp[:,0:1]*(uxzs[:,0:1] - uxxs[:,2:3]) - uvwp[:,1:2]*(uyzs[:,0:1] - uxys[:,2:3]) - uvwp[:,2:3]*(uzzs[:,0:1] - uxzs[:,2:3]) - uxs[:,4:5]
        loss_vor2 = jnp.mean(loss_vor2**2)
        

        loss_ENR = uts[:,4:5] + uvwp[:,0:1]*uxs[:,4:5] + uvwp[:,1:2]*uys[:,4:5] + uvwp[:,2:3]*uzs[:,4:5] - all_params["data"]["viscosity"]*(uxxs[:,4:5]+uyys[:,4:5]+uzzs[:,4:5])/7
        loss_ENR = jnp.mean(loss_ENR**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*loss_ENR + weights[8]*(loss_T_zu + loss_T_zb) + \
                    weights[9]*(loss_vor1 + loss_vor2)
        return total_loss
    @staticmethod
    def Loss_report(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
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
        keys = ['u_ref', 'v_ref', 'w_ref', 'p_ref', 'T_ref']
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
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
        out, out_t = equ_func1(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)

        uvwp = jnp.concatenate([out[:,k:(k+1)]*all_params["data"][keys[k]] for k in range(len(keys))],1)

        uts = jnp.concatenate([out_t[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
        uxs = jnp.concatenate([out_x[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1] for k in range(len(keys))],1)
        uys = jnp.concatenate([out_y[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uzs = jnp.concatenate([out_z[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uxts = jnp.concatenate([out_xt[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
        uyts = jnp.concatenate([out_yt[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
        uzts = jnp.concatenate([out_zt[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
        uxxs = jnp.concatenate([out_xx[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1] for k in range(len(keys))],1)
        uxys = jnp.concatenate([out_xy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uxzs = jnp.concatenate([out_xz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uyys = jnp.concatenate([out_yy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uyzs = jnp.concatenate([out_yz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uzzs = jnp.concatenate([out_zz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uxxxs = jnp.concatenate([out_xxx[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1] for k in range(len(keys))],1)
        uxxys = jnp.concatenate([out_xxy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uxxzs = jnp.concatenate([out_xxz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uxyys = jnp.concatenate([out_xyy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uyyys = jnp.concatenate([out_yyy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uyyzs = jnp.concatenate([out_yyz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uxzzs = jnp.concatenate([out_xzz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uyzzs = jnp.concatenate([out_yzz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uzzzs = jnp.concatenate([out_zzz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)


        p_out = model_fns(all_params, particles)

        b_out1 = model_fns(all_params, boundaries[0])  
        b_out2 = model_fns(all_params, boundaries[1])

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_T_zu = all_params["data"]['T_ref']*b_out1[:,4:5] + all_params["data"]['T_ref']
        loss_T_zu = jnp.mean(loss_T_zu**2)
        loss_T_zb = all_params["data"]['T_ref']*b_out2[:,4:5] - all_params["data"]['T_ref']
        loss_T_zb = jnp.mean(loss_T_zb**2)

        loss_con = uxs[:,0:1] + uys[:,1:2] + uzs[:,2:3]
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = uts[:,0:1] + uvwp[:,0:1]*uxs[:,0:1] + uvwp[:,1:2]*uys[:,0:1] + uvwp[:,2:3]*uzs[:,0:1] + uxs[:,3:4] - all_params["data"]["viscosity"]*(uxxs[:,0:1]+uyys[:,0:1]+uzzs[:,0:1])
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = uts[:,1:2] + uvwp[:,0:1]*uxs[:,1:2] + uvwp[:,1:2]*uys[:,1:2] + uvwp[:,2:3]*uzs[:,1:2] + uys[:,3:4] - all_params["data"]["viscosity"]*(uxxs[:,1:2]+uyys[:,1:2]+uzzs[:,1:2])
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = uts[:,2:3] + uvwp[:,0:1]*uxs[:,2:3] + uvwp[:,1:2]*uys[:,2:3] + uvwp[:,2:3]*uzs[:,2:3] + uzs[:,3:4] - all_params["data"]["viscosity"]*(uxxs[:,2:3]+uyys[:,2:3]+uzzs[:,2:3]) - uvwp[:,4:5]
        loss_NS3 = jnp.mean(loss_NS3**2)

        loss_vor1 = uyts[:,2:3]-uzts[:,1:2] + uvwp[:,0:1]*(uxys[:,2:3] - uxzs[:,1:2]) + uvwp[:,1:2]*(uyys[:,2:3] - uyzs[:,1:2]) + uvwp[:,2:3]*(uyzs[:,2:3] - uzzs[:,1:2]) \
         - (uys[:,2:3] - uzs[:,1:2])*uxs[:,0:1] - (uzs[:,0:1] - uxs[:,2:3])*uys[:,0:1] - (uxs[:,1:2] - uys[:,0:1])*uzs[:,0:1] \
         - all_params["data"]["viscosity"]*(uxxys[:,2:3] + uyyys[:,2:3] + uyzzs[:,2:3] - uxxzs[:,1:2] - uyyzs[:,1:2] - uzzzs[:,1:2]) - uys[:,4:5]
        loss_vor1 = jnp.mean(loss_vor1**2)

        loss_vor2 = (uys[:,2:3] - uzs[:,1:2])*uxs[:,1:2] + (uzs[:,0:1] - uxs[:,2:3])*uys[:,1:2] + (uxs[:,1:2] - uys[:,0:1])*uzs[:,1:2] \
         + all_params["data"]["viscosity"]*(uxxzs[:,0:1] + uyyzs[:,0:1] + uzzzs[:,0:1] - uxxxs[:,2:3] - uxyys[:,2:3] - uxzzs[:,2:3]) \
         - uzts[:,0:1] + uxts[:,2:3] - uvwp[:,0:1]*(uxzs[:,0:1] - uxxs[:,2:3]) - uvwp[:,1:2]*(uyzs[:,0:1] - uxys[:,2:3]) - uvwp[:,2:3]*(uzzs[:,0:1] - uxzs[:,2:3]) - uxs[:,4:5]
        loss_vor2 = jnp.mean(loss_vor2**2)
        
        loss_ENR = uts[:,4:5] + uvwp[:,0:1]*uxs[:,4:5] + uvwp[:,1:2]*uys[:,4:5] + uvwp[:,2:3]*uzs[:,4:5] - all_params["data"]["viscosity"]*(uxxs[:,4:5]+uyys[:,4:5]+uzzs[:,4:5])/7
        loss_ENR = jnp.mean(loss_ENR**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*loss_ENR + weights[8]*(loss_T_zu + loss_T_zb) + \
                    weights[9]*(loss_vor1 + loss_vor2)
        return total_loss, loss_u, loss_v, loss_w, loss_con, loss_NS1, loss_NS2, loss_NS3, loss_ENR, loss_T_zu, loss_T_zb

class Energy_vor_bd(Equation):
    def __init__(self, all_params):
        self.all_params = all_params
    
    @staticmethod
    def Loss(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
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
        keys = ['u_ref', 'v_ref', 'w_ref', 'p_ref', 'T_ref']
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
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
        out, out_t = equ_func1(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)

        uvwp = jnp.concatenate([out[:,k:(k+1)]*all_params["data"][keys[k]] for k in range(len(keys))],1)

        uts = jnp.concatenate([out_t[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
        uxs = jnp.concatenate([out_x[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1] for k in range(len(keys))],1)
        uys = jnp.concatenate([out_y[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uzs = jnp.concatenate([out_z[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uxts = jnp.concatenate([out_xt[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
        uyts = jnp.concatenate([out_yt[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
        uzts = jnp.concatenate([out_zt[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
        uxxs = jnp.concatenate([out_xx[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1] for k in range(len(keys))],1)
        uxys = jnp.concatenate([out_xy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uxzs = jnp.concatenate([out_xz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uyys = jnp.concatenate([out_yy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uyzs = jnp.concatenate([out_yz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uzzs = jnp.concatenate([out_zz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uxxxs = jnp.concatenate([out_xxx[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1] for k in range(len(keys))],1)
        uxxys = jnp.concatenate([out_xxy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uxxzs = jnp.concatenate([out_xxz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uxyys = jnp.concatenate([out_xyy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uyyys = jnp.concatenate([out_yyy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uyyzs = jnp.concatenate([out_yyz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uxzzs = jnp.concatenate([out_xzz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uyzzs = jnp.concatenate([out_yzz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uzzzs = jnp.concatenate([out_zzz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)


        p_out = model_fns(all_params, particles)

        b_out1, b_out1_x = equ_func1(all_params, boundaries[0], jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(boundaries[0].shape[0],1)),model_fns)
        b_out2, b_out2_x = equ_func1(all_params, boundaries[1], jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(boundaries[1].shape[0],1)),model_fns)
        b_out3, b_out3_y = equ_func1(all_params, boundaries[2], jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(boundaries[2].shape[0],1)),model_fns)
        b_out4, b_out4_y = equ_func1(all_params, boundaries[3], jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(boundaries[3].shape[0],1)),model_fns)
        b_out5 = model_fns(all_params, boundaries[4])  
        b_out6 = model_fns(all_params, boundaries[5])



        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_u_zu = all_params["data"]['u_ref']*b_out5[:,0:1]
        loss_u_zu = jnp.mean(loss_u_zu**2)
        loss_u_zl = all_params["data"]['u_ref']*b_out6[:,0:1]
        loss_u_zl = jnp.mean(loss_u_zl**2)
        loss_v_zu = all_params["data"]['v_ref']*b_out5[:,1:2]
        loss_v_zu = jnp.mean(loss_v_zu**2)
        loss_v_zl = all_params["data"]['v_ref']*b_out6[:,1:2]
        loss_v_zl = jnp.mean(loss_v_zl**2)
        loss_w_zu = all_params["data"]['w_ref']*b_out5[:,2:3]
        loss_w_zu = jnp.mean(loss_w_zu**2)
        loss_w_zl = all_params["data"]['w_ref']*b_out6[:,2:3]
        loss_w_zl = jnp.mean(loss_w_zl**2)
        loss_T_zu = all_params["data"]['T_ref']*b_out5[:,4:5] + all_params["data"]['T_ref']
        loss_T_zu = jnp.mean(loss_T_zu**2)
        loss_T_zb = all_params["data"]['T_ref']*b_out6[:,4:5] - all_params["data"]['T_ref']
        loss_T_zb = jnp.mean(loss_T_zb**2)

        loss_u_xu = all_params["data"]['u_ref']*b_out1[:,0:1]
        loss_u_xu = jnp.mean(loss_u_xu**2)
        loss_u_xl = all_params["data"]['u_ref']*b_out2[:,0:1]
        loss_u_xl = jnp.mean(loss_u_xl**2)
        loss_v_xu = all_params["data"]['v_ref']*b_out1[:,1:2]
        loss_v_xu = jnp.mean(loss_v_xu**2)
        loss_v_xl = all_params["data"]['v_ref']*b_out2[:,1:2]
        loss_v_xl = jnp.mean(loss_v_xl**2)
        loss_w_xu = all_params["data"]['w_ref']*b_out1[:,2:3]
        loss_w_xu = jnp.mean(loss_w_xu**2)
        loss_w_xl = all_params["data"]['w_ref']*b_out2[:,2:3]
        loss_w_xl = jnp.mean(loss_w_xl**2)
        loss_Tx_xu = all_params["data"]['T_ref']*b_out1_x[:,4:5]/all_params["domain"]["domain_range"]['x'][1]
        loss_Tx_xu = jnp.mean(loss_Tx_xu**2)
        loss_Tx_xl = all_params["data"]['T_ref']*b_out2_x[:,4:5]/all_params["domain"]["domain_range"]['x'][1]
        loss_Tx_xl = jnp.mean(loss_Tx_xl**2)

        loss_u_yu = all_params["data"]['u_ref']*b_out3[:,0:1]
        loss_u_yu = jnp.mean(loss_u_yu**2)
        loss_u_yl = all_params["data"]['u_ref']*b_out4[:,0:1]
        loss_u_yl = jnp.mean(loss_u_yl**2)
        loss_v_yu = all_params["data"]['v_ref']*b_out3[:,1:2]
        loss_v_yu = jnp.mean(loss_v_yu**2)
        loss_v_yl = all_params["data"]['v_ref']*b_out4[:,1:2]
        loss_v_yl = jnp.mean(loss_v_yl**2)
        loss_w_yu = all_params["data"]['w_ref']*b_out3[:,2:3]
        loss_w_yu = jnp.mean(loss_w_yu**2)
        loss_w_yl = all_params["data"]['w_ref']*b_out4[:,2:3]
        loss_w_yl = jnp.mean(loss_w_yl**2)
        loss_Ty_yu = all_params["data"]['T_ref']*b_out3_y[:,4:5]/all_params["domain"]["domain_range"]['y'][1]
        loss_Ty_yu = jnp.mean(loss_Ty_yu**2)
        loss_Ty_yl = all_params["data"]['T_ref']*b_out4_y[:,4:5]/all_params["domain"]["domain_range"]['y'][1]
        loss_Ty_yl = jnp.mean(loss_Ty_yl**2)

        loss_con = uxs[:,0:1] + uys[:,1:2] + uzs[:,2:3]
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = uts[:,0:1] + uvwp[:,0:1]*uxs[:,0:1] + uvwp[:,1:2]*uys[:,0:1] + uvwp[:,2:3]*uzs[:,0:1] + uxs[:,3:4] - all_params["data"]["viscosity"]*(uxxs[:,0:1]+uyys[:,0:1]+uzzs[:,0:1])
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = uts[:,1:2] + uvwp[:,0:1]*uxs[:,1:2] + uvwp[:,1:2]*uys[:,1:2] + uvwp[:,2:3]*uzs[:,1:2] + uys[:,3:4] - all_params["data"]["viscosity"]*(uxxs[:,1:2]+uyys[:,1:2]+uzzs[:,1:2])
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = uts[:,2:3] + uvwp[:,0:1]*uxs[:,2:3] + uvwp[:,1:2]*uys[:,2:3] + uvwp[:,2:3]*uzs[:,2:3] + uzs[:,3:4] - all_params["data"]["viscosity"]*(uxxs[:,2:3]+uyys[:,2:3]+uzzs[:,2:3]) - uvwp[:,4:5]
        loss_NS3 = jnp.mean(loss_NS3**2)

        loss_vor1 = uyts[:,2:3]-uzts[:,1:2] + uvwp[:,0:1]*(uxys[:,2:3] - uxzs[:,1:2]) + uvwp[:,1:2]*(uyys[:,2:3] - uyzs[:,1:2]) + uvwp[:,2:3]*(uyzs[:,2:3] - uzzs[:,1:2]) \
         - (uys[:,2:3] - uzs[:,1:2])*uxs[:,0:1] - (uzs[:,0:1] - uxs[:,2:3])*uys[:,0:1] - (uxs[:,1:2] - uys[:,0:1])*uzs[:,0:1] \
         - all_params["data"]["viscosity"]*(uxxys[:,2:3] + uyyys[:,2:3] + uyzzs[:,2:3] - uxxzs[:,1:2] - uyyzs[:,1:2] - uzzzs[:,1:2]) - uys[:,4:5]
        loss_vor1 = jnp.mean(loss_vor1**2)

        loss_vor2 = (uys[:,2:3] - uzs[:,1:2])*uxs[:,1:2] + (uzs[:,0:1] - uxs[:,2:3])*uys[:,1:2] + (uxs[:,1:2] - uys[:,0:1])*uzs[:,1:2] \
         + all_params["data"]["viscosity"]*(uxxzs[:,0:1] + uyyzs[:,0:1] + uzzzs[:,0:1] - uxxxs[:,2:3] - uxyys[:,2:3] - uxzzs[:,2:3]) \
         - uzts[:,0:1] + uxts[:,2:3] - uvwp[:,0:1]*(uxzs[:,0:1] - uxxs[:,2:3]) - uvwp[:,1:2]*(uyzs[:,0:1] - uxys[:,2:3]) - uvwp[:,2:3]*(uzzs[:,0:1] - uxzs[:,2:3]) - uxs[:,4:5]
        loss_vor2 = jnp.mean(loss_vor2**2)
        

        loss_ENR = uts[:,4:5] + uvwp[:,0:1]*uxs[:,4:5] + uvwp[:,1:2]*uys[:,4:5] + uvwp[:,2:3]*uzs[:,4:5] - all_params["data"]["viscosity"]*(uxxs[:,4:5]+uyys[:,4:5]+uzzs[:,4:5])/7
        loss_ENR = jnp.mean(loss_ENR**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*loss_ENR + weights[8]*(loss_T_zu + loss_T_zb) + \
                    weights[9]*(loss_vor1 + loss_vor2) + \
                    weights[10]*(loss_u_xu + loss_u_xl + loss_v_xu + loss_v_xl + loss_w_xu + loss_w_xl + loss_u_yu + 
                                loss_u_yl + loss_v_yu + loss_v_yl + loss_w_yu + loss_w_yl) + \
                    weights[11]*(loss_Tx_xu + loss_Tx_xl + loss_Ty_yu + loss_Ty_yl)
        return total_loss
    @staticmethod
    def Loss_report(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
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
        keys = ['u_ref', 'v_ref', 'w_ref', 'p_ref', 'T_ref']
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
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
        out, out_t = equ_func1(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)

        uvwp = jnp.concatenate([out[:,k:(k+1)]*all_params["data"][keys[k]] for k in range(len(keys))],1)

        uts = jnp.concatenate([out_t[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
        uxs = jnp.concatenate([out_x[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1] for k in range(len(keys))],1)
        uys = jnp.concatenate([out_y[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uzs = jnp.concatenate([out_z[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uxts = jnp.concatenate([out_xt[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
        uyts = jnp.concatenate([out_yt[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
        uzts = jnp.concatenate([out_zt[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['t'][1] for k in range(len(keys))],1)
        uxxs = jnp.concatenate([out_xx[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1] for k in range(len(keys))],1)
        uxys = jnp.concatenate([out_xy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uxzs = jnp.concatenate([out_xz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uyys = jnp.concatenate([out_yy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uyzs = jnp.concatenate([out_yz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uzzs = jnp.concatenate([out_zz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uxxxs = jnp.concatenate([out_xxx[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1] for k in range(len(keys))],1)
        uxxys = jnp.concatenate([out_xxy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uxxzs = jnp.concatenate([out_xxz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uxyys = jnp.concatenate([out_xyy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uyyys = jnp.concatenate([out_yyy[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1] for k in range(len(keys))],1)
        uyyzs = jnp.concatenate([out_yyz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uxzzs = jnp.concatenate([out_xzz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['x'][1]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uyzzs = jnp.concatenate([out_yzz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['y'][1]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)
        uzzzs = jnp.concatenate([out_zzz[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1]/all_params["domain"]["domain_range"]['z'][1] for k in range(len(keys))],1)


        p_out = model_fns(all_params, particles)

        b_out1, b_out1_x = equ_func1(all_params, boundaries[0], jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(boundaries[0].shape[0],1)),model_fns)
        b_out2, b_out2_x = equ_func1(all_params, boundaries[1], jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(boundaries[1].shape[0],1)),model_fns)
        b_out3, b_out3_y = equ_func1(all_params, boundaries[2], jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(boundaries[2].shape[0],1)),model_fns)
        b_out4, b_out4_y = equ_func1(all_params, boundaries[3], jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(boundaries[3].shape[0],1)),model_fns)
        b_out5 = model_fns(all_params, boundaries[4])  
        b_out6 = model_fns(all_params, boundaries[5])



        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_u_zu = all_params["data"]['u_ref']*b_out5[:,0:1]
        loss_u_zu = jnp.mean(loss_u_zu**2)
        loss_u_zl = all_params["data"]['u_ref']*b_out6[:,0:1]
        loss_u_zl = jnp.mean(loss_u_zl**2)
        loss_v_zu = all_params["data"]['v_ref']*b_out5[:,1:2]
        loss_v_zu = jnp.mean(loss_v_zu**2)
        loss_v_zl = all_params["data"]['v_ref']*b_out6[:,1:2]
        loss_v_zl = jnp.mean(loss_v_zl**2)
        loss_w_zu = all_params["data"]['w_ref']*b_out5[:,2:3]
        loss_w_zu = jnp.mean(loss_w_zu**2)
        loss_w_zl = all_params["data"]['w_ref']*b_out6[:,2:3]
        loss_w_zl = jnp.mean(loss_w_zl**2)
        loss_T_zu = all_params["data"]['T_ref']*b_out5[:,4:5] + all_params["data"]['T_ref']
        loss_T_zu = jnp.mean(loss_T_zu**2)
        loss_T_zb = all_params["data"]['T_ref']*b_out6[:,4:5] - all_params["data"]['T_ref']
        loss_T_zb = jnp.mean(loss_T_zb**2)

        loss_u_xu = all_params["data"]['u_ref']*b_out1[:,0:1]
        loss_u_xu = jnp.mean(loss_u_xu**2)
        loss_u_xl = all_params["data"]['u_ref']*b_out2[:,0:1]
        loss_u_xl = jnp.mean(loss_u_xl**2)
        loss_v_xu = all_params["data"]['v_ref']*b_out1[:,1:2]
        loss_v_xu = jnp.mean(loss_v_xu**2)
        loss_v_xl = all_params["data"]['v_ref']*b_out2[:,1:2]
        loss_v_xl = jnp.mean(loss_v_xl**2)
        loss_w_xu = all_params["data"]['w_ref']*b_out1[:,2:3]
        loss_w_xu = jnp.mean(loss_w_xu**2)
        loss_w_xl = all_params["data"]['w_ref']*b_out2[:,2:3]
        loss_w_xl = jnp.mean(loss_w_xl**2)
        loss_Tx_xu = all_params["data"]['T_ref']*b_out1_x[:,4:5]/all_params["domain"]["domain_range"]['x'][1]
        loss_Tx_xu = jnp.mean(loss_Tx_xu**2)
        loss_Tx_xl = all_params["data"]['T_ref']*b_out2_x[:,4:5]/all_params["domain"]["domain_range"]['x'][1]
        loss_Tx_xl = jnp.mean(loss_Tx_xl**2)

        loss_u_yu = all_params["data"]['u_ref']*b_out3[:,0:1]
        loss_u_yu = jnp.mean(loss_u_yu**2)
        loss_u_yl = all_params["data"]['u_ref']*b_out4[:,0:1]
        loss_u_yl = jnp.mean(loss_u_yl**2)
        loss_v_yu = all_params["data"]['v_ref']*b_out3[:,1:2]
        loss_v_yu = jnp.mean(loss_v_yu**2)
        loss_v_yl = all_params["data"]['v_ref']*b_out4[:,1:2]
        loss_v_yl = jnp.mean(loss_v_yl**2)
        loss_w_yu = all_params["data"]['w_ref']*b_out3[:,2:3]
        loss_w_yu = jnp.mean(loss_w_yu**2)
        loss_w_yl = all_params["data"]['w_ref']*b_out4[:,2:3]
        loss_w_yl = jnp.mean(loss_w_yl**2)
        loss_Ty_yu = all_params["data"]['T_ref']*b_out3_y[:,4:5]/all_params["domain"]["domain_range"]['y'][1]
        loss_Ty_yu = jnp.mean(loss_Ty_yu**2)
        loss_Ty_yl = all_params["data"]['T_ref']*b_out4_y[:,4:5]/all_params["domain"]["domain_range"]['y'][1]
        loss_Ty_yl = jnp.mean(loss_Ty_yl**2)

        loss_con = uxs[:,0:1] + uys[:,1:2] + uzs[:,2:3]
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = uts[:,0:1] + uvwp[:,0:1]*uxs[:,0:1] + uvwp[:,1:2]*uys[:,0:1] + uvwp[:,2:3]*uzs[:,0:1] + uxs[:,3:4] - all_params["data"]["viscosity"]*(uxxs[:,0:1]+uyys[:,0:1]+uzzs[:,0:1])
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = uts[:,1:2] + uvwp[:,0:1]*uxs[:,1:2] + uvwp[:,1:2]*uys[:,1:2] + uvwp[:,2:3]*uzs[:,1:2] + uys[:,3:4] - all_params["data"]["viscosity"]*(uxxs[:,1:2]+uyys[:,1:2]+uzzs[:,1:2])
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = uts[:,2:3] + uvwp[:,0:1]*uxs[:,2:3] + uvwp[:,1:2]*uys[:,2:3] + uvwp[:,2:3]*uzs[:,2:3] + uzs[:,3:4] - all_params["data"]["viscosity"]*(uxxs[:,2:3]+uyys[:,2:3]+uzzs[:,2:3]) - uvwp[:,4:5]
        loss_NS3 = jnp.mean(loss_NS3**2)

        loss_vor1 = uyts[:,2:3]-uzts[:,1:2] + uvwp[:,0:1]*(uxys[:,2:3] - uxzs[:,1:2]) + uvwp[:,1:2]*(uyys[:,2:3] - uyzs[:,1:2]) + uvwp[:,2:3]*(uyzs[:,2:3] - uzzs[:,1:2]) \
         - (uys[:,2:3] - uzs[:,1:2])*uxs[:,0:1] - (uzs[:,0:1] - uxs[:,2:3])*uys[:,0:1] - (uxs[:,1:2] - uys[:,0:1])*uzs[:,0:1] \
         - all_params["data"]["viscosity"]*(uxxys[:,2:3] + uyyys[:,2:3] + uyzzs[:,2:3] - uxxzs[:,1:2] - uyyzs[:,1:2] - uzzzs[:,1:2]) - uys[:,4:5]
        loss_vor1 = jnp.mean(loss_vor1**2)

        loss_vor2 = (uys[:,2:3] - uzs[:,1:2])*uxs[:,1:2] + (uzs[:,0:1] - uxs[:,2:3])*uys[:,1:2] + (uxs[:,1:2] - uys[:,0:1])*uzs[:,1:2] \
         + all_params["data"]["viscosity"]*(uxxzs[:,0:1] + uyyzs[:,0:1] + uzzzs[:,0:1] - uxxxs[:,2:3] - uxyys[:,2:3] - uxzzs[:,2:3]) \
         - uzts[:,0:1] + uxts[:,2:3] - uvwp[:,0:1]*(uxzs[:,0:1] - uxxs[:,2:3]) - uvwp[:,1:2]*(uyzs[:,0:1] - uxys[:,2:3]) - uvwp[:,2:3]*(uzzs[:,0:1] - uxzs[:,2:3]) - uxs[:,4:5]
        loss_vor2 = jnp.mean(loss_vor2**2)
        

        loss_ENR = uts[:,4:5] + uvwp[:,0:1]*uxs[:,4:5] + uvwp[:,1:2]*uys[:,4:5] + uvwp[:,2:3]*uzs[:,4:5] - all_params["data"]["viscosity"]*(uxxs[:,4:5]+uyys[:,4:5]+uzzs[:,4:5])/7
        loss_ENR = jnp.mean(loss_ENR**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 + weights[7]*loss_ENR + weights[8]*(loss_T_zu + loss_T_zb) + \
                    weights[9]*(loss_vor1 + loss_vor2) + \
                    weights[10]*(loss_u_xu + loss_u_xl + loss_v_xu + loss_v_xl + loss_w_xu + loss_w_xl + loss_u_yu + 
                                loss_u_yl + loss_v_yu + loss_v_yl + loss_w_yu + loss_w_yl) + \
                    weights[11]*(loss_Tx_xu + loss_Tx_xl + loss_Ty_yu + loss_Ty_yl)
        return total_loss, loss_u, loss_v, loss_w, loss_con, loss_NS1, loss_NS2, loss_NS3, loss_ENR, loss_T_zu, loss_T_zb
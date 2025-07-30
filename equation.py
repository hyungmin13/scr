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

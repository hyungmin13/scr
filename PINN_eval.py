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
import h5py
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
        all_params = self.c.problem.constraints(all_params)
        valid_data = self.c.problem.exact_solution(all_params)
        model_fn = c.network.network_fn
        return all_params, model_fn, train_data, valid_data
    
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

def Derivatives(dynamic_params, all_params, g_batch, model_fns):
    keys = ['u_ref', 'v_ref', 'w_ref', 'u_ref']

    all_params["network"]["layers"] = dynamic_params
    out, out_t = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_x, out_xx = equ_func(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_y, out_yy = equ_func(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out_z, out_zz = equ_func(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)    
    uvwp = np.concatenate([pred[:,k:(k+1)]*all_params["data"][keys[k]] for k in range(len(keys))],1)
    uvwp[:,-1] = 1.185*uvwp[:,-1]
    uxs = np.concatenate([out_x[:,k:(k+1)]*all_params["data"][keys[k]] for k in range(len(keys))],1)
    uys = np.concatenate([out_y[:,k:(k+1)]*all_params["data"][keys[k]] for k in range(len(keys))],1)
    uzs = np.concatenate([out_z[:,k:(k+1)]*all_params["data"][keys[k]] for k in range(len(keys))],1)
    deriv_mat = np.concatenate([np.expand_dims(uxs,1),np.expand_dims(uys,1),np.expand_dims(uzs,1),2])
    vor_mag = np.sqrt((deriv_mat[:,1,2]-deriv_mat[:,2,1])**2+
                      (deriv_mat[:,2,0]-deriv_mat[:,0,2])**2+
                      (deriv_mat[:,0,1]-deriv_mat[:,1,0])**2)
    Q = 0.5 * sum(np.abs(0.5 * (deriv_mat[:, i, j] + deriv_mat[:, j, i]))**2 - 
                  np.abs(0.5 * (deriv_mat[:, i, i] - deriv_mat[:, j, j]))**2 
                  for i in range(3) for j in range(3) if i != j)
    return uvwp, vor_mag, Q
def shear_stress(streamwise_vel, normal_axis):
    return np.mean(streamwise_vel[:,1,:,:]-streamwise_vel[:,0,:,:])/(normal_axis[1,0,0]-normal_axis[0,0,0])
def friction_vel(streamwise_vel, normal_axis, viscosity):
    return np.sqrt(viscosity*np.mean(streamwise_vel[:,1,:,:]-streamwise_vel[:,0,:,:])/(normal_axis[1,0,0]-normal_axis[0,0,0]))
def viscous_length(streamwise_vel, normal_axis, viscosity):
    return viscosity/np.sqrt(viscosity*np.mean(streamwise_vel[:,1,:,:]-streamwise_vel[:,0,:,:])/(normal_axis[1,0,0]-normal_axis[0,0,0]))
def y_plus(streamwise_vel, normal_axis, viscosity):
    return normal_axis[:,0,0]/(viscosity/np.sqrt(viscosity*np.mean(streamwise_vel[:,1,:,:]-streamwise_vel[:,0,:,:])/(normal_axis[1,0,0]-normal_axis[0,0,0])))
def u_plus(streamwise_vel, streamwise_vel_ground, normal_axis, viscosity):
    return np.mean(np.mean(np.mean(streamwise_vel,0),1),1)/(np.sqrt(viscosity*np.mean(streamwise_vel_ground[:,1,:,:]-streamwise_vel_ground[:,0,:,:])/(normal_axis[1,0,0]-normal_axis[0,0,0])))

#%%
if __name__ == "__main__":
    from PINN_domain import *
    from PINN_trackdata import *
    from PINN_network import *
    from PINN_constants import *
    from PINN_problem import *

    u_tau = 15*10**(-6)/36.2/10**(-6)
    u_ref_n = 4.9968*10**(-2)/u_tau
    delta = 36.2*10**(-6)
    x_ref_n = 1.0006*10**(-3)/delta

    checkpoint_fol = "TBL_run_06"
    path = "results/summaries/"
    with open(path+checkpoint_fol+'/constants_'+ str(checkpoint_fol) +'.pickle','rb') as f:
        a = pickle.load(f)
    a['data_init_kwargs']['path'] = '/scratch/hyun/TBL/'
    a['problem_init_kwargs']['path_s'] = '/scratch/hyun/Ground/'
    with open(path+checkpoint_fol+'/constants_'+ str(checkpoint_fol) +'.pickle','wb') as f:
        pickle.dump(a,f)

    values = list(a.values())

    c = Constants(run = values[0],
                domain_init_kwargs = values[1],
                data_init_kwargs = values[2],
                network_init_kwargs = values[3],
                problem_init_kwargs = values[4],
                optimization_init_kwargs = values[5],)
    run = PINN(c)

    with open(run.c.model_out_dir + "saved_dic_340000.pkl","rb") as f:
        a = pickle.load(f)
    all_params, model_fn, train_data, valid_data = run.test()

    model = Model(all_params["network"]["layers"], model_fn)
    all_params["network"]["layers"] = from_state_dict(model, a).params
#%% pred_list_total 는 모든 시간 [i] 에 대해 중심으로 부터의 거리 [j] 에 따른 속도, 압력의 예측값(eval_from_center)
    output_shape = (213,141,61)
    total_spatial_error = []
    pos_unnorm = np.concatenate([valid_data['pos'][:,i:i+1]*all_params["domain"]["in_max"][0,i] 
                                 for i in range(4)],1).reshape((-1,)+output_shape+(4,))

    pos_from_center = pos_unnorm[1,:,:,:,:].reshape(-1,4) - np.array([0,0.028,0.006,0.00212]).reshape(-1,4)
    pos_from_center = np.sqrt(pos_from_center[:,1]**2+pos_from_center[:,2]**2+pos_from_center[:,3]**2)
    pos_unnorm = pos_unnorm.reshape(-1,4)
    counts, bins, bars = plt.hist(pos_from_center, bins=50)

    indexes = []
    for i in range(bins.shape[0]-1):
        index = np.where((pos_from_center<bins[i+1])&(pos_from_center>=bins[i]))
        indexes.append(index[0])

    vel_sub_t_list = []
    pos_sub_t_list = []
    
    pos_sub_t_unnorm_list = []
    for j in range(50):
        vel_sub_list = []
        pos_sub_list = []
        pos_sub_unnorm_list = []
        print(j)
        for i in range(len(indexes)):
            #valid_data['vel'][:,3:4] = valid_data['vel'][:,3:4]*1.185
            vel_sub_list.append(valid_data['vel'][213*141*61*j:213*141*61*(j+1),:][indexes[i],:])
            pos_sub_unnorm_list.append(pos_unnorm[213*141*61*j:213*141*61*(j+1),:][indexes[i],:])
            pos_sub_list.append(valid_data['pos'][213*141*61*j:213*141*61*(j+1),:][indexes[i],:])
        vel_sub_t_list.append(vel_sub_list)
        pos_sub_t_list.append(pos_sub_list)
        pos_sub_t_unnorm_list.append(pos_sub_unnorm_list)

    ext_p_list = []
    for i in range(len(vel_sub_t_list)):
        ext_p_array = np.mean(np.concatenate(vel_sub_t_list[i],0)[:,3])
        ext_p_list.append(ext_p_array)
    for i in range(len(vel_sub_t_list)):
        for j in range(len(vel_sub_t_list[i])):
            vel_sub_t_list[i][j][:,3] = vel_sub_t_list[i][j][:,3] - ext_p_list[i]

    pred_list_total = []
    keys = ['u_ref', 'v_ref', 'w_ref', 'u_ref']
    for i in range(len(pos_sub_t_list)):
        pred_list = []
        print(i)
        for j in range(len(indexes)):
            pos_unnorm = np.concatenate(pos_sub_t_list[i][j])
            pred = model_fn(all_params, pos_sub_t_list[i][j])
            pred_unnorm = np.concatenate([pred[:,k:(k+1)]*all_params["data"][keys[k]] for k in range(len(keys))],1)
            pred_unnorm[:,-1] = 1.185*pred_unnorm[:,-1]
            pred_list.append(pred_unnorm)
        pred_list_total.append(pred_list)
    test_p_list = []
    for i in range(len(pred_list_total)):
        test_p_array = np.mean(np.concatenate(pred_list_total[i],0)[:,-1])
        test_p_list.append(test_p_array)
    for i in range(len(pred_list_total)):
        for j in range(len(pred_list_total[i])):
            pred_list_total[i][j][:,-1] = pred_list_total[i][j][:,-1] - test_p_list[i]

    vel_error_t_list = []
    pre_error_t_list = []
    dist = getattr(st,"norm")
    for i in range(len(pred_list_total)):
        vel_error_list = []
        pre_error_list = []
        print(i)
        for j in range(len(pred_list_total[i])):
            vel_error_list.append(np.sqrt((np.sqrt(pred_list_total[i][j][:,0]**2+pred_list_total[i][j][:,1]**2+pred_list_total[i][j][:,2]**2)-np.sqrt(vel_sub_t_list[i][j][:,0]**2+vel_sub_t_list[i][j][:,1]**2+vel_sub_t_list[i][j][:,2]**2))**2)/np.sqrt(vel_sub_t_list[i][j][:,0]**2+vel_sub_t_list[i][j][:,1]**2+vel_sub_t_list[i][j][:,2]**2))
            pre_error_list.append(np.sqrt((np.sqrt(pred_list_total[i][j][:,3]**2)-np.sqrt(vel_sub_t_list[i][j][:,3]**2))**2))
        vel_error_t_list.append(vel_error_list)
        pre_error_t_list.append(pre_error_list)

    import scipy.stats as st
    dist = getattr(st,"norm")
    mean_error_list = []
    for i in range(len(vel_error_t_list[10])):
        mean_std = dist.fit(vel_error_t_list[10][i])
        mean_error_list.append(mean_std[0])
    mean_error_list = np.array(mean_error_list)

    import scipy.stats as st
    dist = getattr(st,"norm")
    mean_error_list = []
    for i in range(len(pre_error_t_list[10])):
        mean_std = dist.fit(pre_error_t_list[10][i])
        mean_error_list.append(mean_std[0])
    mean_error_list = np.array(mean_error_list)

    test_vels = np.concatenate(pred_list_total[25])
    test_vels_ext = np.concatenate(vel_sub_t_list[25])

    f = np.concatenate([(test_vels[:,0]-test_vels_ext[:,0]).reshape(-1,1),
                        (test_vels[:,1]-test_vels_ext[:,1]).reshape(-1,1),
                        (test_vels[:,2]-test_vels_ext[:,2]).reshape(-1,1)],1)
    div = np.concatenate([test_vels_ext[:,0].reshape(-1,1), test_vels_ext[:,1].reshape(-1,1),
                          test_vels_ext[:,2].reshape(-1,1)],1)
    print(np.linalg.norm(f, ord='fro')/np.linalg.norm(div,ord='fro'))
    print(np.linalg.norm(test_vels[:,3]-test_vels_ext[:,3])/np.linalg.norm(test_vels_ext[:,3]))
#%%
    """
    from scipy.integrate import tplquad
    L = 0.056
    W = 0.012
    H = 0.00424
    def sphere_condition(x,y,z):
        return x**2+y**2+z**2<r**2
    def intergrand(x,y,z):
        return 1 if sphere_condition(x,y,z) else 0
    volumes = []
    for i in range(len(bins)):
        print(i)
        r = bins[i]
        volume, error = tplquad(intergrand, -L/2, L/2, lambda x:-W/2, lambda x:W/2, lambda x,y:-H/2, lambda x,y:H/2)
        volumes.append(volume)
        sub_volumes = np.array(volumes[1:])-np.array(volumes[:-1])
        c_vol_avg = np.array(counts)/np.array(sub_volumes)
        plt.hist(bins[:-1], bins,weights=c_vol_avg)
        plt.show()
    with open("datas/sub_volumes.pkl","wb") as f:
        pickle.dump(sub_volumes,f)
    f.close()
    with open("datas/counts.pkl","wb") as f:
        pickle.dump(counts,f)
    f.close()
    """
#%%
    temporal_error_vel_list = []
    temporal_error_pre_list = []
    for j in range(51):
        print(j)
        pred = model_fn(all_params, valid_data['pos'].reshape((51,)+output_shape+(4,))[j,:,:,:,:].reshape(-1,4))
        output_keys = ['u', 'v', 'w', 'p']
        output_unnorm = [all_params["data"]['u_ref'],all_params["data"]['v_ref'],
                        all_params["data"]['w_ref'],1.185*all_params["data"]['u_ref']]
        outputs = {output_keys[i]:pred[:,i]*output_unnorm[i] for i in range(len(output_keys))}
        outputs['p'] = outputs['p'] - np.mean(outputs['p'])
        output_ext = {output_keys[i]:valid_data['vel'].reshape((51,)+output_shape+(4,))[j,:,:,:,i].reshape(-1) for i in range(len(output_keys))}
        output_ext['p'] = output_ext['p'] - np.mean(output_ext['p'])


        f = np.concatenate([(outputs['u']-output_ext['u']).reshape(-1,1), 
                            (outputs['v']-output_ext['v']).reshape(-1,1), 
                            (outputs['w']-output_ext['w']).reshape(-1,1)],1)
        div = np.concatenate([output_ext['u'].reshape(-1,1), output_ext['v'].reshape(-1,1), 
                            output_ext['w'].reshape(-1,1)],1)

        temporal_error_pre_list.append(np.linalg.norm(outputs['p'] - output_ext['p'])/np.linalg.norm(output_ext['p']))
        temporal_error_vel_list.append(np.linalg.norm(f, ord='fro')/np.linalg.norm(div,ord='fro'))    
#%%
    temporal_error = np.concatenate([np.array(temporal_error_vel_list).reshape(-1,1),
                                     np.array(temporal_error_pre_list).reshape(-1,1)],1)
#%%
    plt.plot(temporal_error_vel_list)
    plt.show()
    plt.plot(temporal_error_pre_list)
    plt.show()
#%%
    pred = model_fn(all_params, valid_data['pos'].reshape(51,213,141,61,4)[25,:,:,:,:].reshape(-1,4))
    output_keys = ['u', 'v', 'w', 'p']
    output_unnorm = [all_params["data"]['u_ref'],all_params["data"]['v_ref'],
                     all_params["data"]['w_ref'],1.185*all_params["data"]['u_ref']]
    outputs = {output_keys[i]:pred[:,i]*output_unnorm[i] for i in range(len(output_keys))}
    outputs['p'] = outputs['p'] - np.mean(outputs['p'])
    output_ext = {output_keys[i]:valid_data['vel'].reshape(51,213,141,61,4)[25,:,:,:,i].reshape(-1) for i in range(len(output_keys))}
    output_ext['p'] = output_ext['p'] - np.mean(output_ext['p'])


    f = np.concatenate([(outputs['u']-output_ext['u']).reshape(-1,1), 
                        (outputs['v']-output_ext['v']).reshape(-1,1), 
                        (outputs['w']-output_ext['w']).reshape(-1,1)],1)
    div = np.concatenate([output_ext['u'].reshape(-1,1), output_ext['v'].reshape(-1,1), 
                          output_ext['w'].reshape(-1,1)],1)

    print(np.linalg.norm(outputs['p'] - output_ext['p'])/np.linalg.norm(output_ext['p']))
    print(np.linalg.norm(f, ord='fro')/np.linalg.norm(div,ord='fro'))
#%%
    temporal_error_vel_list = []
    temporla_error_pre_list = []
    for j in range(51):
        print(j)
        pred = model_fn(all_params, valid_data['pos'].reshape(51,129,129,129,4)[j,:,:,:,:].reshape(-1,4))
        output_keys = ['u', 'v', 'w', 'p']
        output_unnorm = [all_params["data"]['u_ref'],all_params["data"]['v_ref'],
                        all_params["data"]['w_ref'],1.185*all_params["data"]['u_ref']]
        outputs = {output_keys[i]:pred[:,i]*output_unnorm[i] for i in range(len(output_keys))}
        outputs['p'] = outputs['p'] - np.mean(outputs['p'])
        output_ext = {output_keys[i]:valid_data['vel'].reshape(51,129,129,129,4)[j,:,:,:,i].reshape(-1) for i in range(len(output_keys))}
        output_ext['p'] = output_ext['p'] - np.mean(output_ext['p'])


        f = np.concatenate([(outputs['u']-output_ext['u']).reshape(-1,1), 
                            (outputs['v']-output_ext['v']).reshape(-1,1), 
                            (outputs['w']-output_ext['w']).reshape(-1,1)],1)
        div = np.concatenate([output_ext['u'].reshape(-1,1), output_ext['v'].reshape(-1,1), 
                            output_ext['w'].reshape(-1,1)],1)

        temporal_error_vel_list.append(np.linalg.norm(outputs['p'] - output_ext['p'])/np.linalg.norm(output_ext['p']))
        temporla_error_pre_list.append(np.linalg.norm(f, ord='fro')/np.linalg.norm(div,ord='fro'))
#%%
filenames_test = sorted(glob('/scratch/hyun/HIT/Particles/'+'*.npy'))[int(0*1250):int(50/1250*1250)+1][::4]
#%%
print(np.expand_dims(np.expand_dims(outputs['u'],1),1).shape)
#%% 
    checkpoint_fols = ["run02","run_k4_timeskip2","run_k4_timeskip4",
                       "run_k4_timeskip6","run_k4_timeskip8"]
    path = "/home/bussard/hyun_sh/TBL_PINN/test12/results/summaries/"
    All_p_data = []
    All_v_data = []
    All_param_list = []
    All_model_list = []
    for checkpoint_fol in checkpoint_fols:
        with open(path+checkpoint_fol+'/constants_'+ str(checkpoint_fol) +'.pickle','rb') as f:
            a = pickle.load(f)
        a['data_init_kwargs']['path'] = '/scratch/hyun/HIT/Particles/'
        a['problem_init_kwargs']['path_s'] = '/scratch/hyun/HIT/IsoturbFlow.mat'
        with open(path+checkpoint_fol+'/constants_'+ str(checkpoint_fol) +'.pickle','wb') as f:
            pickle.dump(a,f)
        values = list(a.values())
        c = Constants(run = values[0],
                      domain_init_kwargs = values[1],
                      data_init_kwargs = values[2],
                      network_init_kwargs = values[3],
                      problem_init_kwargs = values[4],
                      optimization_init_kwargs = values[5],)
        run = PINN(c)
        all_params, model_fn, train_data, valid_data = run.test()
        All_p_data.append(np.concatenate([train_data['pos'][:,i:i+1]*all_params['domain']['in_max'][0,i] 
                                          for i in range(train_data['pos'].shape[1])],1))
        All_v_data.append(train_data['vel'])
        All_param_list.append(all_params)
        All_model_list.append(model_fn)
#%%
    All_count = []
    for i in range(len(All_p_data)):
        p_pos_centered = All_p_data[i] - np.array([0,0.05,0.05,0.05]).reshape(1,-1)
        p_pos_distance = np.sqrt(p_pos_centered[:,1]**2+p_pos_centered[:,2]**2+p_pos_centered[:,3]**2)
        counts, bins, bars = plt.hist(p_pos_distance, bins=50)
        volume = 4/3*np.pi*(bins[1:]**3-bins[:-1]**3)
        new_counts = counts/volume
        All_count.append(new_counts/np.max(new_counts))
    All_count = np.vstack(All_count)   

#%%
    dist = getattr(st,"norm")
    mean_count = []
    for i in range(All_count.shape[1]):
        meanstd = dist.fit(All_count[:,i])
        mean_count.append(meanstd[0])
    mean_count = np.array(mean_count)
    plt.hist(bins[:-1], bins,weights=1-mean_count)
    plt.xlabel("distance from center [m]")
    plt.ylabel("Particle counts per volume")
    plt.show()

#%%
    All_mean = []
    All_mag = []
    checkpoint_fols = ["run02","run_k4_timeskip2","run_k4_timeskip4",
                       "run_k4_timeskip6","run_k4_timeskip8"]
    ref_keys = ['u_ref', 'v_ref', 'w_ref']
    output_keys = ['u', 'v', 'w', 'p']
    for k in range(len(All_p_data)):
        pos_sep_domain = []
        vel_sep_domain = []
        p_pos_centered = All_p_data[k] - np.array([0,0.05,0.05,0.05]).reshape(1,-1)
        p_pos_distance = np.sqrt(p_pos_centered[:,1]**2+p_pos_centered[:,2]**2+p_pos_centered[:,3]**2)
        for i in range(bins.shape[0]-1):
            index = np.where((p_pos_distance<bins[i+1])&(p_pos_distance>bins[i]))
            p_pos_sub = np.concatenate([All_p_data[k][index[0],n:(n+1)]/All_param_list[k]["domain"]["in_max"][0,n] 
                                        for n in range(4)],1, dtype=np.float32)
            pos_sep_domain.append(p_pos_sub)
            vel_sep_domain.append(All_v_data[k][index[0],:])
        outs = []
        with open("results/models/"+checkpoint_fols[k]+'/saved_dic_640000.pkl','rb') as f:
            checkpoint = pickle.load(f)
        model = Model(All_param_list[k]["network"]["layers"], All_model_list[k])
        All_param_list[k]["network"]["layers"] = from_state_dict(model, checkpoint).params

        for j in range(len(pos_sep_domain)):
            out = []
            out_ = All_model_list[k](All_param_list[k],pos_sep_domain[j][:,:])
            All_result = {output_keys[i]:All_param_list[k]["data"][ref_keys[i]]*out_[:,i] for i in range(len(output_keys)-1)}
            All_result[output_keys[-1]] = 1.185*All_param_list[k]["data"][ref_keys[0]]*out_[:,-1]
            All_result[output_keys[-1]] = All_result[output_keys[-1]] - np.mean(All_result[output_keys[-1]])
            outs.append(All_result)
        mean_error_list = []
        for i in range(len(outs)):
            vel_mag = np.sqrt(outs[i]['u']**2+outs[i]['v']**2+outs[i]['w']**2)
            vel_mag_true = np.sqrt(vel_sep_domain[i][:,0]**2+vel_sep_domain[i][:,1]**2+vel_sep_domain[i][:,2]**2)
            vel_error = np.sqrt((vel_mag-vel_mag_true)**2)#/np.sqrt(vel_mag_true**2)
            mean_error_list.append(np.mean(vel_error))
        All_mean.append(np.array(mean_error_list))


#%%
num = 3
fig, host = plt.subplots()
par1 = host.twinx()
host.hist(bins[:-1], bins,weights=1-All_count[num,:], color='tab:blue', label=r"1 - particle count")
par1.plot((bins[:-1]+bins[1:])/2,All_mean[num],'r', label=r'Velocity NRMSE')
host.set_ylabel("1-particle counts per volume")
par1.set_ylabel("velocity NRMSE")
host.set_xlabel("distance from the center [m]")
host.yaxis.label.set_color('tab:blue')
par1.yaxis.label.set_color('r')
tkw = dict(size=4, width=1.5)
host.tick_params(axis='y', colors='tab:blue', **tkw)
par1.tick_params(axis='y', colors='r', **tkw)
#host.legend(loc="upper left")
plt.title('Downselection factor k$_x$ = 4, timeskip = 8')
plt.legend(loc="upper left")
plt.show()
#%%
correlation_coef_vel = [np.corrcoef(1-All_count[i,:],All_mean[i])[1,0] for i in range(len(All_mean))]
k_val = [1, 2, 4, 6, 8]
plt.plot(k_val,correlation_coef_vel,label='vel')
plt.scatter(k_val, correlation_coef_vel)
plt.xlabel('Downselection factor, k$_x$')
plt.ylabel('Correlation coefficient')
plt.ylim(0,1.0)
plt.legend()
plt.show()
#%%
print(run.c.model_out_dir)     
#%% Load exact fluid structures 
    filename = '/home/bussard/hyun_sh/TBL_PINN/data/Q_vor.txt'
    with open(filename, 'r') as f:
        data = np.loadtxt(filename, skiprows=0)
    fluid_structure = {'vor':data[:,23].reshape(129,129,129),
                       'Q':data[:,19].reshape(129,129,129)}
#%% For full training checkpoints #############
    from glob import glob
    save_list = glob("/home/bussard/hyun_sh/TBL_PINN/test12/results/models/" + checkpoint_fol + "/*.pkl")
    #save_list = save_list.sort()
    vel_error_list = []
    pre_error_list = []

#%%
    all_params, model_fn, train_data, valid_data = run.test()
    output_keys = ['u', 'v', 'w', 'p']
    output_unnorm = [all_params["data"]['u_ref'],all_params["data"]['v_ref'],
                     all_params["data"]['w_ref'],1.185*all_params["data"]['u_ref']]

    for i in range(len(save_list)):
        print(save_list[i])
        with open(save_list[i],"rb") as f:
            a = pickle.load(f)
        model = Model(all_params["network"]["layers"], model_fn)
        all_params["network"]["layers"] = from_state_dict(model, a).params
        pred = model_fn(all_params, valid_data['pos'].reshape(51,129,129,129,4)[25,:,:,:,:].reshape(-1,4))
        outputs = {output_keys[i]:pred[:,i]*output_unnorm[i] for i in range(len(output_keys))}
        outputs['p'] = outputs['p'] - np.mean(outputs['p'])
        output_ext = {output_keys[i]:valid_data['vel'].reshape(51,129,129,129,4)[25,:,:,:,i].reshape(-1) for i in range(len(output_keys))}
        output_ext['p'] = output_ext['p'] - np.mean(output_ext['p'])

        pre_error_list.append(np.linalg.norm(outputs['p'] - output_ext['p'])/
                              np.linalg.norm(output_ext['p']))
        f = np.concatenate([(outputs['u']-output_ext['u']).reshape(-1,1), 
                            (outputs['v']-output_ext['v']).reshape(-1,1), 
                            (outputs['w']-output_ext['w']).reshape(-1,1)],1)
        div = np.concatenate([output_ext['u'].reshape(-1,1), output_ext['v'].reshape(-1,1), 
                              output_ext['w'].reshape(-1,1)],1)
        vel_error_list.append(np.linalg.norm(f, ord='fro')/np.linalg.norm(div,ord='fro'))
#%%
    time_ = np.arange(len(save_list))

    plt.plot(np.array(vel_error_list[100:]))
    plt.show()
    plt.plot(np.array(pre_error_list[100:]))
    plt.show()
#%%
    with open("/home/bussard/hyun_sh/TBL_PINN/data/"+"vel_error_k32_300_10.pkl","wb") as f:
        pickle.dump(vel_error_list[72:],f)
    f.close()
    with open("/home/bussard/hyun_sh/TBL_PINN/data/"+"pre_error_k32_300_10.pkl","wb") as f:
        pickle.dump(pre_error_list[72:],f)
    f.close()
# %%

#%%
import numpy as np
import matplotlib.pyplot as plt
import pyff3
from glob import glob
from Tecplot_mesh import tecplot_Mesh
import os
class FlowFit:
    @staticmethod
    def init_params(*args):
        raise NotImplementedError
class FF_linear(FlowFit):
    def __init__(self, all_params):
        self.all_params = all_params

    @staticmethod
    def init_params(**kwargs):
        flowfit_params = {}
        for key, value in kwargs.items():
            flowfit_params[key] = value
        return flowfit_params
    
    @staticmethod
    def FF3_python(gx_min, gx_max, gy_min, gy_max, gz_min, gz_max, h, hf, ep, downsample, pos, vel, timestep):
        recon_params = pyff3.ReconParameters()
        recon_params.min = pyff3.Vec3(gx_min, gy_min, gz_min)
        recon_params.max = pyff3.Vec3(gx_max, gy_max, gz_max)
        recon_params.h = pyff3.Vec3(h, h, h)

        recon_params.hfpen1 = hf
        recon_params.hfpen2 = hf
        recon_params.epsilon1 = ep
        recon_params.epsilon2 = ep
        recon_params.m = 3 # defaults to 6

        sampling_params = pyff3.SamplingParameters()
        sampling_params.min = pyff3.Vec3(gx_min, gy_min, gz_min)
        sampling_params.max = pyff3.Vec3(gx_max, gy_max, gz_max)
        recon_params.h = pyff3.Vec3(h, h, h)


        vfield = pyff3.reconstruct_linear(recon_params, pos, vel, out_file="ff_coeff/lv"+ str(downsample) + f"/HIT{timestep:02d}.hdf5")

        return 
if __name__=="__main__":
    h = 0.003578381815466162
    hf = 0.003171071836210107
    num = 50
    FF_linear.FF3_python(h, hf, 1e-3, 64, num)


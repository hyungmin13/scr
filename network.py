#%%
import jax.numpy as jnp
from jax import random
import numpy as np
class Network:
    @staticmethod
    def init_params(*args):
        raise NotImplementedError

class MLP(Network):
    def __init__(self, all_params):
        self.all_params = all_params
        

    @staticmethod
    def init_params(key, layer_sizes, network_name):
        key_network = random.PRNGKey(key)
        keys = random.split(key_network, len(layer_sizes)-1)
        params = [eval(network_name)._random_layer_params(k, m, n)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        network_params = {"layers": params, "layer_sizes":layer_sizes, "network_name":network_name}
        return network_params
    
    @staticmethod
    def _random_layer_params(key, m, n):
        "Create a random layer parameters"

        w_key, b_key = random.split(key)
        w = random.normal(w_key, (m, n))
        b = jnp.zeros((1,n))
        g = jnp.ones((1,n))
        return w,b,g

    @staticmethod
    def network_fn(all_params, x):
        params = all_params["network"]["layers"]
        inmin = all_params["domain"]["in_min"]
        inmax = all_params["domain"]["in_max"]
        inmean = all_params["data"]["in_mean"]
        instd = all_params["data"]["in_std"]
        x = 2*(x - inmin)/(inmax-inmin) - 1
        #x = (x-inmean)/instd
        for w, b, g in params[:-1]:
            x = g*jnp.dot(x,w/jnp.linalg.norm(w,axis=0, keepdims=True)) + b
            x = jnp.tanh(x)
        w, b, g = params[-1]
        x = jnp.dot(x,w/jnp.linalg.norm(w,axis=0, keepdims=True)) + b
        return x

    def network_fn3(self, x):
        params = self.all_params["network2"]["layers"]
        inmin = self.all_params["domain"]["in_min"]
        inmax = self.all_params["domain"]["in_max"]
        x = 2*(x - inmin)/(inmax-inmin) - 1
        for w, b, g in params[:-1]:
            x = g*jnp.dot(x,w/jnp.linalg.norm(w,axis=0, keepdims=True)) + b
            x = jnp.tanh(x)
        w, b, g = params[-1]
        x = jnp.dot(x,w/jnp.linalg.norm(w,axis=0, keepdims=True)) + b
        return x

class MLP_FF(Network):
    def __init__(self, all_params):
        self.all_params = all_params

    @staticmethod
    def init_params(key, layer_sizes, network_name, input_dim=4, ff_dim=64, ff_sigma=5.0):
        key_network = random.PRNGKey(key)
        key_B, key_layers = random.split(key_network)

        # Fourier projection matrix
        # shape: (input_dim, ff_dim)
        B = ff_sigma * random.normal(key_B, (input_dim, ff_dim))

        keys = random.split(key_layers, len(layer_sizes) - 1)
        params = [eval(network_name)._random_layer_params(k, m, n)
                  for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]

        network_params = {
            "layers": params,
            "layer_sizes": layer_sizes,
            "network_name": network_name,
            "B": B,
            "ff_dim": ff_dim,
            "ff_sigma": ff_sigma,
        }
        return network_params

    @staticmethod
    def _random_layer_params(key, m, n):
        w_key, b_key = random.split(key)
        w = random.normal(w_key, (m, n))
        b = jnp.zeros((1, n))
        g = jnp.ones((1, n))
        return w, b, g

    @staticmethod
    def fourier_features(x, B):
        # x shape: (N, input_dim)
        # B shape: (input_dim, ff_dim)
        x_proj = jnp.dot(x, B)   # (N, ff_dim)
        x_ff = jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
        return x_ff

    @staticmethod
    def network_fn(all_params, x):
        net_params = all_params["network1"]
        params = net_params["layers"]
        B = net_params["B"]

        inmin = all_params["domain"]["in_min"]
        inmax = all_params["domain"]["in_max"]

        # normalize to [-1, 1]
        x = 2 * (x - inmin) / (inmax - inmin) - 1

        # Fourier feature mapping
        x = MLP_FF.fourier_features(x, B)

        for w, b, g in params[:-1]:
            x = g * jnp.dot(x, w / jnp.linalg.norm(w, axis=0, keepdims=True)) + b
            x = jnp.tanh(x)

        w, b, g = params[-1]
        x = jnp.dot(x, w / jnp.linalg.norm(w, axis=0, keepdims=True)) + b

        return x

class MLPSiren(Network):
    def __init__(self, all_params):
        self.all_params = all_params

    @staticmethod
    def init_params(key, layer_sizes, network_name, omega0=30.0):
        key_network = random.PRNGKey(key)
        keys = random.split(key_network, len(layer_sizes) - 1)

        params = []
        for i, (k, m, n) in enumerate(zip(keys, layer_sizes[:-1], layer_sizes[1:])):
            first_layer = (i == 0)
            params.append(
                eval(network_name)._random_layer_params(k, m, n, first_layer, omega0)
            )

        network_params = {
            "layers": params,
            "layer_sizes": layer_sizes,
            "network_name": network_name,
            "omega0": omega0,
        }
        return network_params

    @staticmethod
    def _random_layer_params(key, m, n, first_layer=False, omega0=30.0):
        w_key, b_key = random.split(key)

        if first_layer:
            # SIREN first layer init
            w = random.uniform(w_key, (m, n), minval=-1.0/m, maxval=1.0/m)
        else:
            # SIREN hidden layer init
            limit = jnp.sqrt(6.0 / m) / omega0
            w = random.uniform(w_key, (m, n), minval=-limit, maxval=limit)

        b = jnp.zeros((1, n))
        return w, b

    @staticmethod
    def network_fn(all_params, x):
        net_params = all_params["network1"]
        params = net_params["layers"]
        omega0 = net_params["omega0"]

        inmin = all_params["domain"]["in_min"]
        inmax = all_params["domain"]["in_max"]

        # normalize input
        x = 2 * (x - inmin) / (inmax - inmin) - 1

        # hidden layers
        for i, (w, b) in enumerate(params[:-1]):
            if i == 0:
                x = jnp.sin(omega0 * (jnp.dot(x, w) + b))
            else:
                x = jnp.sin(jnp.dot(x, w) + b)

        # output layer: usually linear
        w, b = params[-1]
        x = jnp.dot(x, w) + b

        return x

if __name__=="__main__":
    from domain import *

    all_params = {"network":{}, "domain":{}}
    domain_range = {'t':(0,8), 'x':(0,1.2), 'y':(0,1.2), 'z':(0,1)}
    frequency = 3000
    grid_size = [9, 200, 200, 200]
    bound_keys = ['ic', 'bcxu', 'bcxl', 'bcyu', 'bcyl', 'bczu', 'bczl']
    all_params["domain"] = Domain.init_params(domain_range, frequency, grid_size, bound_keys)
    grids, all_params = Domain.sampler(all_params)
    x = jnp.ones((100,4))
    key = random.PRNGKey(0)
    layer_sizes = [4,16,32,16,1]
    network = MLP
    all_params["network"] = network.init_params(key, layer_sizes)


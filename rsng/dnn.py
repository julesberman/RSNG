import haiku as hk
import jax
import jax.numpy as jnp


def build_nn(width, depth, period):
    def net(x):
        layers = [Periodic_Linear(width, period=period), Rational()]
        for _ in range(depth-2):
            layers.append(hk.Linear(width))
            layers.append(Rational())
        y = hk.Sequential([*layers, hk.Linear(1)])(x)
        return jnp.squeeze(y)
    return net


def unraveler(f, unravel, axis=0):
    """
    util to deal with pytrees, will wrap net function
    """
    def wrapper(*args, **kwargs):
        val = args[axis]
        if (type(val) != dict):
            args = list(args)
            args[axis] = unravel(val)
            args = tuple(args)
        return f(*args, **kwargs)
    return wrapper


def init_net(net, key, dim):

    trans = hk.without_apply_rng(hk.transform(net))
    net_apply, net_init = trans.apply, trans.init
    theta_init = net_init(key, jnp.zeros(dim))
    theta_init_flat, unravel = jax.flatten_util.ravel_pytree(theta_init)
    u_scalar = unraveler(net_apply, unravel)

    return u_scalar, theta_init_flat, unravel


class Rational(hk.Module):
    """
    Rational activation function
    ref: Nicolas Boullé, Yuji Nakatsukasa, and Alex Townsend,
        Rational neural networks,
        arXiv preprint arXiv:2004.01902 (2020).

    Source: https://github.com/yonesuke/RationalNets/blob/main/src/rationalnets/rational.py

    """

    def __init__(self, p=3, name=None):
        super().__init__(name=name)
        self.p = 3
        self.alpha_init = lambda *args: jnp.array(
            [1.1915, 1.5957, 0.5, 0.0218][:p+1])
        self.beta_init = lambda *args: jnp.array([2.383, 0.0, 1.0][:p])

    def __call__(self, x):
        alpha = hk.get_parameter(
            "alpha", shape=[self.p+1], dtype=x.dtype, init=self.alpha_init)
        beta = hk.get_parameter(
            "beta", shape=[self.p], dtype=x.dtype, init=self.beta_init)
        return jnp.polyval(alpha, x)/jnp.polyval(beta, x)


class Periodic_Linear(hk.Module):
    def __init__(self, nodes, period, name=None):
        super().__init__(name=name)
        self.nodes = nodes
        self.period = period

    def __call__(self, x):
        d, m = x.shape[-1], self.nodes
        w_init = hk.initializers.TruncatedNormal(1.0)
        a = hk.get_parameter("a", shape=[m, d], dtype=x.dtype, init=w_init)
        phi = hk.get_parameter("phi", shape=[m, d], dtype=x.dtype, init=w_init)
        c = hk.get_parameter("c", shape=[m, d], dtype=x.dtype, init=w_init)
        return jnp.sum(a*jnp.cos((jnp.pi*2/self.period)*x+phi)+c, axis=1)

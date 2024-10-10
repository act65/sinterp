import jax.numpy as jnp
from jax import random
import jax

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.solvers.utils import sample_joint


class Coupling():
    def __call__(self, x, y=None, unused_key=None):
        """
        Return a coupling between two distributions.

        Returns:
            x, y: the coupled samples
        """
        raise NotImplementedError
    
class IndependentCoupling(Coupling):
    """
    p(x, y) = p(x) p(y)
    """
    def __call__(self, x, y, unused_key=None):
        return x, y
    
class ConditionalCoupling(Coupling):
    """
    p(x, y) = p(x | y) p(y)
    """
    def __init__(self, cond_fn):
        self.cond_fn = cond_fn

    def __call__(self, x, y=None, unused_key=None):
        y = self.cond_fn(x)
        return x, y

class EMDCoupling(Coupling):
    def __init__(self):
        pass

    def sample(self, x, y, key):
        a = jnp.ones(x.shape[0]) / x.shape[0]
        b = jnp.ones(y.shape[0]) / y.shape[0]
        geom = pointcloud.PointCloud(x, y)
        prob = linear_problem.LinearProblem(geom, a, b)

        solver = sinkhorn.Sinkhorn()
        out = solver(prob)

        i, j = sample_joint(key, out.matrix)
        return x[i], y[j]

class Rectification(Coupling):
    def __init__(self, flow):
        self.flow = flow

    def __call__(self, x, y, t, unused_key=None):
        yield x, self.flow.forward(x, t)
        yield self.flow.backward(y, t), y
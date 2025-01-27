import jax.numpy as jnp
import jax
import numpy as np
from numpyro.distributions import Poisson
from jax import random

class BaseLikelihood():
    
    def __init__(self):    
        pass

    def generate_realization(self, data):
        pass

    def log_likelihood(self, observed_data, model_data):
        pass
    
    def __call__(self, observed_data, model_data):
        return jax.scipy.special.logsumexp(self.log_likelihood(observed_data, model_data)) - jnp.log(model_data.shape[0])


class PoissonPhotonLikelihood(BaseLikelihood):

    def generate_realization(self, data):
        return Poisson(data).sample(random.PRNGKey(np.random.randint(0,100000)))

    def log_likelihood(self, observed_data, model_data):
        return jnp.sum(jnp.atleast_2d(Poisson(model_data).log_prob(observed_data)), axis=1)
    

class MixturePhotonLikelihood(BaseLikelihood):

    # TODO: Implement this class

    def __init__(self):
        super().__init__()

class GaussianStrainLikelihood(BaseLikelihood):

    # TODO: Implement this class

    def __init__(self):
        super().__init__()
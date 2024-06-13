import tensorflow as tf 
import tensorflow_probability as tfp
tfd = tfp.distributions


# Initialize a 2-batch of 3-variate Gaussians.
mvn = tfd.MultivariateNormalDiag(
    loc        = [[1., 2, 3],
                 [11, 22, 33]],           
    scale_diag = [[1., 2, 3],
                 [0.5, 1, 1.5]])  

# Evaluate this on a two observations, each in `R^3`, returning a length-2
# vector.
x = [[-1., 0, 1],
     [-11, 0, 11.]]   # shape: [2, 3].
mvn.prob(x).numpy()           # shape: [2]

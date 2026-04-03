import numpy as np

rng = np.random.default_rng()

print(rng.random())

print(rng.standard_normal(10))

print(rng.integers(0, 10, 5))
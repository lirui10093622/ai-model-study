import numpy as np

for key, value in np.__dict__.items():
    print(key, value, sep=": ")
import numpy as np


np.random.seed()
d_eta = np.round(np.exp(np.random.uniform(np.log(1e-2), np.log(1e-5))), 5)
g_eta = np.round(np.exp(np.random.uniform(np.log(1e-2), np.log(1e-5))), 5)
d_ch = np.random.choice([4, 8, 16, 32, 64])
g_ch = np.random.choice([4, 8, 16, 32, 64])
d_layers = np.random.choice(list(range(1, 7)))
g_layers = np.random.choice(list(range(2, 9)))
lambda_ = np.round(np.exp(np.random.uniform(np.log(1), np.log(10000))), 0).astype("int32")
mu = np.round(np.random.uniform(), 2)

d = {"d_eta": d_eta,
     "g_eta": g_eta,
     "d_ch": d_ch,
     "g_ch": g_ch,
     "d_layers": d_layers,
     "g_layers": g_layers,
     "lambda": lambda_,
     "mu": mu}

print(d)
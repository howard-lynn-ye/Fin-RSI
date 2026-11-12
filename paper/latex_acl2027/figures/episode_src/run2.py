import numpy as np

def solve(prices, split_factors):
  p = prices * split_factors
  m = p.shift(1) / p.shift(21)
  w = np.sign(m - 1) / p.shape[1]
  return w.fillna(0.0)

import time
import jax
import jax.numpy as jnp

min_val = -10
max_val = 10
n_samples = 100

num_features = 10

seed = 0

(
  data_key,
  real_w_key,
  real_b_key,
  w_key,
  b_key,
) = jax.random.split(jax.random.key(seed), 5)

real_w = jax.random.uniform(real_w_key, shape=(num_features,), minval=min_val, maxval=max_val)
real_b = jax.random.uniform(real_b_key, minval=min_val, maxval=max_val)

lr = 1e-2


def generate_data():
  x = jax.random.uniform(
    data_key,
    shape=(n_samples, num_features),
    minval=min_val,
    maxval=max_val,
  )
  y = x @ real_w + real_b
  return x, y

def initialize_weights():
  w = jax.random.uniform(w_key, shape=(num_features,))
  b = jax.random.uniform(b_key)
  return w, b

def forward(w, b, x):
  return x @ w + b

def l2(w, b, x, y):
  return jnp.mean((forward(w, b, x) - y)**2)

@jax.jit
def update(w, b, x, y):
  l2_grad = jax.value_and_grad(l2, argnums=(0, 1))
  loss, (dl_dw, dl_db) = l2_grad(w, b, x, y)
  w -= lr*dl_dw
  b -= lr*dl_db
  return loss, w, b


x, y = generate_data()
w, b = initialize_weights()

st = time.time()
for _ in range(1000):
  loss, w, b = update(w, b, x, y)
  print(f'\nLoss: {loss}')
  print(f'real w: {real_w}')
  print(f'w: {w}')
  print(f'real b: {real_b}')
  print(f'b: {b}')
print(f'\nDone in {time.time()-st} seconds.')


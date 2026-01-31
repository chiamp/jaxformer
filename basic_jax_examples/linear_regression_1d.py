import time
import jax
import jax.numpy as jnp


real_w = 5
real_b = -1

min_val = -10
max_val = 10
n_samples = 10
seed = 0

lr = 1e-2


data_key, w_key, b_key = jax.random.split(jax.random.key(seed), 3)


def generate_data():
  x = jax.random.uniform(
    data_key,
    shape=(n_samples,),
    minval=min_val,
    maxval=max_val,
  )
  y = real_w * x + real_b
  return x, y

def initialize_weights():
  w = jax.random.uniform(w_key)
  b = jax.random.uniform(b_key)
  return w, b

def forward(w, b, x):
  return w*x + b

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
  print(f'Loss: {loss}\tw:{w}\tb:{b}')
print(f'\nDone in {time.time()-st} seconds.')


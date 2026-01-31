import jax
import jax.numpy as jnp

from config import Config


def initialize_arrays(
    key: jax.Array,
    shape: tuple[int, ...],
    num_arrays: int,
) -> tuple[list[jax.Array], jax.Array]:
  arrays: list[jax.Array] = []
  for _ in range(num_arrays):
    key_to_use, key = jax.random.split(key)
    arrays.append(
      jax.random.normal(
        key_to_use,
        shape=shape,
      ) * 0.02
    )
  return arrays, key


def initialize_parameters(config: Config, key: jax.Array) -> dict[str, jax.Array | dict[str, dict[str, jax.Array]]]:
  params: dict[str, jax.Array | dict[str, dict[str, jax.Array]]] = {}

  encoder_params: dict[str, dict[str, jax.Array]] = {
    'attention': {},
    'dense': {},
  }
  decoder_params: dict[str, dict[str, jax.Array]] = {
    'masked_attention': {},
    'cross_attention': {},
    'dense': {},
  }

  (
    v,
    n_e,
    n_d,
    d_model,
    d_k,
    d_v,
    h,
    d_ff,
  ) = (
    config.vocab_size,
    config.num_encoder_layers,
    config.num_decoder_layers,
    config.num_embedding_features,
    config.num_query_key_features,
    config.num_value_features,
    config.num_heads,
    config.num_inner_dense_features,
  )

  # Embedding params
  (params['embedding'],), key = initialize_arrays(key, shape=(v, d_model), num_arrays=1)

  # Attention params
  (
    encoder_params['attention']['query'],
    encoder_params['attention']['key'],
  ), key = initialize_arrays(key, shape=(n_e, d_model, d_k*h), num_arrays=2)

  (
    decoder_params['masked_attention']['query'],
    decoder_params['cross_attention']['query'],
    decoder_params['masked_attention']['key'],
    decoder_params['cross_attention']['key'],
  ), key = initialize_arrays(key, shape=(n_d, d_model, d_k*h), num_arrays=4)

  (encoder_params['attention']['value'],), key = initialize_arrays(key, shape=(n_e, d_model, d_v*h), num_arrays=1)

  (
    decoder_params['masked_attention']['value'],
    decoder_params['cross_attention']['value'],
  ), key = initialize_arrays(key, shape=(n_d, d_model, d_v*h), num_arrays=2)

  # Params for projecting attention outputs back to the embedding features
  (encoder_params['attention']['out_proj'],), key = initialize_arrays(key, shape=(n_e, d_v*h, d_model), num_arrays=1)

  (
    decoder_params['masked_attention']['out_proj'],
    decoder_params['cross_attention']['out_proj'],
  ), key = initialize_arrays(key, shape=(n_d, d_v*h, d_model), num_arrays=2)

  # Layer norm params applied after applying attention or dense inference (gamma=1, beta=0)
  encoder_params['attention']['layernorm_gamma'] = jnp.ones((n_e, d_model))
  encoder_params['attention']['layernorm_beta'] = jnp.zeros((n_e, d_model))
  encoder_params['dense']['layernorm_gamma'] = jnp.ones((n_e, d_model))
  encoder_params['dense']['layernorm_beta'] = jnp.zeros((n_e, d_model))

  decoder_params['masked_attention']['layernorm_gamma'] = jnp.ones((n_d, d_model))
  decoder_params['masked_attention']['layernorm_beta'] = jnp.zeros((n_d, d_model))
  decoder_params['cross_attention']['layernorm_gamma'] = jnp.ones((n_d, d_model))
  decoder_params['cross_attention']['layernorm_beta'] = jnp.zeros((n_d, d_model))
  decoder_params['dense']['layernorm_gamma'] = jnp.ones((n_d, d_model))
  decoder_params['dense']['layernorm_beta'] = jnp.zeros((n_d, d_model))

  # Dense params
  (encoder_params['dense']['w1'],), key = initialize_arrays(key, shape=(n_e, d_model, d_ff), num_arrays=1)
  (decoder_params['dense']['w1'],), key = initialize_arrays(key, shape=(n_d, d_model, d_ff), num_arrays=1)
  encoder_params['dense']['b1'] = jnp.zeros((n_e, d_ff))
  decoder_params['dense']['b1'] = jnp.zeros((n_d, d_ff))

  (encoder_params['dense']['w2'],), key = initialize_arrays(key, shape=(n_e, d_ff, d_model), num_arrays=1)
  (decoder_params['dense']['w2'],), key = initialize_arrays(key, shape=(n_d, d_ff, d_model), num_arrays=1)
  encoder_params['dense']['b2'] = jnp.zeros((n_e, d_model))
  decoder_params['dense']['b2'] = jnp.zeros((n_d, d_model))

  params['encoder'] = encoder_params
  params['decoder'] = decoder_params

  params['final_bias'] = jnp.zeros((v,))  # add to the projected vocab output

  return params

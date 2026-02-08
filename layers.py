'''
Attention is all you need paper: https://arxiv.org/pdf/1706.03762

Comments about dimension/shape:
- v: vocabulary size
- layer: number of layers in the encoder or decoder
- batch: each entry in the batch is a separate input sentence
- sequence: each entry is a token (e.g. character, word, etc.) that makes up the sentence
  - e.g. (3, 4) in a (batch, sequence) matrix represents the 4th token in the 3rd input sentence sample
  - Thus, the length of sequence is the number of tokens in that sentence
- d_model: the number of embedding features
- d_k: the number of features for the query/key
- d_v: the number of features for the value
- d_ff: the number of features for the inner layer of the dense network
- head: the number of attention heads
NOTE: d_k = d_v
NOTE: d_k*head = d_v*head = d_model
'''


import functools

import jax
import jax.numpy as jnp


def position_encode_single(index: int, num_embedding_features: int) -> jax.Array:
  # Position encode a single input token, instead of the full sequence.
  embedding_feature_vector = (jnp.arange(num_embedding_features) // 2) * 2  # vector [0, 0, 2, 2, 4, 4, ...], of shape (d_model,)

  positional_embeddings = index / (10000**(embedding_feature_vector/num_embedding_features))
  positional_embeddings = positional_embeddings[None, :] # shape (1, d_model)

  positional_embeddings = positional_embeddings.at[:,::2].set(jnp.sin(positional_embeddings[:,::2]))
  positional_embeddings = positional_embeddings.at[:,1::2].set(jnp.cos(positional_embeddings[:,1::2]))

  return positional_embeddings  # (1, d_model)

def position_encode(sequence_length: int, num_embedding_features: int) -> jax.Array:
  position_vector = jnp.arange(sequence_length)  # (sequence,)
  embedding_feature_vector = (jnp.arange(num_embedding_features) // 2) * 2  # vector [0, 0, 2, 2, 4, 4, ...], of shape (d_model,)
  positional_embeddings = jnp.einsum(
    'p,d->pd',
    position_vector,
    1 / (10000**(embedding_feature_vector/num_embedding_features))
  )  # outer product, shape (sequence, d_model)

  positional_embeddings = positional_embeddings.at[:,::2].set(jnp.sin(positional_embeddings[:,::2]))
  positional_embeddings = positional_embeddings.at[:,1::2].set(jnp.cos(positional_embeddings[:,1::2]))

  return positional_embeddings  # (sequence, d_model)


def get_1d_causal_mask(sequence_length: int, index: int) -> jax.Array:
  mask = jnp.arange(sequence_length)
  # Insert dummy dimension since the first dimension represents the sequence_q dimension,
  # which would be a single token if we were using kv cache.
  return jnp.where(mask > index, -1e9, 0)[None, :]  # (1, sequence_k)

def get_decoder_only_1d_causal_mask(sequence_length: int, position_index_vector: jax.Array) -> jax.Array:
  # position_index_vector is of shape (batch,), indicating the position of each sample in the batch (since it may be uneven since we are right-padding our sequences)
  # get an index array per row/sample
  mask = jnp.einsum('bs,s->bs', jnp.ones((position_index_vector.shape[0], sequence_length)), jnp.arange(sequence_length))  # (batch, sequence_k)
  # Broadcast `(batch, sequence_k) > (batch, 1)` so that the we check if each value in a row i in mask is strictly larger than the corresponding ith element in position_index_vector.
  # It's still a 1d mask in the sense that we don't have a (sequence, sequence) mask, it's just a different 1D mask for each sample in the batch.
  return jnp.where(mask > position_index_vector[:, None], -1e9, 0)  # (batch, sequence_k)

def get_2d_causal_mask(sequence_length: int) -> jax.Array:
  # Upper triangle mask (NOTE: the diagonal is not masked).
  mask = jnp.ones((sequence_length, sequence_length))
  return jnp.triu(mask, k=1) * (-1e9)  # (sequence_q, sequence_k)


def project_attention(x: jax.Array, attention_matrix: jax.Array, *, num_heads: int, num_query_key_value_features: int) -> jax.Array:
  # x is of shape (batch, sequence, d_model)
  # attention_matrix is of shape (d_model, head*(d_k or d_v))
  batch_size = x.shape[0]
  sequence_length = x.shape[1]

  x = jnp.einsum('bsi,io->bso', x, attention_matrix)  # (batch, sequence, head*(d_k or d_v)))
  x = x.reshape(batch_size, sequence_length, num_heads, num_query_key_value_features)   # (batch, sequence, head, d_k or d_v)
  x = jnp.einsum('bshk->bhsk', x)  # (batch, head, sequence, d_k or d_v)
  return x


def compute_attention_output(
    x: jax.Array,
    q_proj: jax.Array,
    k_proj: jax.Array,
    v_proj: jax.Array,
    params: dict[str, jax.Array],
    *,
    num_query_key_features: int,
    num_embedding_features: int,
    padding_mask: jax.Array | None,
    causal_mask: jax.Array | None,
    decoder_only_causal_mask: jax.Array | None = None,  # non-uniform masking on a per batch basis
) -> tuple[jax.Array, jax.Array]:
  # x is of shape (batch, sequence, d_model)
  # q_proj is of shape (batch, head, sequence_q, d_k)
  # k_proj is of shape (batch, head, sequence_k, d_k)
  # v_proj is of shape (batch, head, sequence_k, d_v)
  # padding_mask is of shape (batch, sequence_k)
  # The subscripts for the `sequence` term is only relevant for cross attention, otherwise ignore since sequence_q==sequence_k then.
    # - sequence_q is derived from the sentence length of inputs fed into the decoder
    # - sequence_k is derived from the sentence length of inputs fed into the encoder

  batch_size = x.shape[0]
  sequence_length = x.shape[1]

  # Calculate QK_T attention scores
  k_proj = jnp.einsum('bhsk->bhks', k_proj)  # (batch, head, d_k, sequence_k)
  attention_scores = jnp.einsum('bhid,bhdo->bhio', q_proj, k_proj) / jnp.sqrt(num_query_key_features)  # (batch, head, sequence_q, sequence_k)
  if padding_mask is not None:  # apply padding mask (only used in encoder attention layers and cross attention layers in the decoder)
    attention_scores += padding_mask[:, None, None, :]  # add dummy dimensions to padding mask (batch, 1, 1, sequence_k) and apply it
  if causal_mask is not None:  # apply causal mask (used only in masked attention layers in the decoder)
    assert decoder_only_causal_mask is None
    attention_scores += causal_mask[None, None, :, :]  # (1, 1, sequence, sequence)
  if decoder_only_causal_mask is not None:  # per batch basis causal masking
    assert causal_mask is None
    attention_scores += decoder_only_causal_mask[:, None, None, :]  # (batch, 1, 1, sequence)
  attention_scores = jax.nn.softmax(attention_scores)  # softmax over the last dimension

  # Calculate (Q_KT) @ V attention weighted sum output
  attention_output = jnp.einsum('bhio,bhov->bhiv', attention_scores, v_proj)  # (batch, head, sequence_q, d_v)
  attention_output = jnp.einsum('bhsv->bshv', attention_output)  # (batch, sequence_q, head, d_v)
  attention_output = attention_output.reshape(
    batch_size,
    sequence_length,
    num_embedding_features,
  )  # (batch, sequence_q, d_model)

  # Calculate output projection
  output_proj = jnp.einsum('bsi,io->bso', attention_output, params['out_proj'])  # (batch, sequence_q, d_model)

  # Residual connection
  x += output_proj  # (batch, sequence_q, d_model)

  # Layer norm: scale with gamma and add beta bias
  x = (x - jnp.mean(x, axis=-1, keepdims=True)) / jnp.sqrt(jnp.var(x, axis=-1, keepdims=True) + 1e-6)  # add small eps to prevent dividing by 0
  x = jnp.einsum('btf,f->btf', x, params['layernorm_gamma'])  # (batch, sequence_q, d_model)
  x += params['layernorm_beta']

  return x, attention_scores


def compute_dense_output(x: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
  # Dense inference
  dense_hidden_output = jnp.einsum('bsm,mf->bsf', x, params['w1'])  # (batch, sequence, d_ff)
  dense_hidden_output = jax.nn.relu(dense_hidden_output + params['b1'])
  dense_output = jnp.einsum('bsf,fm->bsm', dense_hidden_output, params['w2'])  # (batch, sequence, d_model)
  dense_output += params['b2']

  # Residual connection
  x += dense_output  # (batch, sequence, d_model)

  # Layer norm: scale with gamma and add beta bias
  x = (x - jnp.mean(x, axis=-1, keepdims=True)) / jnp.sqrt(jnp.var(x, axis=-1, keepdims=True) + 1e-6)  # add small eps to prevent dividing by 0
  x = jnp.einsum('btf,f->btf', dense_output, params['layernorm_gamma'])  # (batch, sequence, d_model)
  x += params['layernorm_beta']

  return x


def encoder_single_layer(
    x: jax.Array,
    encoder_params: dict[str, dict[str, jax.Array]],
    *,
    padding_mask: jax.Array,
    num_heads: int,
    num_query_key_features: int,
    num_value_features: int,
    num_embedding_features: int,
) -> tuple[jax.Array, jax.Array]:
  # Single pass of one encoder layer: attention -> layernorm -> dense -> layernorm

  # Project attention matrices
  q_proj = project_attention(
    x,
    encoder_params['attention']['query'],
    num_heads=num_heads,
    num_query_key_value_features=num_query_key_features,
  )  # (batch, head, sequence, d_k)
  k_proj = project_attention(
    x,
    encoder_params['attention']['key'],
    num_heads=num_heads,
    num_query_key_value_features=num_query_key_features,
  )  # (batch, head, sequence, d_k)
  v_proj = project_attention(
    x,
    encoder_params['attention']['value'],
    num_heads=num_heads,
    num_query_key_value_features=num_value_features,
  )  # (batch, head, sequence, d_v)

  # Compute attention output
  x, attention_scores = functools.partial(
    compute_attention_output,
    num_query_key_features=num_query_key_features,
    num_embedding_features=num_embedding_features,
    padding_mask=padding_mask,
    causal_mask=None,
  )(x, q_proj, k_proj, v_proj, encoder_params['attention'])  # (batch, sequence, d_model)

  # Compute dense output
  x = compute_dense_output(x, encoder_params['dense'])  # (batch, sequence, d_model)

  # attention scores is of shape (batch, head, sequence, sequence)
  return x, attention_scores


def decoder_single_layer(
    x: jax.Array,
    decoder_params: dict[str, dict[str, jax.Array]],
    *,
    encoder_output: jax.Array,
    cross_padding_mask: jax.Array,
    num_heads: int,
    num_query_key_features: int,
    num_value_features: int,
    num_embedding_features: int,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  # Single pass of one decoder layer: masked_attention -> layernorm -> cross_attention -> layernorm -> dense -> layernorm
  # x is of shape (batch, sequence, d_model)

  # Project attention matrices
  q_proj = project_attention(
    x,
    decoder_params['masked_attention']['query'],
    num_heads=num_heads,
    num_query_key_value_features=num_query_key_features,
  )  # (batch, head, sequence, d_k)
  k_proj = project_attention(
    x,
    decoder_params['masked_attention']['key'],
    num_heads=num_heads,
    num_query_key_value_features=num_query_key_features,
  )  # (batch, head, sequence, d_k)
  v_proj = project_attention(
    x,
    decoder_params['masked_attention']['value'],
    num_heads=num_heads,
    num_query_key_value_features=num_value_features,
  )  # (batch, head, sequence, d_v)

  # Compute masked attention output
  x, masked_attention_scores = functools.partial(
    compute_attention_output,
    num_query_key_features=num_query_key_features,
    num_embedding_features=num_embedding_features,
    padding_mask=None,  # causal mask will already mask out future tokens, so padding mask is not needed.
    causal_mask=get_2d_causal_mask(sequence_length=x.shape[1]),
  )(x, q_proj, k_proj, v_proj, decoder_params['masked_attention'])  # (batch, sequence, d_model)

  # Project cross attention matrices
  q_proj = project_attention(
    x,
    decoder_params['cross_attention']['query'],
    num_heads=num_heads,
    num_query_key_value_features=num_query_key_features,
  )  # (batch, head, sequence, d_k)
  k_proj = project_attention(
    encoder_output,
    decoder_params['cross_attention']['key'],
    num_heads=num_heads,
    num_query_key_value_features=num_query_key_features,
  )  # (batch, head, sequence, d_k)
  v_proj = project_attention(
    encoder_output,
    decoder_params['cross_attention']['value'],
    num_heads=num_heads,
    num_query_key_value_features=num_value_features,
  )  # (batch, head, sequence, d_v)

  # Compute cross attention output
  x, cross_attention_scores = functools.partial(
    compute_attention_output,
    num_query_key_features=num_query_key_features,
    num_embedding_features=num_embedding_features,
    padding_mask=cross_padding_mask,  # (batch, sequence), derived from input sentence batch to encoder
    causal_mask=None,
  )(x, q_proj, k_proj, v_proj, decoder_params['cross_attention'])  # (batch, sequence, d_model)

  # Compute dense output
  x = compute_dense_output(x, decoder_params['dense'])  # (batch, sequence, d_model)

  return x, {
    'masked_attention_scores': masked_attention_scores,  # (batch, head, sequence, sequence)
    'cross_attention_scores': cross_attention_scores,  # (batch, head, sequence, sequence)
  }


# For encoder-decoder model
def decoder_single_layer_with_kv_cache(
    x: jax.Array,
    decoder_params_and_kv_cache_and_projs: tuple[dict[str, dict[str, jax.Array]], dict[str, jax.Array], jax.Array, jax.Array],
    *,
    cross_padding_mask: jax.Array,
    index: int,
    num_heads: int,
    num_query_key_features: int,
    num_value_features: int,
    num_embedding_features: int,
) -> tuple[jax.Array, tuple[dict[str, jax.Array], dict[str, jax.Array]]]:
  # Single pass of one decoder layer: masked_attention -> layernorm -> cross_attention -> layernorm -> dense -> layernorm
  # x is of shape (batch, 1, d_model)

  # kv_cache and projection arrays are of shape (batch, head, sequence, d_k or d_v)
  decoder_params, kv_cache, cross_k_proj, cross_v_proj = decoder_params_and_kv_cache_and_projs
  max_sequence_length = cross_k_proj.shape[2]

  # Project attention matrices
  q_proj_vector = project_attention(
    x,
    decoder_params['masked_attention']['query'],
    num_heads=num_heads,
    num_query_key_value_features=num_query_key_features,
  )  # (batch, head, 1, d_k)
  k_proj_vector = project_attention(
    x,
    decoder_params['masked_attention']['key'],
    num_heads=num_heads,
    num_query_key_value_features=num_query_key_features,
  )  # (batch, head, 1, d_k)
  v_proj_vector = project_attention(
    x,
    decoder_params['masked_attention']['value'],
    num_heads=num_heads,
    num_query_key_value_features=num_value_features,
  )  # (batch, head, 1, d_v)

  # Load the KV cache
  k_proj = kv_cache['key_cache']  # (batch, head, sequence, d_k)
  v_proj = kv_cache['value_cache']  # (batch, head, sequence, d_v)

  # Update the KV cache with the most recently computed key and value projections for the current input token / time step
  k_proj = k_proj.at[:, :, index, :].set(k_proj_vector[:, :, 0, :])  # match (batch, head, d_k) dimensions
  v_proj = v_proj.at[:, :, index, :].set(v_proj_vector[:, :, 0, :])
  kv_cache['key_cache'] = k_proj
  kv_cache['value_cache'] = v_proj

  # Compute masked attention output
  x, masked_attention_scores = functools.partial(
    compute_attention_output,
    num_query_key_features=num_query_key_features,
    num_embedding_features=num_embedding_features,
    padding_mask=None,
    causal_mask=get_1d_causal_mask(sequence_length=max_sequence_length, index=index),
  )(x, q_proj_vector, k_proj, v_proj, decoder_params['masked_attention'])  # (batch, 1, d_model)
  # k_proj and v_proj are of shapes (batch, head, sequence, d_k) and (batch, head, sequence, d_v), respectively

  # Project cross attention matrices
  q_proj = project_attention(
    x,
    decoder_params['cross_attention']['query'],
    num_heads=num_heads,
    num_query_key_value_features=num_query_key_features,
  )  # (batch, head, 1, d_k)
  # Cross attention projection for key and value are dependent on the encoder output.
  # Since the encoder output is static (i.e. processed once in the beginning),
  # there is no need to re-compute the cross attention projection key and value every time we run the decoder auto-regressively.

  # Compute cross attention output
  x, cross_attention_scores = functools.partial(
    compute_attention_output,
    num_query_key_features=num_query_key_features,
    num_embedding_features=num_embedding_features,
    padding_mask=cross_padding_mask,  # (batch, sequence), derived from input sentence batch to encoder
    causal_mask=None,
  )(x, q_proj, cross_k_proj, cross_v_proj, decoder_params['cross_attention'])  # (batch, 1, d_model)
  # cross_k_proj and cross_v_proj are of shapes (batch, head, sequence, d_k) and (batch, head, sequence, d_v), respectively

  # Compute dense output
  x = compute_dense_output(x, decoder_params['dense'])  # (batch, 1, d_model)

  return (
    x,
    (
      kv_cache,
      {
        'masked_attention_scores': masked_attention_scores,  # (batch, head, sequence, sequence)
        'cross_attention_scores': cross_attention_scores,  # (batch, head, sequence, sequence)
      },
    ),
  )

# For decoder-only model teacher forced pre-fill.
def decoder_only_single_layer(
    x: jax.Array,
    decoder_params: dict[str, dict[str, jax.Array]],
    *,
    num_heads: int,
    num_query_key_features: int,
    num_value_features: int,
    num_embedding_features: int,
) -> tuple[jax.Array, tuple[dict[str, jax.Array], dict[str, jax.Array]]]:
  # Single pass of one decoder layer: masked_attention -> layernorm -> cross_attention -> layernorm -> dense -> layernorm
  # x is of shape (batch, sequence, d_model)

  # Project attention matrices
  q_proj = project_attention(
    x,
    decoder_params['masked_attention']['query'],
    num_heads=num_heads,
    num_query_key_value_features=num_query_key_features,
  )  # (batch, head, sequence, d_k)
  k_proj = project_attention(
    x,
    decoder_params['masked_attention']['key'],
    num_heads=num_heads,
    num_query_key_value_features=num_query_key_features,
  )  # (batch, head, sequence, d_k)
  v_proj = project_attention(
    x,
    decoder_params['masked_attention']['value'],
    num_heads=num_heads,
    num_query_key_value_features=num_value_features,
  )  # (batch, head, sequence, d_v)

  # Compute masked attention output
  x, masked_attention_scores = functools.partial(
    compute_attention_output,
    num_query_key_features=num_query_key_features,
    num_embedding_features=num_embedding_features,
    padding_mask=None,  # causal mask will already mask out future tokens, so padding mask is not needed.
    causal_mask=get_2d_causal_mask(sequence_length=x.shape[1]),
  )(x, q_proj, k_proj, v_proj, decoder_params['masked_attention'])  # (batch, sequence, d_model)

  # Compute dense output
  x = compute_dense_output(x, decoder_params['dense'])  # (batch, sequence, d_model)

  return (
    x,
    (
      {'key_cache': k_proj, 'value_cache': v_proj},  # (batch, head, sequence, d_k or d_v)
      {'masked_attention_scores': masked_attention_scores},  # (batch, head, sequence, sequence)
    ),
  )

# For decoder-only model auto-regressive inference.
def decoder_only_single_layer_with_kv_cache(
    x: jax.Array,
    decoder_params_and_kv_cache: tuple[dict[str, dict[str, jax.Array]], dict[str, jax.Array]],
    *,
    position_index_vector: jax.Array,
    num_heads: int,
    num_query_key_features: int,
    num_value_features: int,
    num_embedding_features: int,
) -> tuple[jax.Array, tuple[dict[str, jax.Array], dict[str, jax.Array]]]:
  # Single pass of one decoder layer: masked_attention -> layernorm -> cross_attention -> layernorm -> dense -> layernorm
  # x is of shape (batch, 1, d_model)
  # position_index_vector is of shape (batch,), indicating the position of each sample in the batch (since it may be uneven since we are right-padding our sequences)

  # kv_cache and projection arrays are of shape (batch, head, sequence, d_k or d_v)
  decoder_params, kv_cache = decoder_params_and_kv_cache
  max_sequence_length = kv_cache['key_cache'].shape[2]
  batch_size = position_index_vector.shape[0]

  # Project attention matrices
  q_proj_vector = project_attention(
    x,
    decoder_params['masked_attention']['query'],
    num_heads=num_heads,
    num_query_key_value_features=num_query_key_features,
  )  # (batch, head, 1, d_k)
  k_proj_vector = project_attention(
    x,
    decoder_params['masked_attention']['key'],
    num_heads=num_heads,
    num_query_key_value_features=num_query_key_features,
  )  # (batch, head, 1, d_k)
  v_proj_vector = project_attention(
    x,
    decoder_params['masked_attention']['value'],
    num_heads=num_heads,
    num_query_key_value_features=num_value_features,
  )  # (batch, head, 1, d_v)

  # Load the KV cache
  k_proj = kv_cache['key_cache']  # (batch, head, sequence, d_k)
  v_proj = kv_cache['value_cache']  # (batch, head, sequence, d_v)

  # Update the KV cache with the most recently computed key and value projections for the current input token / time step, on a per batch basis
  # We index different position indices (specified in position_index_vector) for every batch to write the new kv cache entry to.
  k_proj = k_proj.at[jnp.arange(batch_size), :, position_index_vector, :].set(k_proj_vector[:, :, 0, :])  # match (batch, head, d_k) dimensions
  v_proj = v_proj.at[jnp.arange(batch_size), :, position_index_vector, :].set(v_proj_vector[:, :, 0, :])

  kv_cache['key_cache'] = k_proj
  kv_cache['value_cache'] = v_proj

  # Compute masked attention output
  x, masked_attention_scores = functools.partial(
    compute_attention_output,
    num_query_key_features=num_query_key_features,
    num_embedding_features=num_embedding_features,
    padding_mask=None,
    causal_mask=None,
    decoder_only_causal_mask=get_decoder_only_1d_causal_mask(sequence_length=max_sequence_length, position_index_vector=position_index_vector),
  )(x, q_proj_vector, k_proj, v_proj, decoder_params['masked_attention'])  # (batch, 1, d_model)
  # k_proj and v_proj are of shapes (batch, head, sequence, d_k) and (batch, head, sequence, d_v), respectively

  # Compute dense output
  x = compute_dense_output(x, decoder_params['dense'])  # (batch, 1, d_model)

  return (
    x,
    (
      kv_cache,
      {'masked_attention_scores': masked_attention_scores},  # (batch, head, sequence, sequence)
    ),
  )


def encoder_forward(
    x: jax.Array,  # array of token indices
    params: dict,
    *,
    padding_mask: jax.Array,
    num_heads: int,
    num_query_key_features: int,
    num_value_features: int,
    num_embedding_features: int,
) -> tuple[jax.Array, jax.Array]:
  # Forward pass of all layers in the encoder stack

  # x is of shape (batch, sequence)
  sequence_length = x.shape[1]

  # Embedding and position encoding
  x = params['embedding'][x] * jnp.sqrt(num_embedding_features) # (batch, sequence, d_model)
  x += position_encode(sequence_length=sequence_length, num_embedding_features=num_embedding_features)[None, :, :]

  # Apply a loop of encoder layer forward passes
  x, attention_scores = jax.lax.scan(
    functools.partial(
      encoder_single_layer,
      padding_mask=padding_mask,
      num_heads=num_heads,
      num_query_key_features=num_query_key_features,
      num_value_features=num_value_features,
      num_embedding_features=num_embedding_features,
    ),
    x,
    params['encoder'],
  )  # (batch, sequence, d_model)

  return x, attention_scores


def decoder_forward(
    x: jax.Array,  # array of token indices
    params: dict,
    *,
    encoder_output: jax.Array,
    cross_padding_mask: jax.Array,
    num_heads: int,
    num_query_key_features: int,
    num_value_features: int,
    num_embedding_features: int,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  # Forward pass of all layers in the decoder stack

  # x is of shape (batch, sequence)
  sequence_length = x.shape[1]

  # Embedding and position encoding
  embedding_matrix: jax.Array = params['embedding']  # (vocab, d_model)
  x = embedding_matrix[x] * jnp.sqrt(num_embedding_features) # (batch, sequence, d_model)
  x += position_encode(sequence_length=sequence_length, num_embedding_features=num_embedding_features)[None, :, :]

  # Apply a loop of encoder layer forward passes
  x, attention_scores_dict = jax.lax.scan(
    functools.partial(
      decoder_single_layer,
      encoder_output=encoder_output,
      cross_padding_mask=cross_padding_mask,
      num_heads=num_heads,
      num_query_key_features=num_query_key_features,
      num_value_features=num_value_features,
      num_embedding_features=num_embedding_features,
    ),
    x,
    params['decoder'],
  )  # (batch, sequence, d_model)

  # Project embeddings back to vocab
  x = jnp.einsum('bsd,dv->bsv', x, embedding_matrix.T) + params['final_bias'][None, None, :]  # (batch, sequence, vocab)

  # return raw logits, do softmax/hardmax in an outer function
  return x, attention_scores_dict


# For encoder-decoder model.
def decoder_forward_with_kv_cache(
    x: jax.Array,  # array of token indices
    params_and_kv_cache: tuple[dict, dict[str, jax.Array]],
    *,
    cross_k_proj: jax.Array,
    cross_v_proj: jax.Array,
    cross_padding_mask: jax.Array,
    index: int,
    num_heads: int,
    num_query_key_features: int,
    num_value_features: int,
    num_embedding_features: int,
) -> tuple[jax.Array, tuple[dict[str, jax.Array], dict[str, jax.Array]]]:
  # Forward pass of all layers in the decoder stack
  # x is of shape (batch, 1)

  params, kv_cache = params_and_kv_cache  # kv_cache arrays are of shape (layer, batch, head, sequence, d_k or d_v)

  # Embedding and position encoding
  embedding_matrix: jax.Array = params['embedding']  # (vocab, d_model)
  x = embedding_matrix[x] * jnp.sqrt(num_embedding_features) # (batch, 1, d_model)
  x += position_encode_single(index=index, num_embedding_features=num_embedding_features)

  # Apply a loop of encoder layer forward passes
  x, (new_kv_cache, attention_scores_dict) = jax.lax.scan(
    functools.partial(
      decoder_single_layer_with_kv_cache,
      cross_padding_mask=cross_padding_mask,
      index=index,
      num_heads=num_heads,
      num_query_key_features=num_query_key_features,
      num_value_features=num_value_features,
      num_embedding_features=num_embedding_features,
    ),
    x,
    (params['decoder'], kv_cache, cross_k_proj, cross_v_proj),
  )  # (batch, 1, d_model)

  # Project embeddings back to vocab
  x = jnp.einsum('bsd,dv->bsv', x, embedding_matrix.T) + params['final_bias'][None, None, :]  # (batch, 1, vocab)

  # return raw logits, do softmax/hardmax in an outer function
  return x, (new_kv_cache, attention_scores_dict)


# For decoder-only model teacher forced pre-fill.
def decoder_only_forward(
    x: jax.Array,  # array of token indices
    params: dict,
    *,
    num_heads: int,
    num_query_key_features: int,
    num_value_features: int,
    num_embedding_features: int,
) -> tuple[jax.Array, tuple[dict[str, jax.Array], dict[str, jax.Array]]]:
  # Forward pass of all layers in the decoder stack

  # x is of shape (batch, sequence)
  sequence_length = x.shape[1]

  # Embedding and position encoding
  embedding_matrix: jax.Array = params['embedding']  # (vocab, d_model)
  x = embedding_matrix[x] * jnp.sqrt(num_embedding_features) # (batch, sequence, d_model)
  x += position_encode(sequence_length=sequence_length, num_embedding_features=num_embedding_features)[None, :, :]

  # Apply a loop of encoder layer forward passes
  x, (kv_cache, attention_scores_dict) = jax.lax.scan(
    functools.partial(
      decoder_only_single_layer,
      num_heads=num_heads,
      num_query_key_features=num_query_key_features,
      num_value_features=num_value_features,
      num_embedding_features=num_embedding_features,
    ),
    x,
    params['decoder'],
  )
  # x is of shape (batch, sequence, d_model)
  # kv_cache entries are of shape (layer, batch, head, sequence, d_k or d_v)

  # Project embeddings back to vocab
  x = jnp.einsum('bsd,dv->bsv', x, embedding_matrix.T) + params['final_bias'][None, None, :]  # (batch, sequence, vocab)

  # return raw logits, do softmax/hardmax in an outer function
  return x, (kv_cache, attention_scores_dict)

# For decoder-only model auto-regressive inference.
def decoder_only_forward_with_kv_cache(
    x: jax.Array,  # array of token indices
    params_and_kv_cache: tuple[dict, dict[str, jax.Array]],
    *,
    position_index_vector: jax.Array,
    num_heads: int,
    num_query_key_features: int,
    num_value_features: int,
    num_embedding_features: int,
) -> tuple[jax.Array, tuple[dict[str, jax.Array], jax.Array, dict[str, jax.Array]]]:
  # Forward pass of all layers in the decoder stack
  # x is of shape (batch, 1)
  # position_index_vector is of shape (batch,), indicating the position of each sample in the batch (since it may be uneven since we are right-padding our sequences)

  params, kv_cache = params_and_kv_cache  # kv_cache arrays are of shape (layer, batch, head, sequence, d_k or d_v)
  max_sequence_length = kv_cache['key_cache'].shape[3]

  # Embedding and position encoding
  embedding_matrix: jax.Array = params['embedding']  # (vocab, d_model)
  x = embedding_matrix[x] * jnp.sqrt(num_embedding_features) # (batch, 1, d_model)
  # Calculate the positional encoding for each position index in a per batch basis
  x += jax.vmap(position_encode_single, in_axes=(0, None))(position_index_vector, num_embedding_features)  # type: ignore

  # Apply a loop of encoder layer forward passes
  x, (new_kv_cache, attention_scores_dict) = jax.lax.scan(
    functools.partial(
      decoder_only_single_layer_with_kv_cache,
      position_index_vector=position_index_vector,
      num_heads=num_heads,
      num_query_key_features=num_query_key_features,
      num_value_features=num_value_features,
      num_embedding_features=num_embedding_features,
    ),
    x,
    (params['decoder'], kv_cache),
  )  # (batch, 1, d_model)

  # Project embeddings back to vocab
  x = jnp.einsum('bsd,dv->bsv', x, embedding_matrix.T) + params['final_bias'][None, None, :]  # (batch, 1, vocab)

  # Increment the position_index_vector, cap out the position indices to the max index of the sequence.
  # Capping out happens with uneven sequence lengths within the same batch where some sequences finish decoding faster than others.
  # In order to avoid index errors, we simply cap it to the max index of the sequence, which would cause overwriting in the last kv cache entry,
  # but that's fine since we're done decoding and everything else that gets decoded onward for that sequence is garbage.
  new_position_index_vector = jnp.minimum(
    position_index_vector+1,
    jnp.ones(position_index_vector.shape[0], dtype=int) * (max_sequence_length-1),
  )  # (batch,)

  # return raw logits, do softmax/hardmax in an outer function
  return x, (new_kv_cache, new_position_index_vector, attention_scores_dict)


# To go from encoder-decoder model to decoder-only model:
# - remove the encoder layer
# - remove the cross attention layer; all decoder layers will just be masked self-attention (with causal mask) and then a dense layer
# - i don't have a separation between input and target sequences anymore (e.g. "1+1" and "2"), but rather they are combined into a single sequence (e.g. "1+1=2") and I should teacher force train my model to always just predict the next token from this combined sequence, whether it's predicting the prompt or predicting the actual "response"/"answer" to the prompt
# - I also need to implement a loss mask so that only the actual "response"/"answer" is used to calculate gradients (e.g. mask out "1+1=" outputted tokens and only use the "2" token when calculating the loss)
#   - also need to implement a loss mask for the pad tokens (identical to the encoder-decoder model)
# - right pad the input tokens (same as encoder-decoder model)
# - kv cache:
#   - instantiate a kv cache of zeroes with shape (num_layers, batch, head, max_sequence_length, d_k or d_v)
#   - during inference:
#     - first, we pre-fill this kv cache:
#       - apply sinusoidal positional encoding
#       - each token at index i in the input sequence will have its projection calculated and written in the ith index of the kv cache
#         - apply a causal mask (see the example below on how to implement it) on the raw attention score before computing softmax
#       - with non-uniform sequence lengths, you will have some parts of the kv cache with more zeroes than others
#       - keep a pointer (index counter) for each sequence in the kv cache that points to the earliest (most-left) zero entry
#         - we will write the next projection entry to this pointer and then increment it
#           - can do something like this to selectively update entries in a matrix: `matrix.at[jnp.arange(5),[4,2,3,1,2]].set(-1)`
#         - output this pointer vector for auto-regressive decoding
#       - discard all outputs in this stage, since we are just pre-filling to compute the kv cache of the prompt
#     - next, we do auto-regressive decoding:
#       - as soon as the separator token is processed from the prompt (i.e. indicates the end of the "prompt prediction" and the start of the "response" to the model), take the decoded output as the first token in the "response" of the model
#       - also feed in the pointer vector which denotes both the position index of the token (for sinusoidal positional encoding) and also the index to write the new projections to in the kv cache for each corresponding sequence
#       - for the given input token batch of shape (batch, 1), calculate the q_proj, k_proj and v_proj of shape (batch, head, 1, d_k or d_v)
#       - write the k_proj and v_proj in the kv cache in the corresponding pointer for each sequence in the batch, and then increment the pointer
#       - get the causal mask of the kv cache of shape (batch, sequence) which masks out every column i if i is greater than the current position indicated by the pointer for each corresponding sequence
#         - can do something like this:
# ```
# >>> a
# Array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14],
#        [15, 16, 17, 18, 19],
#        [20, 21, 22, 23, 24]], dtype=int32)
# >>> jnp.where(a>=jnp.array([1,7,13,17,21])[:, None], 1, 0)
# Array([[0, 1, 1, 1, 1],
#        [0, 0, 1, 1, 1],
#        [0, 0, 0, 1, 1],
#        [0, 0, 1, 1, 1],
#        [0, 1, 1, 1, 1]], dtype=int32, weak_type=True)
# ```
#       - calculate the attention output:
#         - calculate the raw attention score logits using q_proj and the k_proj from the kv cache
#         - apply the causal mask
#         - softmax to get the attention scores
#         - mat-mul with the v_proj from the kv cache to get the attention output of shape (batch, head, 1, d_v)
#     - the rest of the inference should be the same as the encoder-decoder model (minus cross attention)
#     - output the incremented position vector (to be fed in again via jax.lax.scan)
#     - once an EOS token is emitted or the sequence length (input tokens + output tokens) is at the max sequence length, we just ignore the rest of the outputs for a particular sequence when decoding
#       - this could happen if the one sequence was length (max_sequence_length-1) while another sequence is length 1,
#       - in which case we would still decode for potentially (max_sequence_length-1) more time steps for the shorter sequence,
#       - but we would only take the first token decoded from the longer sequence, and ignore the rest
# - the input sequence now includes the "original user prompt" plus the decoded tokens instead of only the decoded tokens
# - teacher forcing should be identical to encoder-decoder model except:
#   - we only use the decoder model
#   - we still use the same causal mask
#   - instead of just forcing the target output sequence, we force the concatenation of the input prompt and the target output sequence
#     - essentially the teacher forced part of the input prompt is the "pre-fill",
#     - but we can continue to teacher force the target sequence as well since we know the "ground-truth" of the target sequence during training



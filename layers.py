'''
Attention is all you need paper: https://arxiv.org/pdf/1706.03762

Comments about dimension/shape:
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


def get_causal_mask(sequence_length: int) -> jax.Array:
  # Upper triangle mask (NOTE: the diagonal is not masked).
  mask = jnp.ones((sequence_length, sequence_length))
  return jnp.triu(mask, k=1) * (-1e9)  # (sequence, sequence)


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
    padding_mask: jax.Array,
    apply_causal_mask: bool,
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
  attention_scores += padding_mask[:, None, None, :]  # add dummy dimensions to padding mask (batch, 1, 1, sequence_k) and apply it
  if apply_causal_mask:
    # Apply causal mask.
    # Note this function returns a square causal mask to be applied to the attention score matrix, so we assume the attention score matrix is square.
    # The only time the attention score matrix is not a square is during cross attention, but we do not apply a causal mask for cross attention anyways.
    attention_scores += get_causal_mask(sequence_length=sequence_length)[None, None, :, :]  # (1, 1, sequence, sequence)
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
  x, attention_scores = compute_attention_output(
    x,
    q_proj,
    k_proj,
    v_proj,
    encoder_params['attention'],
    num_query_key_features=num_query_key_features,
    num_embedding_features=num_embedding_features,
    padding_mask=padding_mask,
    apply_causal_mask=False,
  )  # (batch, sequence, d_model)

  # Compute dense output
  x = compute_dense_output(x, encoder_params['dense'])  # (batch, sequence, d_model)

  # attention scores is of shape (batch, head, sequence, sequence)
  return x, attention_scores


def decoder_single_layer(
    x: jax.Array,
    decoder_params: dict[str, dict[str, jax.Array]],
    *,
    encoder_output: jax.Array,
    masked_padding_mask: jax.Array,
    cross_padding_mask: jax.Array,
    num_heads: int,
    num_query_key_features: int,
    num_value_features: int,
    num_embedding_features: int,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  # Single pass of one decoder layer: masked_attention -> layernorm -> cross_attention -> layernorm -> dense -> layernorm

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
  x, masked_attention_scores = compute_attention_output(
    x,
    q_proj,
    k_proj,
    v_proj,
    decoder_params['masked_attention'],
    num_query_key_features=num_query_key_features,
    num_embedding_features=num_embedding_features,
    padding_mask=masked_padding_mask,  # (batch, sequence-1), derived from input sentence batch to decoder
    apply_causal_mask=True,
  )  # (batch, sequence, d_model)

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
  x, cross_attention_scores = compute_attention_output(
    x,
    q_proj,
    k_proj,
    v_proj,
    decoder_params['cross_attention'],
    num_query_key_features=num_query_key_features,
    num_embedding_features=num_embedding_features,
    padding_mask=cross_padding_mask,  # (batch, sequence), derived from input sentence batch to encoder
    apply_causal_mask=False,
  )  # (batch, sequence, d_model)

  # Compute dense output
  x = compute_dense_output(x, decoder_params['dense'])  # (batch, sequence, d_model)

  return x, {
    'masked_attention_scores': masked_attention_scores,  # (batch, head, sequence, sequence)
    'cross_attention_scores': cross_attention_scores,  # (batch, head, sequence, sequence)
  }


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
  x += position_encode(sequence_length=sequence_length, num_embedding_features=num_embedding_features)

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
    masked_padding_mask: jax.Array,
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
  x += position_encode(sequence_length=sequence_length, num_embedding_features=num_embedding_features)

  # Apply a loop of encoder layer forward passes
  x, attention_scores_dict = jax.lax.scan(
    functools.partial(
      decoder_single_layer,
      encoder_output=encoder_output,
      masked_padding_mask=masked_padding_mask,
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

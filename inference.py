from typing import Callable

import functools

import jax
import jax.numpy as jnp

from config import Config
from layers import project_attention, encoder_forward, decoder_forward, decoder_forward_with_kv_cache


def make_teacher_force_forward_fn(config: Config) -> Callable[[dict[str, jax.Array], jax.Array, jax.Array], tuple[jax.Array, dict[str, jax.Array]]]:

  @jax.jit
  def teacher_force_forward(params: dict[str, jax.Array], input_sequence_batch: jax.Array, target_sequence_batch: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    # both input_sequence_batch and target_sequence_batch are of shape (batch, sequence)

    # additive mask for encoder attention and decoder cross attention layers
    encoder_input_padding_mask = jnp.where(input_sequence_batch==config.pad_index, -1e9, 0)  # (batch, sequence)

    encoder_output, encoder_attention_scores = encoder_forward(
      input_sequence_batch,  # an array of token indices of shape (batch, sequence)
      params,
      padding_mask=encoder_input_padding_mask,
      num_heads=config.num_heads,
      num_query_key_features=config.num_query_key_features,
      num_value_features=config.num_value_features,
      num_embedding_features=config.num_embedding_features,
    )  # (batch, sequence, d_model)

    # Prepare teacher-forcing input/target batch pairs, each of shape (batch, sequence-1)
    decoder_input_sequence_batch = target_sequence_batch[:, :-1]  # We don't predict anything if we feed in the last token of the sequence (we don't have a corresponding target for this).

    decoder_output, decoder_attention_scores_dict = decoder_forward(
      decoder_input_sequence_batch,  # an array of token indices of shape (batch, sequence-1)
      params,
      encoder_output=encoder_output,  # (batch, sequence, d_model)
      cross_padding_mask=encoder_input_padding_mask,  # (batch, sequence)
      num_heads=config.num_heads,
      num_query_key_features=config.num_query_key_features,
      num_value_features=config.num_value_features,
      num_embedding_features=config.num_embedding_features,
    )  # (batch, sequence-1, vocab)

    attention_scores_dict = {
      'encoder_attention_scores': encoder_attention_scores,
      **decoder_attention_scores_dict
    }

    return decoder_output, attention_scores_dict

  return teacher_force_forward


def make_autoregressive_encode_decode_fn(config: Config) -> Callable[[dict[str, jax.Array], jax.Array], tuple[jax.Array, dict[str, jax.Array]]]:

  @jax.jit
  def autoregressive_encode_decode(params: dict[str, jax.Array], encoder_input_sequence_batch: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    # both input_sequence_batch and target_sequence_batch are of shape (batch, sequence)
    batch_size = encoder_input_sequence_batch.shape[0]
    max_sequence_length = encoder_input_sequence_batch.shape[1]

    # additive mask for encoder attention and decoder cross attention layers
    encoder_input_padding_mask = jnp.where(encoder_input_sequence_batch==config.pad_index, -1e9, 0)  # (batch, sequence)

    encoder_output, encoder_attention_scores = encoder_forward(
      encoder_input_sequence_batch,  # an array of token indices of shape (batch, sequence)
      params,
      padding_mask=encoder_input_padding_mask,
      num_heads=config.num_heads,
      num_query_key_features=config.num_query_key_features,
      num_value_features=config.num_value_features,
      num_embedding_features=config.num_embedding_features,
    )  # (batch, sequence, d_model)

    decoder_input_sequence_batch = jnp.ones((batch_size, max_sequence_length), dtype=int) * config.pad_index  # (batch, sequence)
    decoder_input_sequence_batch = decoder_input_sequence_batch.at[:, 0].set(config.sos_index)  # populate 0th column with SOS tokens.

    def autoregressive_decode(decoder_input_sequence_batch: jax.Array, index: int) -> tuple[jax.Array, dict[str, jax.Array]]:
      decoder_output_sequence_batch = decoder_input_sequence_batch.copy()  # (batch, sequence)

      # truncate since don't predict anything if we feed in the last token of the sequence.
      decoder_input_sequence_batch = decoder_input_sequence_batch[:, :-1]

      decoder_output, decoder_attention_scores_dict = decoder_forward(
        decoder_input_sequence_batch,  # an array of token indices of shape (batch, sequence-1)
        params,
        encoder_output=encoder_output,  # (batch, sequence, d_model)
        cross_padding_mask=encoder_input_padding_mask,  # (batch, sequence)
        num_heads=config.num_heads,
        num_query_key_features=config.num_query_key_features,
        num_value_features=config.num_value_features,
        num_embedding_features=config.num_embedding_features,
      )  # (batch, sequence-1, vocab)

      decoder_output = decoder_output.argmax(-1)  # (batch, sequence-1)

      # The 0th column of the input are SOS tokens, whereas the 0th column of the output are the predicted tokens based on the input SOS token only.
      # We append the predicted tokens to the next column of the input and return the updated input array, for the next iteration of the scan loop.
      decoder_output_sequence_batch = decoder_output_sequence_batch.at[:, index+1].set(decoder_output[:, index])

      return decoder_output_sequence_batch, decoder_attention_scores_dict

    decoder_output_sequence_batch, decoder_attention_scores_dict = jax.lax.scan(
      autoregressive_decode,
      decoder_input_sequence_batch,
      # we don't predict anything if we feed in the last token of the sequence
      jnp.arange(max_sequence_length-1),  # type: ignore
    )  # decoder_output_sequence_batch is of shape (batch, sequence-1)

    attention_scores_dict = {
      'encoder_attention_scores': encoder_attention_scores,
      **decoder_attention_scores_dict,
    }

    return decoder_output_sequence_batch, attention_scores_dict

  return autoregressive_encode_decode


def make_autoregressive_encode_decode_with_kv_cache_fn(config: Config) -> Callable[[dict[str, jax.Array], jax.Array], tuple[jax.Array, dict[str, jax.Array]]]:

  @jax.jit
  def autoregressive_encode_decode_with_kv_cache(params: dict[str, jax.Array], encoder_input_sequence_batch: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    # both input_sequence_batch and target_sequence_batch are of shape (batch, sequence)
    batch_size = encoder_input_sequence_batch.shape[0]
    max_sequence_length = encoder_input_sequence_batch.shape[1]

    # additive mask for encoder attention and decoder cross attention layers
    encoder_input_padding_mask = jnp.where(encoder_input_sequence_batch==config.pad_index, -1e9, 0)  # (batch, sequence)

    encoder_output, encoder_attention_scores = encoder_forward(
      encoder_input_sequence_batch,  # an array of token indices of shape (batch, sequence)
      params,
      padding_mask=encoder_input_padding_mask,
      num_heads=config.num_heads,
      num_query_key_features=config.num_query_key_features,
      num_value_features=config.num_value_features,
      num_embedding_features=config.num_embedding_features,
    )  # (batch, sequence, d_model)

    # Pre-compute cross attention projections, since it is solely dependent on the encoder output (which is static)
    project_attention_all_layers = jax.vmap(
      # Bake in kwargs so the function only has two positional arguments: input array and attention matrix
      functools.partial(
        project_attention,
        num_heads=config.num_heads,
        num_query_key_value_features=config.num_query_key_features
      ),
      # broadcast the input array (i.e. the encoder input),
      # vectorize over the 0th dimension of the attention matrix of shape (layer, d_model, head*(d_k or d_v))
      in_axes=(None, 0),
    )
    cross_k_proj = project_attention_all_layers(
      encoder_output,
      params['decoder']['cross_attention']['key']
    )  # (layer, batch, head, sequence, d_k)
    cross_v_proj = project_attention_all_layers(
      encoder_output,
      params['decoder']['cross_attention']['value']
    )  # (layer, batch, head, sequence, d_v)

    decoder_input_token_batch = jnp.ones((batch_size, 1), dtype=int) * config.sos_index  # (batch, 1)

    # Instantiate a blank KV cache
    # Shape (layer, batch, head, sequence, d_k or d_v)
    kv_cache = {
      'key_cache': jnp.zeros((
        config.num_decoder_layers,
        batch_size,
        config.num_heads,
        max_sequence_length,
        config.num_query_key_features
      )),
      'value_cache': jnp.zeros((
        config.num_decoder_layers,
        batch_size,
        config.num_heads,
        max_sequence_length,
        config.num_value_features
      )),
    }

    def autoregressive_decode_with_kv_cache(decoder_input_token_batch_and_kv_cache: tuple[jax.Array, dict[str, jax.Array]], index: int) -> tuple[tuple[jax.Array, dict[str, jax.Array]], tuple[jax.Array, dict[str, jax.Array]]]:
      # decoder_input_token_batch is of shape (batch, 1)
      # kv_cache has attention weights of shape (batch, head, sequence, d_k or d_v)
      decoder_input_token_batch, kv_cache = decoder_input_token_batch_and_kv_cache

      decoder_output, (new_kv_cache, decoder_attention_scores_dict) = decoder_forward_with_kv_cache(
        decoder_input_token_batch,  # an array of token indices of shape (batch, 1)
        (params, kv_cache),
        cross_k_proj=cross_k_proj,  # (layer, batch, head, sequence, d_k)
        cross_v_proj=cross_v_proj,  # (layer, batch, head, sequence, d_v)
        # masked attention layers do not use padding mask
        cross_padding_mask=encoder_input_padding_mask,  # (batch, sequence)
        index=index,
        num_heads=config.num_heads,
        num_query_key_features=config.num_query_key_features,
        num_value_features=config.num_value_features,
        num_embedding_features=config.num_embedding_features,
      )  # (batch, 1, vocab)

      decoder_output = decoder_output.argmax(-1)  # (batch, 1)

      # We need to output the decoder_output in the scan output as well since the carry output will only output the last token,
      # whereas the scan output will have the whole history of output tokens.
      return (decoder_output, new_kv_cache), (decoder_output, decoder_attention_scores_dict)

    (decoder_output_token_batch, kv_cache), (decoder_output_sequence_batch, decoder_attention_scores_dict) = jax.lax.scan(
      autoregressive_decode_with_kv_cache,
      (decoder_input_token_batch, kv_cache),
      # we don't predict anything if we feed in the last token of the sequence
      jnp.arange(max_sequence_length-1),  # type: ignore
    )  # decoder_output_sequence_batch is of shape (sequence-1, batch, 1)
    decoder_output_sequence_batch = jnp.einsum('sbt->bst', decoder_output_sequence_batch)  # (batch, sequence-1, 1)
    decoder_output_sequence_batch = decoder_output_sequence_batch[:, :, 0]  # (batch, sequence-1)

    attention_scores_dict = {
      'encoder_attention_scores': encoder_attention_scores,
      **decoder_attention_scores_dict,
    }

    return decoder_output_sequence_batch, attention_scores_dict

  return autoregressive_encode_decode_with_kv_cache


# KV caching summary:
# - create a k_proj and v_proj buffer of (batch, head, sequence, d_k) and (batch, head, sequence, d_v), respectively, initialized with zeroes
# - at time step i
#   - input just the CURRENT token (batch, 1) to decoder
#   - compute the q_proj, k_proj and v_proj VECTORS of shapes (batch, head, 1, d_k), (batch, head, 1, d_k)  and (batch, head, 1, d_v), respectively, from the input x
#   - append the new k_proj and v_proj to their respective buffers at the index slice [:, :, i, :]
#   - then compute attention score and output and end up with shape: softmax((batch, head, 1, d_k) x (batch, head, d_k, sequence)) @ (batch, head, sequence, d_v) -> (batch, 1, d_model)
#   - then output (batch, 1, d_model) as the next predicted token
# NOTE: instead of upper triangular causal mask, we use a 1D causal mask that simply just masks all future tokens i+1, i+2, ...

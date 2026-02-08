from typing import Callable

import os
import time
import pickle

import jax
import jax.numpy as jnp

import optax

from config import Config, get_config
from data import load_data, compute_dataset_entropy
from parameters import initialize_parameters
from inference import make_teacher_force_forward_fn, make_teacher_force_decoder_only_forward_fn


def make_compute_encoder_decoder_loss_fn(config: Config) -> Callable[[dict[str, jax.Array], jax.Array, jax.Array], jax.Array]:
  teacher_force_forward = make_teacher_force_forward_fn(config)

  def compute_encoder_decoder_loss(params: dict[str, jax.Array], input_sequence_batch: jax.Array, target_sequence_batch: jax.Array) -> jax.Array:
    decoder_output, _ = teacher_force_forward(params, input_sequence_batch, target_sequence_batch)

    decoder_target_sequence_batch = target_sequence_batch[:, 1:]  # We don't predict the first token (i.e. SOS token), it's simply fed in as the first time step of decoding.
    loss_padding_mask = jnp.where(decoder_target_sequence_batch==config.pad_index, 0, 1)  # (batch, sequence-1)

    # softmax the decoder_output and calculate the cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(
      logits=decoder_output,
      labels=decoder_target_sequence_batch,
    )  # (batch, sequence-1)

    loss *= loss_padding_mask  # (batch, sequence-1)
    loss = jnp.sum(loss) / jnp.sum(loss_padding_mask)  # take the mean out of all loss components derived from non-pad tokens

    return loss

  return compute_encoder_decoder_loss

def make_compute_decoder_only_loss_fn(config: Config) -> Callable[[dict[str, jax.Array], jax.Array], jax.Array]:
  teacher_force_forward = make_teacher_force_decoder_only_forward_fn(config)

  def compute_decoder_only_loss(params: dict[str, jax.Array], input_sequence_batch: jax.Array) -> jax.Array:
    # input_sequence_batch is of shape (batch, sequence)
    # Omit predicting the last token of the sequence.
    decoder_output, _ = teacher_force_forward(params, input_sequence_batch[:, :-1])  # (batch, sequence-1)

    target_sequence_batch = input_sequence_batch[:, 1:]  # We don't predict the first token (i.e. SOS token), it's simply fed in as the first time step of decoding.
    loss_padding_mask = jnp.where(target_sequence_batch==config.pad_index, 0, 1)  # (batch, sequence-1)

    # Mask out the loss components derived from the outputted prompt tokens up until (and including) the outputted separator token.
    separator_indices = jnp.argmax(target_sequence_batch==config.sep_index, axis=1)  # (batch,)
    prompt_mask = jnp.ones(target_sequence_batch.shape) * jnp.arange(target_sequence_batch.shape[1])[None, :]  # (batch, sequence-1)
    prompt_mask = prompt_mask <= separator_indices[:, None]  # we mask out the outputted separator token because that's derived from an input prompt token
    prompt_mask = jnp.where(prompt_mask==True, 0, 1)  # (batch, sequence-1)

    # softmax the decoder_output and calculate the cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(
      logits=decoder_output,
      labels=target_sequence_batch,
    )  # (batch, sequence-1)

    loss *= loss_padding_mask  # (batch, sequence-1)
    loss *= prompt_mask  # (batch, sequence-1)
    # take the mean out of all loss components that weren't masked out
    loss = jnp.sum(loss) / jnp.sum(loss_padding_mask & prompt_mask)  # bitwise AND to take the intersection of loss components that were not excluded from either mask

    return loss

  return compute_decoder_only_loss


def split_data(input_data: jax.Array, target_data: jax.Array | None, train_split_ratio: float, key: jax.Array) -> tuple[jax.Array, jax.Array | None, jax.Array, jax.Array | None]:
  # data is of shape (batch, sequence)
  n_samples = input_data.shape[0]
  shuffled_indices = jax.random.choice(key, jnp.arange(n_samples), shape=(n_samples,), replace=False)
  split_index = round(n_samples*train_split_ratio)
  return (
    input_data[shuffled_indices[:split_index], :],
    target_data[shuffled_indices[:split_index], :] if (target_data is not None) else None,
    input_data[shuffled_indices[split_index:], :],
    target_data[shuffled_indices[split_index:], :] if (target_data is not None) else None,
  )


def sample_batch(input_data: jax.Array, target_data: jax.Array | None, batch_size: int, key: jax.Array) -> tuple[jax.Array, jax.Array | None, jax.Array]:
  key_to_use, key = jax.random.split(key)
  sampled_indices = jax.random.choice(key_to_use, jnp.arange(input_data.shape[0]), shape=(batch_size,), replace=False)
  return (
    input_data[sampled_indices, :],
    target_data[sampled_indices, :] if (target_data is not None) else None,
    key,
  )


def train_encoder_decoder(config: Config):
  params_key, data_key, key = jax.random.split(jax.random.key(config.seed), 3)

  # Load data
  input_data, target_data = load_data(config.task)  # (batch, sequence)
  assert target_data is not None
  (
    train_input_data,
    train_target_data,
    validation_input_data,
    validation_target_data,
  ) = split_data(input_data, target_data, config.train_split_ratio, data_key)
  assert train_target_data is not None
  assert validation_target_data is not None

  # Exclude SOS token since the cross entropy loss in the train loop is calculated by similarly truncating the first token of every target sequence
  train_max_entropy, train_marginal_entropy = compute_dataset_entropy(train_target_data[:, 1:], config.pad_index)
  validation_max_entropy, validation_marginal_entropy = compute_dataset_entropy(validation_target_data[:, 1:], config.pad_index)
  print(f'[Train dataset]\tMax entropy: {train_max_entropy}\tMarginal entropy: {train_marginal_entropy}')
  print(f'[Validation dataset]\tMax entropy: {validation_max_entropy}\tMarginal entropy: {validation_marginal_entropy}')

  # Load parameters
  params = initialize_parameters(config, params_key)

  # Make loss function
  compute_loss = make_compute_encoder_decoder_loss_fn(config)

  # Load optimizer
  optimizer = optax.adam(config.lr_schedule)
  opt_state = optimizer.init(params)

  # Define update function
  @jax.jit
  def update(params, opt_state: optax.OptState, input_data: jax.Array, target_data: jax.Array):
    loss, grads = jax.value_and_grad(compute_loss)(params, input_data, target_data)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

  # Training loop
  start_time = time.time()
  max_num_train_epochs = config.max_num_train_epochs if (config.max_num_train_epochs is not None) else float('inf')

  epoch = 0
  patience_counter = 0
  lowest_validation_loss = float('inf')
  lowest_validation_loss_epoch = None
  best_params = None
  prev_iteration_start_time = time.time()
  while (lowest_validation_loss > config.validation_loss_cutoff) and (epoch < max_num_train_epochs) and (patience_counter <= config.max_patience):
    train_input_batch, train_target_batch, key = sample_batch(train_input_data, train_target_data, config.batch_size, key)
    assert train_target_batch is not None
    params, opt_state, train_loss = update(params, opt_state, train_input_batch, train_target_batch)

    if epoch % config.eval_every_n_epochs == 0:
      validation_loss = compute_loss(params, validation_input_data, validation_target_data)
      if validation_loss >= lowest_validation_loss:
        patience_counter += 1
      else:
        lowest_validation_loss = validation_loss
        lowest_validation_loss_epoch = epoch
        best_params = params
        patience_counter = 0
      print(f'Epoch: {epoch}\tTrain loss: {train_loss}\tValidation loss {validation_loss}\tDuration: {round((time.time()-prev_iteration_start_time)/60, 2)} minutes')
      prev_iteration_start_time = time.time()
    epoch += 1

  checkpoint_path = f'checkpoints/{config.task}'
  if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)
  with open(f'{checkpoint_path}/{str(time.time()).replace(".", "_")}.pkl', 'wb') as file:
    pickle.dump(best_params, file)

  print(f'\nLowest validation loss {lowest_validation_loss} reached at epoch {lowest_validation_loss_epoch}.')
  print(f'Training converged in {(time.time()-start_time)/60} minutes.')


def train_decoder_only(config: Config):
  params_key, data_key, key = jax.random.split(jax.random.key(config.seed), 3)

  # Load data
  input_data, _ = load_data(config.task)  # (batch, sequence)
  (
    train_input_data,
    _,
    validation_input_data,
    _,
  ) = split_data(input_data, None, config.train_split_ratio, data_key)

  # Exclude SOS token since the cross entropy loss in the train loop is calculated by similarly truncating the first token of every target sequence
  train_max_entropy, train_marginal_entropy = compute_dataset_entropy(train_input_data[:, 1:], config.pad_index)
  validation_max_entropy, validation_marginal_entropy = compute_dataset_entropy(validation_input_data[:, 1:], config.pad_index)
  print(f'[Train dataset]\tMax entropy: {train_max_entropy}\tMarginal entropy: {train_marginal_entropy}')
  print(f'[Validation dataset]\tMax entropy: {validation_max_entropy}\tMarginal entropy: {validation_marginal_entropy}')

  # Load parameters
  params = initialize_parameters(config, params_key)  # this would initialize encoder and cross attention params, but should be fine since they're not used

  # Make loss function
  compute_loss = make_compute_decoder_only_loss_fn(config)

  # Load optimizer
  optimizer = optax.adam(config.lr_schedule)
  opt_state = optimizer.init(params)

  # Define update function
  @jax.jit
  def update(params, opt_state: optax.OptState, input_data: jax.Array):
    loss, grads = jax.value_and_grad(compute_loss)(params, input_data)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

  # Training loop
  start_time = time.time()
  max_num_train_epochs = config.max_num_train_epochs if (config.max_num_train_epochs is not None) else float('inf')

  epoch = 0
  patience_counter = 0
  lowest_validation_loss = float('inf')
  lowest_validation_loss_epoch = None
  best_params = None
  prev_iteration_start_time = time.time()
  while (lowest_validation_loss > config.validation_loss_cutoff) and (epoch < max_num_train_epochs) and (patience_counter <= config.max_patience):
    train_input_batch, _, key = sample_batch(train_input_data, None, config.batch_size, key)
    params, opt_state, train_loss = update(params, opt_state, train_input_batch)

    if epoch % config.eval_every_n_epochs == 0:
      validation_loss = compute_loss(params, validation_input_data)
      if validation_loss >= lowest_validation_loss:
        patience_counter += 1
      else:
        lowest_validation_loss = validation_loss
        lowest_validation_loss_epoch = epoch
        best_params = params
        patience_counter = 0
      print(f'Epoch: {epoch}\tTrain loss: {train_loss}\tValidation loss {validation_loss}\tDuration: {round((time.time()-prev_iteration_start_time)/60, 2)} minutes')
      prev_iteration_start_time = time.time()
    epoch += 1

  checkpoint_path = f'checkpoints/{config.task}'
  if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)
  with open(f'{checkpoint_path}/{str(time.time()).replace(".", "_")}.pkl', 'wb') as file:
    pickle.dump(best_params, file)

  print(f'\nLowest validation loss {lowest_validation_loss} reached at epoch {lowest_validation_loss_epoch}.')
  print(f'Training converged in {(time.time()-start_time)/60} minutes.')


if __name__ == '__main__':
  # task = 'string_reverse_encoder_decoder'
  # task = 'addition_encoder_decoder'
  # config = get_config(task)
  # train_encoder_decoder(config)

  # task = 'string_reverse_decoder_only'
  task = 'addition_decoder_only'
  config = get_config(task)
  train_decoder_only(config)

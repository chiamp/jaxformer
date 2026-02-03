import json
import pickle

import time

import jax.numpy as jnp

from data import load_data
from inference import make_teacher_force_forward_fn, make_autoregressive_encode_decode_fn, make_autoregressive_encode_decode_with_kv_cache_fn
from train import Config, get_config
from visualize_attention import plot_attention_grid


def eval_teacher_forcing(config: Config, file_name: str, n_samples: int = 10):
  # Load data
  input_data, target_data = load_data(config.task)  # (batch, sequence)

  # Load trained parameters
  with open(f'checkpoints/{config.task}/{file_name}.pkl', 'rb') as file:
    learned_params = pickle.load(file)

  teacher_force_forward = make_teacher_force_forward_fn(config)

  print(input_data[:n_samples,:])
  print(target_data[:n_samples,:])
  output_tokens, attention_scores_dict = teacher_force_forward(learned_params, input_data[:n_samples,:], target_data[:n_samples,:])
  print(output_tokens.argmax(-1))

  plot_attention_grid(
    attention_scores_dict,
    max_batch=2,
    tokens_dict={
      'encoder_tokens': [config.tokenizer.decode(sequence) for sequence in input_data[:2, :]],
      'decoder_tokens': [config.tokenizer.decode(sequence) for sequence in target_data[:2, :-1]],
    }
  )

def eval_autoregressive_decoding(config: Config, file_name: str):
  with open(f'data/{config.task}/max_sequence_length.json', 'r') as file:
    max_sequence_length = json.load(file)

  print(
    '\n'.join(
      [
        f'Input sample sequences for the {config.task} task.',
        f'Max sequence length is {max_sequence_length-2}.',  # make room for SOS and EOS tokens
        'After entering your sample sequence, press Enter to submit it.',
        'Press Enter on an empty string to finish submitting sequences\n',
      ]
    )
  )

  sequences: list[list[int]] = []

  sequence = input('')
  while sequence:
    sequence = config.tokenizer.SOS + sequence + config.tokenizer.EOS
    if len(sequence) > max_sequence_length:
      print('Input sequence is too long')
    else:
      sequence += (max_sequence_length - len(sequence)) * config.tokenizer.PAD
      assert len(sequence) == max_sequence_length, (sequence, len(sequence), max_sequence_length)
      sequences.append(config.tokenizer.encode(sequence))
    sequence = input('')

  input_data = jnp.array(sequences)  # (batch, sequence)

  # Load trained parameters
  with open(f'checkpoints/{config.task}/{file_name}.pkl', 'rb') as file:
    learned_params = pickle.load(file)

  for str_label, autoregressive_encode_decode_fn in (
    ('without kv cache', make_autoregressive_encode_decode_fn(config)),
    ('with kv cache', make_autoregressive_encode_decode_with_kv_cache_fn(config)),
  ):
    start_time = time.time()
    # output array is of shape # (batch, sequence-1)
    output_array, attention_scores_dict = autoregressive_encode_decode_fn(learned_params, input_data)
    print(f'Done decoding in {time.time() - start_time} seconds.')

    print(f'Decoded output sequences {str_label}:')
    for i in range(output_array.shape[0]):
      output_sequence = config.tokenizer.decode(output_array[i, :].tolist())
      start = output_sequence.index(config.tokenizer.SOS) if (config.tokenizer.SOS in output_sequence) else -1
      end = output_sequence.index(config.tokenizer.EOS) if (config.tokenizer.EOS in output_sequence) else None
      if end is None:
        print(output_sequence[start+1:])
      else:
        print(output_sequence[start+1:end])
    print()

  # plot_attention_grid(
  #   attention_scores_dict,
  #   max_batch=2,
  #   tokens_dict={
  #     'encoder_tokens': [tokenizer.decode(sequence) for sequence in input_data[:2, :]],
  #     'decoder_tokens': [tokenizer.decode(sequence) for sequence in target_data[:2, :-1]],
  #   }
  # )


if __name__ == '__main__':
  task = 'string_reverse'
  file_name = '1770142654_4459379'

  # task = 'addition'
  # file_name = '1770143050_624529'

  config = get_config(task)

  # eval_teacher_forcing(config, file_name)
  eval_autoregressive_decoding(config, file_name)
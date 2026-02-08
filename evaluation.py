import json
import pickle

import time

import jax.numpy as jnp

# from data import load_data
from inference import (
  # make_teacher_force_forward_fn,
  make_autoregressive_encode_decode_fn,
  make_autoregressive_encode_decode_with_kv_cache_fn,
  make_autoregressive_decode_only_with_kv_cache_fn,
)
from train import Config, get_config
# from visualize_attention import plot_attention_grid


# def eval_teacher_forcing(config: Config, file_name: str, n_samples: int = 10):
#   # Load data
#   input_data, target_data = load_data(config.task)  # (batch, sequence)

#   # Load trained parameters
#   with open(f'checkpoints/{config.task}/{file_name}.pkl', 'rb') as file:
#     learned_params = pickle.load(file)

#   teacher_force_forward = make_teacher_force_forward_fn(config)

#   print(input_data[:n_samples,:])
#   print(target_data[:n_samples,:])
#   output_tokens, attention_scores_dict = teacher_force_forward(learned_params, input_data[:n_samples,:], target_data[:n_samples,:])
#   print(output_tokens.argmax(-1))

#   plot_attention_grid(
#     attention_scores_dict,
#     max_batch=2,
#     tokens_dict={
#       'encoder_tokens': [config.tokenizer.decode(sequence) for sequence in input_data[:2, :]],
#       'decoder_tokens': [config.tokenizer.decode(sequence) for sequence in target_data[:2, :-1]],
#     }
#   )

def eval_encoder_decoder_autoregressive_decoding(config: Config, file_name: str):
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


def eval_decoder_only_autoregressive_decoding(config: Config, file_name: str):
  with open(f'data/{config.task}/max_sequence_length.json', 'r') as file:
    max_sequence_length = json.load(file)
  max_prompt_sequence_length = int((max_sequence_length-3) / 2)  # SOS, EOS and SEP tokens, and divide by two for prompt and response sequences

  print(
    '\n'.join(
      [
        f'Input sample sequences for the {config.task} task.',
        f'Max sequence length is {max_prompt_sequence_length}.',
        'After entering your sample sequence, press Enter to submit it.',
        'Press Enter on an empty string to finish submitting sequences\n',
      ]
    )
  )

  sequences: list[list[int]] = []

  sequence = input('')
  while sequence:
    if len(sequence) > max_prompt_sequence_length:
      print('Input sequence is too long')
    else:
      sequence = config.tokenizer.SOS + sequence + config.tokenizer.SEP
      sequence += (max_sequence_length - len(sequence)) * config.tokenizer.PAD
      assert len(sequence) == max_sequence_length, (sequence, len(sequence), max_sequence_length)
      sequences.append(config.tokenizer.encode(sequence))
    sequence = input('')

  input_data = jnp.array(sequences)  # (batch, sequence)

  # Load trained parameters
  with open(f'checkpoints/{config.task}/{file_name}.pkl', 'rb') as file:
    learned_params = pickle.load(file)


  autoregressive_decode_only_fn = make_autoregressive_decode_only_with_kv_cache_fn(config)
  start_time = time.time()
  # output array is of shape # (batch, sequence-1)
  output_array, attention_scores_dict = autoregressive_decode_only_fn(learned_params, input_data)
  print(f'Done decoding in {time.time() - start_time} seconds.')

  print(f'Decoded output sequences:')
  for i in range(output_array.shape[0]):
    output_sequence = config.tokenizer.decode(output_array[i, :].tolist())
    # `make_autoregressive_decode_only_with_kv_cache_fn` always outputs the output_sequence with the first token being
    # the first decoded token given the inputted separator token, so we don't have to do any parsing/splitting for an output separator token.
    end = output_sequence.index(config.tokenizer.EOS) if (config.tokenizer.EOS in output_sequence) else None
    if end is None:
      print(output_sequence)
    else:
      print(output_sequence[:end])
  print()


if __name__ == '__main__':
  # eval_teacher_forcing(config, file_name)

  # task = 'string_reverse_encoder_decoder'
  # file_name = '1770507851_7381482'
  # task = 'addition_encoder_decoder'
  # file_name = '1770522533_6116748'
  # config = get_config(task)
  # eval_encoder_decoder_autoregressive_decoding(config, file_name)

  # task = 'string_reverse_decoder_only'
  # file_name = '1770516728_693825'
  task = 'addition_decoder_only'
  file_name = '1770534126_438244'
  config = get_config(task)
  eval_decoder_only_autoregressive_decoding(config, file_name)
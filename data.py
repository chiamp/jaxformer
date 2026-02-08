import os
import json
import tqdm

import random

import numpy as np

import jax
import jax.numpy as jnp

from tokenizer import get_tokenizer


def generate_string_reverse_encoder_decoder_data(
    min_length=1,
    max_length=8,  # max length of input word
    n_samples=1e4,
):
  data_path = 'data/string_reverse_encoder_decoder'
  if not os.path.exists(data_path):
    os.mkdir(data_path)

  tokenizer = get_tokenizer('string_reverse_encoder_decoder')

  max_sequence_length = max_length + 2  # make room for SOS and EOS tokens

  words: list[str] = []
  reverse_words: list[str] = []
  tokenized_words: list[list[int]] = []
  tokenized_reverse_words: list[list[int]] = []
  for _ in tqdm.tqdm(range(int(n_samples)), desc='Generating string reverse data'):
    length = random.randint(min_length, max_length)
    word = ''.join(chr(random.randint(97, 122)) for _ in range(length))

    # Add SOS, EOS and padding tokens to make it fixed length of max_length
    padding = tokenizer.PAD * (max_sequence_length - len(word) - 2)
    reverse_word = tokenizer.SOS + word[::-1] + tokenizer.EOS + padding
    word = tokenizer.SOS + word + tokenizer.EOS + padding

    words.append(word)
    reverse_words.append(reverse_word)
    tokenized_words.append(tokenizer.encode(word))
    tokenized_reverse_words.append(tokenizer.encode(reverse_word))

  with open(f'{data_path}/input.json', 'w') as file:
    json.dump(words, file, indent=2)
  with open(f'{data_path}/target.json', 'w') as file:
    json.dump(reverse_words, file, indent=2)
  np.save(f'{data_path}/tokenized_input.npy', np.array(tokenized_words, dtype=np.int16))
  np.save(f'{data_path}/tokenized_target.npy', np.array(tokenized_reverse_words, dtype=np.int16))
  with open(f'{data_path}/vocab_size.json', 'w') as file:
    json.dump(tokenizer.VOCAB_SIZE, file)
  with open(f'{data_path}/max_sequence_length.json', 'w') as file:
    json.dump(max_sequence_length, file)  # make room for SOS and EOS tokens

def generate_string_reverse_decoder_only_data(
    min_length=1,
    max_length=8,  # max length of input+reversed word
    n_samples=1e4,
):
  data_path = 'data/string_reverse_decoder_only'
  if not os.path.exists(data_path):
    os.mkdir(data_path)

  tokenizer = get_tokenizer('string_reverse_decoder_only')

  max_sequence_length = max_length*2 + 3  # make room for SOS, EOS and SEP tokens

  words: list[str] = []
  # reverse_words: list[str] = []
  tokenized_words: list[list[int]] = []
  # tokenized_reverse_words: list[list[int]] = []
  for _ in tqdm.tqdm(range(int(n_samples)), desc='Generating string reverse data'):
    length = random.randint(min_length, max_length)
    word = ''.join(chr(random.randint(97, 122)) for _ in range(length))
    word = tokenizer.SOS + word + tokenizer.SEP + word[::-1] + tokenizer.EOS  # add the correct "response"/answer

    # Add padding tokens to make it fixed length of max_sequence_length
    padding = tokenizer.PAD * (max_sequence_length - len(word))
    word += padding

    words.append(word)
    tokenized_words.append(tokenizer.encode(word))

  with open(f'{data_path}/input.json', 'w') as file:
    json.dump(words, file, indent=2)
  np.save(f'{data_path}/tokenized_input.npy', np.array(tokenized_words, dtype=np.int16))
  with open(f'{data_path}/vocab_size.json', 'w') as file:
    json.dump(tokenizer.VOCAB_SIZE, file)
  with open(f'{data_path}/max_sequence_length.json', 'w') as file:
    json.dump(max_sequence_length, file)


def _generate_number(length: int, allow_leading_zero=False) -> str:
  assert length >= 0, length
  if length == 0:
    return ''
  if length == 1:
    return str(random.randint(0, 9)) if allow_leading_zero else str(random.randint(1, 9))
  return (
    (str(random.randint(0, 9)) if allow_leading_zero else str(random.randint(1, 9)))
    +
    ''.join(str(random.randint(0, 9)) for _ in range(length-1))
  )

def _generate_carry_chain_number(length: int, carry_chain_length: int) -> str:
  '''Generate a string representation of a number of length `length`,
  and ensure there is a carry chain of 9's of at least length `carry_chain_length`.
  We say "at least" since we cannot guarantee if the other digits sampled are 9's (which could extend the chainn) or not.'''
  assert length >= carry_chain_length >= 0, (length, carry_chain_length)
  insertion_index = random.randint(0, length-carry_chain_length)
  end_index = insertion_index + carry_chain_length - 1  # inclusive
  return (
    _generate_number(insertion_index) +
    ('9'*carry_chain_length) +
    _generate_number(length-end_index-1, allow_leading_zero=True)
  )


def generate_addition_encoder_decoder_data(
    min_length=1,
    max_length=5,  # max length of operand
    n_samples=5e4,
):
  data_path = 'data/addition_encoder_decoder'
  if not os.path.exists(data_path):
    os.mkdir(data_path)

  tokenizer = get_tokenizer('addition_encoder_decoder')

  max_sequence_length = max_length*2+1+2  # the max length of two operands, plus the '+' operator, plus SOS and EOS tokens

  input_sequences: list[str] = []
  target_sequences: list[str] = []
  tokenized_sequences: list[list[int]] = []
  tokenized_target_sequences: list[list[int]] = []
  for _ in tqdm.tqdm(range(int(n_samples)), desc='Generating addition data'):
    length_1 = random.randint(min_length, max_length)
    length_2 = random.randint(min_length, max_length)
    allow_leading_zero_1 = (length_1==1)  # can generate a single digit 0
    allow_leading_zero_2 = (length_2==1)

    if random.randint(0, 1):  # no carry chaining
      input_sequence_1 = _generate_number(length_1, allow_leading_zero_1)
      input_sequence_2 = _generate_number(length_2, allow_leading_zero_2)
    else:  # yes carry chaining
      carry_chain_length_1 = random.randint(min(2, length_1), length_1)  # anything less than length 2 is not a chain
      carry_chain_length_2 = random.randint(min(2, length_2), length_2)
      mode = random.randint(0, 2)
      match mode:
        case 0:  # only first sequence explicitly has carry chains
          input_sequence_1 = _generate_carry_chain_number(length_1, carry_chain_length_1)
          input_sequence_2 = _generate_number(length_2, allow_leading_zero_2)
        case 1:  # only second sequence explicitly has carry chains
          input_sequence_1 = _generate_number(length_1, allow_leading_zero_1)
          input_sequence_2 = _generate_carry_chain_number(length_2, carry_chain_length_2)
        case 2:  # both sequences explicitly have carry chains
          input_sequence_1 = _generate_carry_chain_number(length_1, carry_chain_length_1)
          input_sequence_2 = _generate_carry_chain_number(length_2, carry_chain_length_2)
        case _:
          raise ValueError()

    input_sequence = f'{tokenizer.SOS}{input_sequence_1}+{input_sequence_2}{tokenizer.EOS}'
    input_sequence += (max_sequence_length - len(input_sequence)) * tokenizer.PAD
    target_sequence = f'{tokenizer.SOS}{str(int(input_sequence_1)+int(input_sequence_2))}{tokenizer.EOS}'
    target_sequence += (max_sequence_length - len(target_sequence)) * tokenizer.PAD

    input_sequences.append(input_sequence)
    target_sequences.append(target_sequence)
    tokenized_sequences.append(tokenizer.encode(input_sequence))
    tokenized_target_sequences.append(tokenizer.encode(target_sequence))

  with open(f'{data_path}/input.json', 'w') as file:
    json.dump(input_sequences, file, indent=2)
  with open(f'{data_path}/target.json', 'w') as file:
    json.dump(target_sequences, file, indent=2)
  np.save(f'{data_path}/tokenized_input.npy', np.array(tokenized_sequences, dtype=np.int16))
  np.save(f'{data_path}/tokenized_target.npy', np.array(tokenized_target_sequences, dtype=np.int16))
  with open(f'{data_path}/vocab_size.json', 'w') as file:
    json.dump(tokenizer.VOCAB_SIZE, file)
  with open(f'{data_path}/max_sequence_length.json', 'w') as file:
    json.dump(max_sequence_length, file)


def generate_addition_decoder_only_data(
    min_length=1,
    max_length=5,  # max length of operand
    n_samples=5e4,
):
  data_path = 'data/addition_decoder_only'
  if not os.path.exists(data_path):
    os.mkdir(data_path)

  tokenizer = get_tokenizer('addition_decoder_only')

  max_prompt_length = max_length*2+1  # the max length of two operands, plus the '+' operator
  max_sequence_length = max_prompt_length*2+3 # prompt and response strings, plus SOS, EOS and SEP tokens

  input_sequences: list[str] = []
  tokenized_sequences: list[list[int]] = []
  for _ in tqdm.tqdm(range(int(n_samples)), desc='Generating addition data'):
    length_1 = random.randint(min_length, max_length)
    length_2 = random.randint(min_length, max_length)
    allow_leading_zero_1 = (length_1==1)  # can generate a single digit 0
    allow_leading_zero_2 = (length_2==1)

    if random.randint(0, 1):  # no carry chaining
      input_sequence_1 = _generate_number(length_1, allow_leading_zero_1)
      input_sequence_2 = _generate_number(length_2, allow_leading_zero_2)
    else:  # yes carry chaining
      carry_chain_length_1 = random.randint(min(2, length_1), length_1)  # anything less than length 2 is not a chain
      carry_chain_length_2 = random.randint(min(2, length_2), length_2)
      mode = random.randint(0, 2)
      match mode:
        case 0:  # only first sequence explicitly has carry chains
          input_sequence_1 = _generate_carry_chain_number(length_1, carry_chain_length_1)
          input_sequence_2 = _generate_number(length_2, allow_leading_zero_2)
        case 1:  # only second sequence explicitly has carry chains
          input_sequence_1 = _generate_number(length_1, allow_leading_zero_1)
          input_sequence_2 = _generate_carry_chain_number(length_2, carry_chain_length_2)
        case 2:  # both sequences explicitly have carry chains
          input_sequence_1 = _generate_carry_chain_number(length_1, carry_chain_length_1)
          input_sequence_2 = _generate_carry_chain_number(length_2, carry_chain_length_2)
        case _:
          raise ValueError()

    input_sequence = f'{input_sequence_1}+{input_sequence_2}'
    target_sequence = f'{str(int(input_sequence_1)+int(input_sequence_2))}'
    input_sequence = f'{tokenizer.SOS}{input_sequence}{tokenizer.SEP}{target_sequence}{tokenizer.EOS}'
    input_sequence += (max_sequence_length - len(input_sequence)) * tokenizer.PAD

    input_sequences.append(input_sequence)
    tokenized_sequences.append(tokenizer.encode(input_sequence))

  with open(f'{data_path}/input.json', 'w') as file:
    json.dump(input_sequences, file, indent=2)
  np.save(f'{data_path}/tokenized_input.npy', np.array(tokenized_sequences, dtype=np.int16))
  with open(f'{data_path}/vocab_size.json', 'w') as file:
    json.dump(tokenizer.VOCAB_SIZE, file)
  with open(f'{data_path}/max_sequence_length.json', 'w') as file:
    json.dump(max_sequence_length, file)


def load_data(task: str) -> tuple[jax.Array, jax.Array | None]:
  # with open(f'data/{task}/input.json', 'w') as file:
  #   json.dump(words, file, indent=2)
  # with open(f'data/{task}/target.json', 'w') as file:
  #   json.dump(reverse_words, file, indent=2)
  input_data = jnp.array(np.load(f'data/{task}/tokenized_input.npy'))
  if os.path.exists(f'data/{task}/tokenized_target.npy'):
    target_data = jnp.array(np.load(f'data/{task}/tokenized_target.npy'))
  else:
    target_data = None
  # with open(f'data/{task}/vocab_size.json', 'w') as file:
  #   vocab_size = json.load(StringReverseTokenizer.VOCAB_SIZE, file)
  # with open(f'data/{task}/max_sequence_length.json', 'w') as file:
  #   max_sequence_length = json.dump(max_length, file)
  return input_data, target_data  # (batch, sequence)


def compute_dataset_entropy(target_data: jax.Array, pad_token: int) -> tuple[float, float]:
  '''Compute maximum entropy and marginal entropy of the target dataset.

  Maximum entropy is if we assume that every output token is likely.
  This is calculated via `-sum_t(p_t * ln(p_t))`, where p_t is the frequency of a token t showing up in the target dataset (excluding pad tokens).
  But we know `p_t` is the same for every token because it's uniform. So `p_t = 1/vocab_size`.
  Then it follows that `-sum_(vocab_size)(1/vocab_size * ln(1/vocab_size)) = -1 * ln(1/vocab_size) = -ln(1/vocab_size)`.
  This gives us the entropy of a "totally ignorant" guesser that just randomly samples an output token.
  If the model's cross entropy loss is around this value, then the model has learned NOTHING.

  Marginal entropy is if we output a token at a probability equal to the observed frequency of that token in the target dataset.
  This is calculated via `-sum_t(p_t * ln(p_t))`, where p_t is the frequency of a token t showing up in the target dataset (excluding pad tokens).
  It's called marginal entropy because we are essentially calculating P(Y) instead of P(Y|X);
  we essentially ignore the input X (sum up all probabilities conditioned on X) and always just output a token based on the frequency observed in the target dataset.
  If the model's cross entropy loss is around this value, then the model has learned to approximate the frequency of the dataset, but completely ignores the input.

  Mathematically, it is IMPOSSIBLE to get a cross entropy loss below the marginal entropy from just purely blindly guessing or using statistic biases (i.e. without using information from the input).
  Therefore if the model loss goes below the marginal entropy, then the model is definitely learning some kind of mapping from input to output.

  If you get a cross entropy loss that's greater than the maximum entropy, that means the model is performing WORSE than a random guesser.
  This means that the model is confidently wrong (e.g. guessing high probability for output tokens that should be low probability, etc.) instead of just being ignorant.

  This function returns: tuple[maximum_entropy, marginal_entropy]
  '''
  unique_tokens, counts = jnp.unique(target_data, return_counts=True)
  counts = counts[unique_tokens!=pad_token]  # omit pad token
  unique_tokens = unique_tokens[unique_tokens!=pad_token]  # omit pad token

  max_entropy = -jnp.log(1 / jnp.size(unique_tokens))

  occurrence_prob = counts / jnp.sum(counts)
  marginal_entropy = -jnp.sum(occurrence_prob * jnp.log(occurrence_prob))

  return (max_entropy.item(), marginal_entropy.item())


if __name__ == '__main__':
  # generate_string_reverse_encoder_decoder_data()
  # generate_string_reverse_decoder_only_data()
  # generate_addition_encoder_decoder_data()
  # generate_addition_decoder_only_data()

  pass


# TODO: add other tasks
# - addition
# - multiplication
# - list integer sorting
# - nested arithmetic expressions (bedmas)
# - all tasks above TOGETHER
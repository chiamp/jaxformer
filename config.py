import functools
import dataclasses

import optax

from tokenizer import Tokenizer, get_tokenizer


@dataclasses.dataclass(kw_only=True)
class Config:
  task: str

  # Dataset
  batch_size: int
  train_split_ratio: float  # how much of the dataset to allocate for the train dataset and the rest to the validation dataset

  # Train loop
  lr_schedule: optax.Schedule
  validation_loss_cutoff: float  # if the validation loss reaches this cutoff or below, terminate training
  max_num_train_epochs: int | None  # if None, training will continue forever until one of the other termination conditions trigger
  max_patience: int  # if the validation loss doesn't improve after max_patience, terminate training
  eval_every_n_epochs: int

  # Model params
  num_embedding_features: int  # d_model
  num_query_key_features: int  # d_k
  num_value_features: int  # d_v
  num_heads: int
  num_inner_dense_features: int  # d_ff
  num_encoder_layers: int  # n
  num_decoder_layers: int  # n

  seed: int

  def __post_init__(self):
    if (self.num_heads * self.num_query_key_features) != self.num_embedding_features:
      raise ValueError(f'Number of heads ({self.num_heads}) multiplied by the number of query/key features ({self.num_query_key_features}) does not equal the number of embedding features ({self.num_embedding_features}).')
    if (self.num_heads * self.num_value_features) != self.num_embedding_features:
      raise ValueError(f'Number of heads ({self.num_heads}) multiplied by the number of value features ({self.num_value_features}) does not equal the number of embedding features ({self.num_embedding_features}).')

  @functools.cached_property
  def tokenizer(self) -> type[Tokenizer]:
    return get_tokenizer(self.task)

  @functools.cached_property
  def vocab_size(self) -> int:
    return self.tokenizer.VOCAB_SIZE

  @functools.cached_property
  def sos_index(self) -> int:
    return self.tokenizer.SOS_INDEX

  @functools.cached_property
  def eos_index(self) -> int:
    return self.tokenizer.EOS_INDEX

  @functools.cached_property
  def pad_index(self) -> int:
    return self.tokenizer.PAD_INDEX

  @functools.cached_property
  def sep_index(self) -> int:
    return self.tokenizer.SEP_INDEX


STRING_REVERSE_ENCODER_DECODER_CONFIG = Config(
  task='string_reverse_encoder_decoder',
  batch_size=64,
  train_split_ratio=0.9,
  lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=3e-4,
    peak_value=3e-4,
    warmup_steps=400,
    decay_steps=3500,
    end_value=1e-6,
  ),
  validation_loss_cutoff=0.03,
  max_num_train_epochs=None,
  max_patience=50,
  eval_every_n_epochs=20,
  num_embedding_features=128,
  num_query_key_features=64,
  num_value_features=64,
  num_heads=2,
  num_inner_dense_features=2048,
  num_encoder_layers=2,
  num_decoder_layers=2,
  seed=0,
)

STRING_REVERSE_DECODER_ONLY_CONFIG = Config(
  task='string_reverse_decoder_only',
  batch_size=64,
  train_split_ratio=0.9,
  lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=3e-4,
    peak_value=3e-4,
    warmup_steps=400,
    decay_steps=3500,
    end_value=1e-6,
  ),
  validation_loss_cutoff=0.03,
  max_num_train_epochs=None,
  max_patience=100,
  eval_every_n_epochs=20,
  num_embedding_features=128,
  num_query_key_features=64,
  num_value_features=64,
  num_heads=2,
  num_inner_dense_features=2048,
  num_encoder_layers=2,
  num_decoder_layers=2,
  seed=0,
)


ADDITION_ENCODER_DECODER_CONFIG = Config(
  task='addition_encoder_decoder',
  batch_size=64,
  train_split_ratio=0.9,
  lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=3e-4,
    peak_value=3e-4,
    warmup_steps=1600,
    decay_steps=5000,
    end_value=1e-5,
  ),
  validation_loss_cutoff=0.03,
  max_num_train_epochs=None,
  max_patience=20,
  eval_every_n_epochs=20,
  num_embedding_features=128,
  num_query_key_features=64,
  num_value_features=64,
  num_heads=2,
  num_inner_dense_features=2048,
  num_encoder_layers=2,
  num_decoder_layers=2,
  seed=0,
)

ADDITION_DECODER_ONLY_CONFIG = Config(
  task='addition_decoder_only',
  batch_size=64,
  train_split_ratio=0.9,
  lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-3,
    peak_value=1e-3,
    warmup_steps=1600,
    # decay_steps=5000,
    decay_steps=15000,
    end_value=5e-5,
  ),
  validation_loss_cutoff=0.03,
  max_num_train_epochs=None,
  max_patience=1000,
  eval_every_n_epochs=20,
  num_embedding_features=128,
  num_query_key_features=64,
  num_value_features=64,
  num_heads=2,
  num_inner_dense_features=2048,
  num_encoder_layers=2,
  num_decoder_layers=2,
  seed=0,
)


def get_config(task: str) -> Config:
  match task:
    case 'string_reverse_encoder_decoder':
      return STRING_REVERSE_ENCODER_DECODER_CONFIG
    case 'string_reverse_decoder_only':
      return STRING_REVERSE_DECODER_ONLY_CONFIG
    case 'addition_encoder_decoder':
      return ADDITION_ENCODER_DECODER_CONFIG
    case 'addition_decoder_only':
      return ADDITION_DECODER_ONLY_CONFIG
    case _:
      raise ValueError(f'Unknown task {task}')

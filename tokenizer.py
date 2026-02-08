

class Tokenizer:

  PAD: str = ''
  SOS: str = ''
  EOS: str = ''
  SEP: str = ''  # separator token for decoder-only tasks
  PAD_INDEX: int = -1
  SOS_INDEX: int = -1
  EOS_INDEX: int = -1
  SEP_INDEX: int = -1

  VOCAB_SIZE: int = -1

  @classmethod
  def encode(cls, sentence: str) -> list[int]:
    ...

  @classmethod
  def decode(cls, tokenized_sentence: list[int]) -> str:
    ...


class StringReverseTokenizer(Tokenizer):

  PAD = ' '
  SOS = '>'
  EOS = '<'
  SEP = '='
  PAD_INDEX = 0
  SOS_INDEX = 1
  EOS_INDEX = 2
  SEP_INDEX = 3

  VOCAB_SIZE = 26+4  # alphabet plus PAD, SOS, EOS and SEP tokens

  @classmethod
  def encode(cls, sentence: str) -> list[int]:
    tokens: list[int] = []
    for ch in sentence:
      match ch:
        case cls.PAD:
          tokens.append(cls.PAD_INDEX)
        case cls.SOS:
          tokens.append(cls.SOS_INDEX)
        case cls.EOS:
          tokens.append(cls.EOS_INDEX)
        case cls.SEP:
          tokens.append(cls.SEP_INDEX)
        case _:
          tokens.append(ord(ch) - 97 + 4)  # shift alphabet index by 3 to make room for special tokens
    return tokens

  @classmethod
  def decode(cls, tokenized_sentence: list[int]) -> str:
    characters: list[str] = []
    for token in tokenized_sentence:
      match token:
        case cls.PAD_INDEX:
          characters.append(cls.PAD)
        case cls.SOS_INDEX:
          characters.append(cls.SOS)
        case cls.EOS_INDEX:
          characters.append(cls.EOS)
        case cls.SEP_INDEX:
          characters.append(cls.SEP)
        case _:
          characters.append(chr(token + 97 - 4))  # shift alphabet index by 3 to make room for special tokens
    return ''.join(characters)


class AdditionTokenizer(Tokenizer):

  PAD = ' '
  SOS = '>'
  EOS = '<'
  SEP = '='
  PAD_INDEX = 11
  SOS_INDEX = 12
  EOS_INDEX = 13
  SEP_INDEX = 14

  PLUS_INDEX = 10

  VOCAB_SIZE = 10+1+4  # 0-9, '+', PAD, SOS EOS, and SEP tokens

  @classmethod
  def encode(cls, sentence: str) -> list[int]:
    tokens: list[int] = []
    for ch in sentence:
      match ch:
        case cls.PAD:
          tokens.append(cls.PAD_INDEX)
        case cls.SOS:
          tokens.append(cls.SOS_INDEX)
        case cls.EOS:
          tokens.append(cls.EOS_INDEX)
        case cls.SEP:
          tokens.append(cls.SEP_INDEX)
        case '+':
          tokens.append(cls.PLUS_INDEX)
        case _:
          tokens.append(int(ch))
    return tokens

  @classmethod
  def decode(cls, tokenized_sentence: list[int]) -> str:
    characters: list[str] = []
    for token in tokenized_sentence:
      match token:
        case cls.PAD_INDEX:
          characters.append(cls.PAD)
        case cls.SOS_INDEX:
          characters.append(cls.SOS)
        case cls.EOS_INDEX:
          characters.append(cls.EOS)
        case cls.SEP_INDEX:
          characters.append(cls.SEP)
        case cls.PLUS_INDEX:
          characters.append('+')
        case _:
          characters.append(str(token))
    return ''.join(characters)


def get_tokenizer(task: str) -> type[Tokenizer]:
  match task:
    case 'string_reverse_encoder_decoder' | 'string_reverse_decoder_only':
      return StringReverseTokenizer
    case 'addition_encoder_decoder' | 'addition_decoder_only':
      return AdditionTokenizer
    case _:
      raise ValueError(f'Invalid task name {task}')

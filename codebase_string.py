'''For printing out the entire codebase so you can copy and paste it to Gemini for feedback.'''

import inspect
import pyperclip

import config
import tokenizer
import data
import parameters
import layers
import inference
import train
import evaluation


def get_codebase_string():
  module_strings: list[str] = [
     f'{name}.py:\n```\n{inspect.getsource(module).strip()}\n```'
     for name, module in [
        ('config', config),
        ('tokenizer', tokenizer),
        ('data', data),
        ('parameters', parameters),
        ('layers', layers),
        ('inference', inference),
        ('train', train),
        ('evaluation', evaluation),
     ]
  ]
  pyperclip.copy('\n\n'.join(module_strings))


if __name__ == '__main__':
   get_codebase_string()

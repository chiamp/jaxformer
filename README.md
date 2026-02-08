# Jaxformer
This repo has JAX implementations of the following:
- the encoder-decoder model from the [Attention is all you need paper](https://arxiv.org/pdf/1706.03762)
- the same encoder-decoder model as above with KV caching
- a decoder-only model with KV caching

# Tasks
All models were trained on the following tasks:
- string reversal (max input character limit: 8)
- addition of two operands (max input character limit per operand: 5)

In reality, there are "four" tasks, since I separated the tasks based on model type:
- `string_reverse_encoder_decoder`
- `string_reverse_decoder_only`
- `addition_encoder_decoder`
- `addition_decoder_only`

# Test
You can test out the models on the tasks above by running `python evaluation.py`. Edit the main block to try out different tasks.

# Files and Directories
- `evaluation.py`: test the JAX models I pre-trained on the tasks
- `train.py`: train your own checkpoints of the JAX models. To test out your saved checkpoints, copy the filename of the checkpoint (excluding the file extension) and paste it into the main block in `evaluation.py` and run `python evaluation.py`.
- `data.py`: generate new data for the tasks (WARNING: this will overwrite the current data, unless you move the existing data elsewhere)
- `config.py`: configs for training. There's a separate config for each of the four tasks. The default config values guarantee convergence for the default generated data. (WARNING: The `addition_decoder_only` task for the default config values will converge to a validation loss value of `0.03668` at step/epoch 30880, but the script won't stop until step/epoch 50900 because the `max_patience` hyper parameter is set very high. All the other tasks should converge to a validation loss value of `0.03` at a much shorter step/epoch with the default config values).
- `parameters.py`: contains a function to instantiate the model parameters
- `layers.py`: contains the layers of the transformer (implemented as inference/forward functions)
- `inference.py`: contains teacher-forcing and auto-regressive decoding functions that call the layer functions from `layers.py`
- `tokenizer.py`: contains the tokenizer class for each task
- `codebase_string.py`: script that copies the entire codebase as a string so I can copy and paste it to Gemini for feedback/debugging
- `visualize_attention.py`: for visualizing attention weights. (WARNING: does not work currently).
- `data/`: directory where the generated data for each task is written to
- `checkpoints/`: directory where the model checkpoints for each task are saved
- `basic_jax_examples/`: unrelated directory where I was practicing basic JAX scripts for linear regression
- `requirements.txt`: install by running `pip install -r requirements.txt`. (NOTE: there are some unnecessary dependencies here, but I just copied my virtual environment's dependencies over by running `pip freeze`). I'm using Python version `3.10.10`.
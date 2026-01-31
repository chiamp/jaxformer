# attention weights are of shape (layer, batch, head, sequence, sequence)


import matplotlib.pyplot as plt
import jax.numpy as jnp


def plot_attention_grid(attention_scores_dict, max_batch=5, tokens_dict=None):
  """
  Visualizes attention scores with full nesting: Batch -> Layer -> Type -> Head.

  Args:
    attention_scores_dict: Dict containing 'encoder_attention_scores', etc.
    max_batch: Maximum number of batch samples to iterate over.
    tokens_dict: Optional dict with 'encoder_tokens' and 'decoder_tokens'
           (lists of lists of strings) for axis labeling.
  """

  config = {
    'encoder_attention_scores': {
      'title': 'Encoder Self-Attention',
      'x_token_type': 'encoder_tokens',
      'y_token_type': 'encoder_tokens'
    },
    'masked_attention_scores': {
      'title': 'Decoder Masked Self-Attention',
      'x_token_type': 'decoder_tokens',
      'y_token_type': 'decoder_tokens'
    },
    'cross_attention_scores': {
      'title': 'Decoder-Encoder Cross-Attention',
      'x_token_type': 'encoder_tokens', # Key comes from Encoder
      'y_token_type': 'decoder_tokens'  # Query comes from Decoder
    }
  }

  # Infer dimensions
  # Shape assumption: (num_layers, batch, num_heads, seq_q, seq_k)
  first_key = list(attention_scores_dict.keys())[0]
  total_batch = attention_scores_dict[first_key].shape[1]
  num_layers = attention_scores_dict[first_key].shape[0]

  # 1. Loop over Batch Index
  for batch_idx in range(min(total_batch, max_batch)):

    # 2. Loop over Layers
    for layer_idx in range(num_layers):
      print(f"\n=== Batch: {batch_idx} | Layer: {layer_idx} ===")

      # 3. Loop over Attention Types
      for key in ['encoder_attention_scores', 'masked_attention_scores', 'cross_attention_scores']:
        if key not in attention_scores_dict: continue

        scores = attention_scores_dict[key]
        num_heads = scores.shape[2]

        # 4. Loop over Heads
        for head_idx in range(num_heads):

          # Extract matrix: [Layer, Batch, Head, Seq_Q, Seq_K]
          matrix = scores[layer_idx, batch_idx, head_idx]

          # --- Plotting ---
          plt.figure(figsize=(6, 6))
          plt.imshow(matrix, cmap='viridis')

          # Move x-axis to top
          ax = plt.gca()
          ax.xaxis.tick_top()
          ax.xaxis.set_label_position('top')

          # Labeling
          if tokens_dict:
            x_type = config[key]['x_token_type']
            y_type = config[key]['y_token_type']

            if x_type in tokens_dict:
              x_labels = tokens_dict[x_type][batch_idx]
              # Slice labels to match matrix shape (handles padding/truncation mismatch)
              plt.xticks(range(len(x_labels)), x_labels[:matrix.shape[1]], rotation=90)

            if y_type in tokens_dict:
              y_labels = tokens_dict[y_type][batch_idx]
              plt.yticks(range(len(y_labels)), y_labels[:matrix.shape[0]])

          plt.title(f"{config[key]['title']}\nBatch {batch_idx} | Layer {layer_idx} | Head {head_idx}", y=1.15)
          plt.xlabel(f"Key ({'Input' if 'cross' in key or 'encoder' in key else 'Target'})")
          plt.ylabel(f"Query ({'Target' if 'masked' in key or 'cross' in key else 'Input'})")
          plt.colorbar(fraction=0.046, pad=0.04)
          plt.show()

          # Critical: Close figure to prevent OOM
          plt.close()


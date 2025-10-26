import tensorflow as tf
from tensorflow.keras import layers, Model

# Positional encoding (simple sinusoidal)
class PositionalEncoding(layers.Layer):
    def call(self, x):
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[2]
        positions = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        dims = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000, (2 * (dims // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = positions * angle_rates
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return x + pos_encoding[tf.newaxis, :, :]

# Transformer encoder block
def transformer_encoder(inputs, num_heads, key_dim, ff_dim, dropout=0.1):
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    attn_output = layers.Dropout(dropout)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn = tf.keras.Sequential([
        layers.Dense(ff_dim, activation='relu'),
        layers.Dense(inputs.shape[-1])
    ])
    ffn_output = ffn(out1)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

# ConvTransformer for time series
def build_convtransformer(seq_len=100, num_features=1):
    inputs = layers.Input(shape=(seq_len, num_features))
    
    # Convolutional feature extraction
    x = layers.Conv1D(64, 3, activation='relu', padding='causal')(inputs)
    x = layers.Conv1D(128, 3, activation='relu', padding='causal')(x)
    
    # Positional encoding
    x = PositionalEncoding()(x)
    
    # Transformer encoder blocks
    for _ in range(2):
        x = transformer_encoder(x, num_heads=4, key_dim=64, ff_dim=128)
    
    # Global average pooling and output
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)
    
    return Model(inputs, outputs)

# Build and compile
model = build_convtransformer(seq_len=100, num_features=1)
model.compile(optimizer='adam', loss='mse')
model.summary()

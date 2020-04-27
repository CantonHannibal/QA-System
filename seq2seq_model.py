import tensorflow_addons as tfa
import tensorflow as tf

#Encoder
encoder = tf.keras.layers.LSTM(num_units, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_emb_inp)
encoder_state = [state_h, state_c]

# Sampler
sampler = tfa.seq2seq.sampler.TrainingSampler()

# Decoder
decoder_cell = tf.keras.layers.LSTMCell(num_units)
projection_layer = tf.keras.layers.Dense(num_outputs)
decoder = tfa.seq2seq.BasicDecoder(
    decoder_cell, sampler, output_layer=projection_layer)

outputs, _, _ = decoder(
    decoder_emb_inp,
    initial_state=encoder_state,
    sequence_length=decoder_lengths)
logits = outputs.rnn_output

#Attention
attention_mechanism = tfa.seq2seq.LuongAttention(
    num_units,
    encoder_state,
    memory_sequence_length=encoder_sequence_length)

decoder_cell = tfa.seq2seq.AttentionWrapper(
    decoder_cell, attention_mechanism,
    attention_layer_size=num_units)
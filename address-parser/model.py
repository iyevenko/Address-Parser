import json
import os

import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Embedding, GRU, Dense, SpatialDropout1D

import dataset as ds

import numpy as np

import matplotlib.pyplot as plt


CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', "saved_models", "model1.ckpt")


def model_fn(embedding_dim, GRU_units, vocab_size, show_summary=False):
    model = tf.keras.Sequential([
        Embedding(vocab_size, embedding_dim, batch_input_shape=[None, None], mask_zero=True),
        SpatialDropout1D(0.2),
        Bidirectional(GRU(units=GRU_units,
                          dropout=0.5,
                          recurrent_dropout=0.5,
                          return_sequences=True), merge_mode='sum'),
        Dense(units=15, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    if show_summary:
        model.summary()

    return model

def train_model(model, dataset, epochs):
    callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=True, verbose=1)

    history = model.fit(x=dataset['train'], epochs=epochs, callbacks=[callback])
    loss_plot = plt.plot(history['loss'])
    plt.show(loss_plot)

    loss, accuracy = model.evaluate(x=dataset['test'])

    print('Test Loss: {}'.format(loss))
    print('Test Accuracy: {}'.format(accuracy))



def predictions_to_json(address, predictions):
    prev_p = -1
    addr_idx = 0
    curr_str = ""

    data = []

    for p in predictions:
        if p != prev_p and prev_p > -1:
            data.append((curr_str, ds.classes[prev_p]))
            curr_str = ""

        curr_str += address[addr_idx]
        addr_idx += 1

        prev_p = p

    data.append((curr_str, ds.classes[predictions[-1]]))

    return json.dumps(data)


def predict_one(address):
    tokenizer = ds.get_saved_tokinizer()

    vocab_size = len(tokenizer.word_index) + 1
    model = model_fn(embedding_dim=256, GRU_units=128, vocab_size=vocab_size)
    model.load_weights(CHECKPOINT_PATH)

    sequence = tokenizer.texts_to_sequences([address.upper()])

    predictions = model.predict(x=sequence, batch_size=1)
    # 1 x addr_len x num_classes -> addr_len
    predictions = np.argmax(predictions, axis=2)[0]
    # noinspection PyTypeChecker
    print(predictions_to_json(address, predictions))

if __name__ == '__main__':
    ds.generate_en_us_dataset(n_x=1e6)

    tokenizer = ds.get_saved_tokinizer()
    model = model_fn(embedding_dim=256, GRU_units=128,  vocab_size=len(tokenizer.word_index)+1, show_summary=True)
    dataset = ds.input_fn(128, 1e6)

    train_model(model, dataset, epochs=10)

    predict_one("799 E Drgram Suite 5A, Tucson AZ 85705, USA")
    predict_one("1600 Amphitheatre Pkwy, Mountain View, CA 94043, United States")


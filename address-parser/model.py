import json
import os

import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Embedding, GRU, Dense

import dataset as ds

import numpy as np

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', "saved_models", "model1.ckpt")


def model_fn(embedding_dim, GRU_units, vocab_size, show_summary=False):
    model = tf.keras.Sequential([
        Embedding(vocab_size, embedding_dim, batch_input_shape=[None, None], mask_zero=True),
        Bidirectional(GRU(units=GRU_units, dropout=0.8, recurrent_dropout=0.8, return_sequences=True),
                      merge_mode='sum'),
        Dense(units=15, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    if show_summary:
        model.summary()

    return model


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
    with open('/Users/iyevenko/Documents/GitHub/Address-Parser/address-parser/tokenizer.json') as f:
        tokenizer_json = f.readline()
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

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
    predict_one("914 Maramis crt")


    # model = model_fn(embedding_dim=256, GRU_units=128,  vocab_size=)
    #
    # loss, accuracy = model.evaluate(x=dataset['test'])
    # print('Test Loss: {}'.format(loss))
    # print('Test Accuracy: {}'.format(accuracy))
    # callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=True, verbose=1)
    # model.fit(x=dataset['train'], epochs=5, callbacks=[callback])

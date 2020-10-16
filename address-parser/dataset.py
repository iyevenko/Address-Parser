import os
import random

import tensorflow as tf

ADDRESS_FILE = '/Users/iyevenko/Documents/GitHub/Address-Parser/data/libpostal-parser-training-data-20170304/openaddresses_formatted_addresses_tagged.random.tsv'
TOKENIZER_JSON_FILE = '/Users/iyevenko/Documents/GitHub/Address-Parser/address-parser/tokenizer.json'
classes = ['road',
           'SEP',
           'house_number',
           'FSEP',
           'city',
           'state_district',
           'state',
           'postcode',
           'suburb',
           'country',
           'unit',
           'city_district',
           'island',
           'country_region',
           'world_region']
classes = list(s.upper() for s in classes)
inv_classes = {classes[i] : i for i in range(len(classes))}

def input_fn(batch_size, dataset_size):
    DATA_PATH = '/Users/iyevenko/Documents/GitHub/Address-Parser/data/en-us'

    labels = tf.data.TextLineDataset(os.path.join(DATA_PATH, 'labels'))
    labels = labels.map(lambda labels: tf.strings.split(labels))
    labels = labels.map(lambda labels: tf.strings.to_number(labels, tf.int32))

    with open(os.path.join(DATA_PATH, 'addresses')) as f:
        texts = [x.strip() for x in f.readlines()]

    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="\t\n", lower=False, char_level=True)
    tokenizer.fit_on_texts(texts)

    with open(TOKENIZER_JSON_FILE, 'w') as f:
        f.write(tokenizer.to_json())

    sequences = tokenizer.texts_to_sequences(texts)

    def generator():
        for x in sequences:
            yield x

    addresses = tf.data.Dataset.from_generator(generator, output_types=tf.int32, output_shapes=(None, ))

    full_dataset = tf.data.Dataset.zip((addresses, labels))
    full_dataset = full_dataset.padded_batch(batch_size, padded_shapes=None, drop_remainder=True)

    train_size = int(0.8 * int(dataset_size / batch_size))
    train_set = full_dataset.take(train_size)
    test_set = full_dataset.skip(train_size)

    dataset = {
        'train': train_set,
        'test': test_set
    }

    return dataset


def get_saved_tokinizer():
    with open(TOKENIZER_JSON_FILE, 'r') as f:
        json_string = f.readline()

    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
    return tokenizer


def generate_en_us_dataset(labels_filename='labels', inputs_filename='addresses', n_x=DATASET_SIZE):
    labels_path = os.path.join('/Users/iyevenko/Documents/GitHub/Address-Parser/data/en-us', labels_filename)
    inputs_path = os.path.join('/Users/iyevenko/Documents/GitHub/Address-Parser/data/en-us', inputs_filename)
    with open(labels_path, 'w') as labels_file, \
            open(inputs_path, 'w') as inputs_file, \
            open(ADDRESS_FILE) as data:

        count = 0
        while count < n_x or n_x < 0:
            raw_line = data.readline().upper()
            if raw_line == '':
                break
            if raw_line[0:2] == 'EN' and raw_line[3:5] == 'US':
                generate_input_output_pairs(raw_line[6:], labels_file, inputs_file)
                count += 1
        print('Saved %d labelled addresses to \n%s\n%s' % (count, labels_path, inputs_path))


def get_random_separator():
    separators = ['',',', ' |']
    idx = random.randint(0, len(separators)-1)
    return separators[idx]


def generate_input_output_pairs(labelled_line, labels_file, inputs_file):
    split = [(x.split('/')[0], x.split('/')[1]) for x in labelled_line.split()]

    input_addr = ''
    labels = ''
    first_label = True

    for text, label in split:
        if label == 'SEP' or label == 'FSEP':
            input_addr = input_addr[:-1]
            text = get_random_separator()
            label = 'SEP'
        elif first_label == False:
            labels += str(inv_classes['SEP']) + ' '

        first_label = False
        input_addr += text + ' '

        if label not in inv_classes.keys():
            # something wrong with label
            return

        for _ in range(len(text)):
            labels += str(inv_classes[label]) + ' '

    input_addr = input_addr[:-1]
    labels = labels[:-1]

    inputs_file.write(input_addr)
    labels_file.write(labels)

    inputs_file.write('\n')
    labels_file.write('\n')
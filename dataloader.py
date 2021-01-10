'''Data Loader

2021-01-09 first created
'''

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers
K = tfk.backend


def preprocess(data_file, speaker, normalize=True, one_hot=True, num_pc=None,
               spkr_list=['F01', 'F02', 'F03', 'F04',
                          'M01', 'M02', 'M03', 'M04'],
               vowel_list=['IY1', 'IH1', 'EH1', 'AE1',
                           'AA1', 'AH1', 'AO1', 'UH1', 'UW1'],
               artic_col=['TRx', 'TRz', 'TBx', 'TBz', 'TTx', 'TTz',
                          'JAWx', 'JAWz', 'ULx', 'ULz', 'LLx', 'LLz'],
               acous_col=['F1', 'F2', 'F3']):
    '''Preprocess data'''
    assert speaker in spkr_list, f'Speaker ({speaker}) does not exist in {spkr_list}'
    assert os.path.exists(data_file), f'Data file ({data_file}) does not exsit'
    assert len(vowel_list) == len(set(vowel_list)), f'vowel list is not unique'
    if num_pc is not None:
        assert num_pc > 1, f'num_pc must be larger than 1 at least (current={num_pc})'

    df = pd.read_csv(data_file, index_col=None)
    artic = df.loc[(df.Speaker == speaker) & (
        df.Vowel.isin(vowel_list)), artic_col].values
    acous = df.loc[(df.Speaker == speaker) & (
        df.Vowel.isin(vowel_list)), acous_col].values

    assert artic.shape[0] > 0, 'artic data is empty'
    assert acous.shape[0] > 0, 'acous data is empty'

    labels = df.loc[(df.Speaker == speaker) & (
        df.Vowel.isin(vowel_list)), 'Vowel'].values
    vowel2idx = {v: i for i, v in enumerate(vowel_list)}
    idx2vowel = {vowel2idx[v]: v for v in vowel2idx.keys()}
    n_vowels = len(set(vowel_list))
    labels = np.array([vowel2idx[label] for label in labels], dtype='int32')
    assert labels.shape[0] > 0, 'label data is empty'

    params = {}
    # Normalize data using z-scoring
    if normalize:
        scaler = StandardScaler()
        scaler.fit(artic)
        artic = scaler.transform(artic)
        params['artic'] = {
            'mean': scaler.mean_.tolist(), 'sd': scaler.var_.tolist()}

        scaler = StandardScaler()
        scaler.fit(acous)
        acous = scaler.transform(acous)
        params['acous'] = {
            'mean': scaler.mean_.tolist(), 'sd': scaler.var_.tolist()}

    # Run PCA on articulatory data
    if num_pc is not None:
        pca = PCA(n_components=num_pc)
        pca.fit(artic)
        artic = pca.transform(artic)
        params['artic']['pca'] = pca.components_.tolist()

    # One-hot encoding
    if one_hot:
        labels = np.eye(n_vowels)[labels].astype('int32')
        params['vowel2idx'] = vowel2idx
        params['idx2vowel'] = idx2vowel

    artic = artic.astype('float32')
    acous = acous.astype('float32')
    labels = labels.astype('int32')
    return artic, acous, labels, params


def gen(x_data, y_data):
    '''Generate examples'''
    for x, y in zip(x_data, y_data):
        yield x, y


def make_dataloader(x_data, y_data, x_type, y_type, batch_size):
    '''Data loader with tf.data'''
    N = x_data.shape[0]
    x_dim = x_data.shape[1]
    y_dim = y_data.shape[1]
    dataset = tf.data.Dataset.from_generator(
        gen, args=(x_data, y_data),
        output_types=(x_type, y_type),
        output_shapes=(x_dim, y_dim)
    )
    dataset = dataset.shuffle(N).batch(
        batch_size, drop_remainder=True).repeat()
    return dataset


if __name__ == '__main__':
    # Settings
    batch_size = 10
    data_file = 'data/data.csv'

    # Make data loader and text it
    artic, acous, label, params = preprocess(
        'data/data.csv', 'F01', normalize=True, one_hot=True, num_pc=3)
    data_loader = make_dataloader(artic, label, 'float32', 'int32', batch_size)
    it = iter(data_loader)
    next(it)

    print('Done')

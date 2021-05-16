import pickle
import numpy as np
import opensmile
import argparse

import torch
import torchaudio
import pandas as pd

from multiprocessing import Pool, cpu_count
from itertools import repeat
from speechbrain.pretrained import EncoderClassifier


# get list of files
def get_files(file_fp):
    file_list = []

    # get relative path of file for outside access
    relative_path = file_fp.split('/')
    if len(relative_path) > 1:
        relative_path = '/'.join(file_fp.split('/')[:-1]) + '/'
    else:
        relative_path = ''

    with open(file_fp, 'r') as f:
        for fline in f:
            data = fline.split()
            # create list of files with metadata
            file_list.append((relative_path + data[0], data[2], data[3], data[4]))
    return file_list


def get_feat_smile(file_in, csv, smile, header=False):
    # get metadata
    fp, spk_id, age, gender = file_in

    # output is a pandas df
    y = smile.process_file(fp)

    # include name as index
    y.reset_index(inplace=True)

    # add metada columns
    y['id'] = spk_id
    y['age'] = age
    y['gender'] = gender
    y.drop(['start', 'end'], axis=1, inplace=True)

    with open(csv, 'a') as f:
        y.to_csv(f, index=False, header=header)


def get_feat_xvector(file_in, csv, header=False):

    # get metadata
    fp, spk_id, age, gender = file_in

    # ---- Create embeddings ----
    # - Xvector
    # classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    # - ECAPA-TDNN
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                                savedir="pretrained_models/spkrec-xvect-voxceleb")
    signal, fs = torchaudio.load(fp)
    embedding = classifier.encode_batch(signal)

    # create dataframe from metadata and embedding
    y = {'file': fp, 'id': spk_id, 'age': age, 'gender': gender}
    y2 = pd.DataFrame(data=y, index=[0])
    y = pd.DataFrame(embedding.squeeze()).T
    result = pd.concat([y2, y], axis=1).reindex(y2.index)
    with open(csv, 'a') as f:
        result.to_csv(f, index=False, header=header)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process file.')

    parser.add_argument('--file', '-f', required=True, help='file list')
    parser.add_argument('--out', '-o', required=True, help='output feature file')
    parser.add_argument('--feat', '-x', required=True, choices=['eGeMAPS', 'xvector'], help='Feature type')

    args = parser.parse_args()

    file_list = get_files(args.file)

    if args.feat == 'eGeMAPS':
        smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                feature_level=opensmile.FeatureLevel.Functionals)

        # write 1 example to include header
        get_feat_smile(file_list[0], args.out, smile, header=True)

        # generate iterable with arguments to func
        iterable = zip(file_list[1:], repeat(args.out), repeat(smile))

        func = get_feat_smile

    elif args.feat == 'xvector':
        get_feat_xvector(file_list[0], args.out, header=True)
        iterable = zip(file_list[1:], repeat(args.out))
        func = get_feat_xvector

    # create pool of workers for multiprocessing
    p = Pool(cpu_count())
    p.starmap(func, iterable)
    p.close()

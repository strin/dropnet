import caffe
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array

import time
import os
import sys

import numpy as np
import numpy as np
import numpy.random as npr

import theano

import hickle as hkl

from proc_load import crop_and_mirror

import lmdb
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight = False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(unicode(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


class Timer(object):
    def __init__(self, name=None, output=sys.stdout):
        self.name = name
        if output and type(output) == str:
            self.output = open(output, 'w')
        else:
            self.output = output

    def __enter__(self):
        if self.name:
            print >>self.output, colorize('[%s]\t' % self.name, 'green'),
        print >>self.output, colorize('Start', 'green')
        self.tstart = time.time()
        self.output.flush()

    def __exit__(self, type, value, traceback):
        if self.name:
            print >>self.output, colorize('[%s]\t' % self.name, 'green'),
        print >>self.output, colorize('Elapsed: %s' % (time.time() - self.tstart),
                                      'green')
        self.output.flush()


def choice(objs, size, replace=True, p=None):
    all_inds = range(len(objs))
    inds = npr.choice(all_inds, size=size, replace=replace, p=p)
    return [objs[int(ind)] for ind in inds]

def proc_configs(config):
    if not os.path.exists(config['weights_dir']):
        os.makedirs(config['weights_dir'])
        print "Creat folder: " + config['weights_dir']

    return config


def unpack_configs(config):
    mean_file = config['mean_file']
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(mean_file, 'rb' ).read()
    blob.ParseFromString(data)
    mean_arr = np.array( caffe.io.blobproto_to_array(blob) )[0]
    mean_arr = mean_arr.astype(theano.config.floatX)

    train_file = config['train_file']
    test_file = config['test_file']

    return (train_file, test_file, mean_arr)





def adjust_learning_rate(config, epoch, step_idx, val_record, learning_rate):
    # Adapt Learning Rate
    if config['lr_policy'] == 'step':
        if epoch == config['lr_step'][step_idx]:
            learning_rate.set_value(
                np.float32(learning_rate.get_value() / 10))
            step_idx += 1
            if step_idx >= len(config['lr_step']):
                step_idx = 0  # prevent index out of range error
            print 'Learning rate changed to:', learning_rate.get_value()

    if config['lr_policy'] == 'auto':
        if (epoch > 5) and (val_record[-3] - val_record[-1] <
                            config['lr_adapt_threshold']):
            learning_rate.set_value(
                np.float32(learning_rate.get_value() / 10))
            print 'Learning rate changed to::', learning_rate.get_value()

    return step_idx



def get_rand3d():
    tmp_rand = np.float32(np.random.rand(3))
    tmp_rand[2] = round(tmp_rand[2])
    return tmp_rand


class LMDBImageSet(object):
    def __init__(self, lmdb_env, mean_arr = np.array([])):
        self.lmdb_env = lmdb_env
        self.mean_arr = mean_arr
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        self.length = len([1 for _ in lmdb_cursor])
        self.datum = caffe.proto.caffe_pb2.Datum()
        # go through lmdb keys.
        self.keys = []
        lmdb_txn = self.lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        for key, _ in lmdb_cursor:
            self.keys.append(key)


    def __len__(self):
        return self.length

    def __getitem__(self, k):
        assert k >= 0 and k < self.length
        assert type(k) == int
        count = 0
        lmdb_txn = self.lmdb_env.begin()
        key = self.keys[k]
        value = lmdb_txn.get(key)
        self.datum.ParseFromString(value)
        img = caffe.io.datum_to_array(self.datum)
        # preprocess image.
        img = np.array(img, dtype=theano.config.floatX) # convert to single.
        if len(img.shape) == 2:
            old_img = img
            img = np.zeros(img.shape + (3,), dtype=theano.config.floatX)
            img[:, :, 0] = old_img
            img[:, :, 1] = old_img
            img[:, :, 2] = old_img
        if self.mean_arr.shape:
            img = img - self.mean_arr # substract mean.
        label = self.datum.label
        return (img, label)


def load_lmdb(path, mean_arr):
    '''
    given the path to imdb data. return (images, labels) pair.
    all data are read in memory for fast processing.
    not feasible for large datasets.
    '''
    lmdb_env = lmdb.open(path)
    data_list = []
    data_label = []
    imageset = LMDBImageSet(lmdb_env, mean_arr)
    return imageset



def sample_minibatch(train_set, batch_size):
    minibatch_list = choice(train_set, batch_size)
    im_label = np.zeros(batch_size, dtype=int)
    for (pi, (img, label)) in enumerate(minibatch_list):
        if pi == 0:
            im_input = np.zeros(img.shape + (batch_size,), dtype=theano.config.floatX)
        im_input[:, :, :, pi] = img
        im_label[pi] = label

    param_rand = get_rand3d()
    im_input = crop_and_mirror(im_input, param_rand, flag_batch=False)

    return (im_input, im_label)



import sys
import os
import time

import yaml
import numpy as np

sys.path.append('./lib')

from tools import (save_weights, load_weights,
                   save_momentums, load_momentums)
from common import *
import theano

theano.config.floatX = 'float32'

# LOAD CONFIGS.
with open('config.yaml', 'r') as f:
    config = yaml.load(f)

config = proc_configs(config)

# UNPACK CONFIGS
(train_file, test_file, img_mean) = unpack_configs(config)

import theano.sandbox.cuda
theano.sandbox.cuda.use(config['gpu'])
import theano
theano.config.on_unused_input = 'warn'

from layers import DropoutLayer
from alex_net import AlexNet, compile_models

## BUILD NETWORK ##
model = AlexNet(config)
layers = model.layers
batch_size = model.batch_size

## LOAD DATASET.
with Timer('loading training data'):
    train_set = load_lmdb(train_file, img_mean)


with Timer('loading test data'):
    test_set = load_lmdb(test_file, img_mean)

## COMPILE FUNCTIONS ##
(train_model, validate_model, train_error, learning_rate,
    shared_x, shared_y, rand_arr, vels) = compile_models(model, config)


######################### TRAIN MODEL ################################
print '... training'


# Start Training Loop
epoch = 0
step_idx = 0
val_record = []

while epoch < config['n_epochs']:
    epoch = epoch + 1

    if config['resume_train'] and epoch == 1:
        load_epoch = config['load_epoch']
        load_weights(layers, config['weights_dir'], load_epoch)
        lr_to_load = np.load(
            config['weights_dir'] + 'lr_' + str(load_epoch) + '.npy')
        val_record = list(
            np.load(config['weights_dir'] + 'val_record.npy'))
        learning_rate.set_value(lr_to_load)
        load_momentums(vels, config['weights_dir'], load_epoch)
        epoch = load_epoch + 1

    n_batches = len(train_set) / batch_size
    for it in range(n_batches):
        num_iter = (epoch - 1) * len(train_set) + it
        print 'epoch', epoch, 'num_iter', num_iter, '/', n_batches

        with Timer('sample minibatch'):
            (batch_x, batch_y) = sample_minibatch(train_set, batch_size)
            shared_x.set_value(batch_x)
            shared_y.set_value(batch_y)

        with Timer('forward-backward pass'):
            cost = train_model()

        if num_iter % config['print_freq'] == 0:
            print 'training @ iter = ', num_iter
            print 'training cost:', cost
            if config['print_train_error']:
                print 'training error rate:', train_error()

    ############### Test on Validation Set ##################

    DropoutLayer.SetDropoutOff()

    def validate(dataset):
        validation_losses = []
        validation_errors = []

        for di_offset in range(0, len(dataset), batch_size):
            for di in range(di_offset, min(di_offset + batch_size, len(dataset))):
                (val_img, label) = dataset[di]
                if di == 0:
                    im_input = np.zeros(val_img.shape + (batch_size,), dtype=theano.config.floatX)
                    im_label = np.zeros(batch_size, dtype=int)
                im_input[:, :, :, di - di_offset] = val_img
                im_label[di - di_offset] = label

            param_rand = [0.5,0.5,0]
            im_input_cropped = crop_and_mirror(im_input, param_rand, flag_batch=True)
            shared_x.set_value(im_input_cropped)
            shared_y.set_value(im_label)

            loss, error = validate_model()
            validation_losses.append(loss)
            validation_errors.append(error)

        return (np.mean(validation_losses),
                np.mean(validation_errors))


    (val_loss, val_error) = validate(test_set)


    print('epoch %i: validation loss %f ' %
            (epoch, val_loss))
    print('epoch %i: validation error %f %%' %
            (epoch, val_error * 100.))

    val_record.append([val_error, val_loss])
    np.save(config['weights_dir'] + 'val_record.npy', val_record)

    DropoutLayer.SetDropoutOn()
    ############################################

    # Adapt Learning Rate
    step_idx = adjust_learning_rate(config, epoch, step_idx,
                                    val_record, learning_rate)

    # Save weights
    if epoch % config['snapshot_freq'] == 0:
        save_weights(layers, config['weights_dir'], epoch)
        np.save(config['weights_dir'] + 'lr_' + str(epoch) + '.npy',
                    learning_rate.get_value())
        save_momentums(vels, config['weights_dir'], epoch)

print('Optimization complete.')


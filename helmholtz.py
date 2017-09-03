#!/bin/python3

"""
A Helmholtz machine trained with the Wake Sleep algorithm.
"""

import numpy

import os
import sys
import random
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.optim as optim

from torchvision import datasets, transforms

import gflags


try:
    import visdom
    USING_VISDOM = True
except:
    USING_VISDOM = False


FLAGS = gflags.FLAGS


class HelmholtzMachine(nn.Module):
    """
    A Helmholtz machine trained with the Wake Sleep algorithm.
    """

    _awake = True
    _verbose = True

    def __init__(self, layers=[784, 128, 32]):
        super(HelmholtzMachine, self).__init__()

        assert len(layers) >= 3, "We require at least two sets of weights. |I| = 1, |J| >= 1, |K| = 1"
        
        self.layers = layers
        self.num_layers = len(self.layers) - 1

        # recognition layers
        for ii, i in enumerate(range(self.num_layers)):
            self.set_r(ii, nn.Linear(self.layers[i], self.layers[i+1]))

        # generation layers
        for ii, i in enumerate(range(self.num_layers)[::-1]):
            self.set_g(ii, nn.Linear(self.layers[i+1], self.layers[i]))

        self.g_bias = Parameter(torch.FloatTensor(self.layers[-1]))

        self.reset_parameters()

    def reset_parameters(self):
        self.g_bias.data.uniform_(-1, 1)

    def set_verbose(self, verbose=True):
        self._verbose = verbose

    def r(self, i):
        return getattr(self, "recognition_{}".format(i))

    def set_r(self, i, layer):
        setattr(self, "recognition_{}".format(i), layer)

    def g(self, i):
        return getattr(self, "generation_{}".format(i))

    def set_g(self, i, layer):
        setattr(self, "generation_{}".format(i), layer)

    def wake(self):
        self._awake = True

    def sleep(self):
        self._awake = False

    def layer_output(self, x, training=True):
        """
        If training, treat x as bernoulli distribution and sample output,
        otherwise simply round x, giving binary output in either case.
        """
        if training:
            out = torch.bernoulli(x).detach()
        else:
            out = torch.round(x)
        return out

    def _run_wake_recognition(self, x):
        results = []

        # Run recognition layers, saving stochastic outputs.
        for i in range(self.num_layers):
            x = self.r(i)(x)
            x = F.sigmoid(x)
            x = self.layer_output(x, self.training)
            results.append(x)

        return results

    def _run_wake_generation(self, x_original, recognition_outputs):
        results = []

        # Run generative layers, predicting the input to each layer.
        for i in range(self.num_layers):
            x_input = recognition_outputs[-(i+1)]
            if i == self.num_layers - 1:
                x_target = x_original
            else:
                x_target = recognition_outputs[-(i+2)]
            x = self.g(i)(x_input)
            x = F.sigmoid(x)
            results.append(nn.BCELoss()(x, x_target))

        return results

    def run_wake(self, x):
        x_first = x
        batch_size = x.size(0)

        # Run Recognition Net.
        recognition_outputs = self._run_wake_recognition(x)

        # Fit the bias to the final layer.
        x_last = recognition_outputs[-1]
        x = self.g_bias.view(1, -1).expand(batch_size, self.g_bias.size(0))
        x = F.sigmoid(x)
        generation_bias_loss = nn.BCELoss()(x, x_last)

        # Run Generation Net.
        generation_loss = self._run_wake_generation(x_first, recognition_outputs)

        return recognition_outputs, generation_bias_loss, generation_loss

    def _run_sleep_recognition(self, x_initial, generative_outputs):
        results = []

        # Run recognition layers to predict fantasies.
        for i in range(self.num_layers):
            x_input = generative_outputs[-(i+1)]
            if i == self.num_layers - 1:
                x_target = x_initial
            else:
                x_target = generative_outputs[-(i+2)]
            x = self.r(i)(x_input)
            x = F.sigmoid(x)
            results.append(nn.BCELoss()(x, x_target))

        return results

    def _run_sleep_generation(self, x_initial):
        results = []

        # Fantasize each layers output.
        for i in range(self.num_layers):
            if i == 0:
                x = self.g(i)(x_initial)
            else:
                x = self.g(i)(x)
            x = F.sigmoid(x)
            x = self.layer_output(x, self.training)
            results.append(x)

        return results

    def run_sleep(self, x):
        batch_size = x.size(0)
        recognition_loss = []

        # We do not use the input `x`, rather we use the bias.
        bias = self.g_bias.view(1, -1)
        x = F.sigmoid(bias)
        x = x.expand(batch_size, self.g_bias.size(0))
        x = self.layer_output(x, self.training)
        generation_bias_output = x

        # Fantasize each layers output.
        generative_outputs = self._run_sleep_generation(generation_bias_output)

        # Run recognition layers to predict fantasies.
        recognition_loss = self._run_sleep_recognition(generation_bias_output, generative_outputs)

        return recognition_loss, generation_bias_output, generative_outputs
        
    def forward(self, x):
        if self._awake:
            out = self.run_wake(x)
        else:
            out = self.run_sleep(x)
        return out


class Logger(object):

    def __init__(self, using_visdom=True):
        self.using_visdom = using_visdom

        if self.using_visdom:
            self.vis = visdom.Visdom()

        self.prev_dict = dict()

    def visualize(self, x):
        if not self.using_visdom:
            return

        toshow = x.cpu().data.view(-1, 28, 28).numpy()
        self.vis.images(numpy.split(toshow, toshow.shape[0]))

    def visualize_latent_space(self, model):
        if not self.using_visdom:
            return

        K = 8

        x_base = model.g_bias.view(1, -1)
        x_base = F.sigmoid(x_base).round()
        x_base = x_base.data
        inputs = []

        model.eval()
        kk_both = random.sample(range(x_base.size(1)), k=K*2-1)
        kk1, kk2 = kk_both[:K], kk_both[K:]
        for i in range(K):
            x_new = x_base.clone()
            for ii in range(i+1):
                x_new[0, kk1[ii]] -= 1
                x_new = x_new.abs()
            for jj in range(K-1):
                x_new[0, kk2[jj]] -= 1
                x_new = x_new.abs()
                inputs.append(x_new.clone())

        inputs = Variable(torch.cat(inputs, 0))
        outputs = model._run_sleep_generation(inputs)
        toshow = outputs[-1].cpu().data.view(-1, 28, 28).numpy()
        self.vis.images(numpy.split(toshow, toshow.shape[0]))

    def _log(self, step, key, val):
        prevX, prevY, win, update = self.prev_dict.get(key, (0, 0, None, None))
        X = numpy.array([prevX, step])
        Y = numpy.array([prevY, val])
        opts = dict(legend=[key], title=key, xlabel='Training Steps', ylabel='Loss')
        win = self.vis.line(X=X, Y=Y, win=win, opts=opts, update=update)
        self.prev_dict[key] = (step, val, win, 'append')

    def log(self, step, generation_bias_loss, generative_loss, recognition_loss, total_loss):
        if not self.using_visdom:
            return

        self._log(step, 'gbias', generation_bias_loss.data[0])
        for i, loss in enumerate(generative_loss):
            self._log(step, 'g_{}'.format(i), loss.data[0])
        for i, loss in enumerate(recognition_loss):
            self._log(step, 'r_{}'.format(i), loss.data[0])
        self._log(step, 'total', total_loss.data[0])


def data_iterator(batch_size=16):
    data_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.expanduser('~/data/mnist'), train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           lambda x: x.view(-1),  # flatten
                           lambda x: x.round(),  # binarize
                           # transforms.Normalize((0.1307,), (0.3081,)),
                       ])),
        batch_size=batch_size, shuffle=True)
    while True:
        for data, target in data_loader:
            yield Variable(data)


def main():
    gflags.DEFINE_boolean("verbose", False, "Set to True for additional log output.")
    gflags.DEFINE_integer("max_iterations", 100000, "Number of total training steps.")
    gflags.DEFINE_integer("batch_size", 16, "Batch size.")
    gflags.DEFINE_integer("log_every", 100, "Log every N steps.")
    gflags.DEFINE_integer("vis_every", 1000, "Visualize examples every N steps.")
    gflags.DEFINE_integer("plot_every", 100, "Plot loss every N steps.")
    gflags.DEFINE_float("learning_rate", 0.01, "Learning rate.")

    FLAGS(sys.argv)

    iterations = FLAGS.max_iterations
    it = data_iterator(batch_size=FLAGS.batch_size)
    log_every = FLAGS.log_every
    vis_every = FLAGS.vis_every
    plot_every = FLAGS.plot_every
    logger = Logger(USING_VISDOM)

    model = HelmholtzMachine()

    print(model)

    opt = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    model.set_verbose(FLAGS.verbose)

    for step in range(iterations):
        model.train()
        
        model.wake()
        recognition_outputs, generation_bias_loss, generative_loss = model.forward(next(it))

        model.sleep()
        recognition_loss, generation_bias_output, generative_outputs = model.forward(next(it))

        total_loss = 0.0
        total_loss += generation_bias_loss
        for i, loss in enumerate(generative_loss):
            total_loss += loss
        for i, loss in enumerate(recognition_loss):
            total_loss += loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if step % log_every == 0:
            print("Step: {}".format(step))
            sys.stdout.write("\n")
            print("\tgeneration_bias_loss", generation_bias_loss.data[0])
            for i, loss in enumerate(generative_loss):
                print("\tgenerative loss", i, loss.data[0])
            for i, loss in enumerate(recognition_loss):
                print("\trecognition loss", i, loss.data[0])
            sys.stdout.write("\n")
            print("\ttotal loss", total_loss.data[0])
            sys.stdout.write("\n")

        if vis_every > 0 and step % vis_every == 0:
            # logger.visualize(generative_outputs[-1])
            logger.visualize_latent_space(model)

        if plot_every > 0 and step % plot_every == 0:
            logger.log(step, generation_bias_loss, generative_loss, recognition_loss, total_loss)


if __name__ == '__main__':
    main()

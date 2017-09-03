#!/bin/python3

"""
A Helmholtz machine trained with the Wake Sleep algorithm.
"""

import numpy

import os
import sys

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

    r0 : 784 x 100
    r1 : 100 x 10

    gbias : 10
    g0 : 10 x 100
    g1 : 100 x 784

    """

    _awake = True
    _verbose = True

    def __init__(self, layers=[784, 100, 10]):
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

    def run_wake(self, x):
        batch_size = x.size(0)
        recognition_outputs = []
        generation_loss = []

        recognition_outputs.append(x)

        # First run recognition layers, saving stochastic outputs.
        for i in range(self.num_layers):
            x = self.r(i)(x)
            x = F.sigmoid(x)
            x = self.layer_output(x, self.training)
            recognition_outputs.append(x)
            if self._verbose:
                print("wake", "r", i, x)

        # Fit the bias to the final layer.
        x_last = recognition_outputs[-1]
        x = self.g_bias.view(1, -1).expand(batch_size, self.g_bias.size(0))
        x = F.sigmoid(x)
        generation_loss.append(nn.BCELoss()(x, x_last))

        # Then run generative layers, predicting the input to each layer.
        for i in range(self.num_layers):
            # TODO: Right now, only considers the recognition outputs, but should
            # have the ability to reuse generative outputs.
            x_input = recognition_outputs[-(i+1)]
            x_target = recognition_outputs[-(i+2)]
            x = self.g(i)(x_input)
            x = F.sigmoid(x)
            generation_loss.append(nn.BCELoss()(x, x_target))

        return recognition_outputs, generation_loss

    def run_sleep(self, x):
        batch_size = x.size(0)
        recognition_loss = []
        generative_outputs = []

        # We do not use the input `x`, rather we use the bias.
        bias = self.g_bias.view(1, -1)
        x = F.sigmoid(bias)
        x = x.expand(batch_size, self.g_bias.size(0))
        x = self.layer_output(x, self.training)
        generative_outputs.append(x)

        # First fantasize each layers output.
        for i in range(self.num_layers):
            x = self.g(i)(x)
            x = F.sigmoid(x)
            x = self.layer_output(x, self.training)
            generative_outputs.append(x)
            if self._verbose:
                print("sleep", "g", i, x)

        # Then run recognition layers to predict fantasies.
        for i in range(self.num_layers):
            x_input = generative_outputs[-(i+1)]
            x_target = generative_outputs[-(i+2)]
            x = self.r(i)(x_input)
            x = F.sigmoid(x)
            recognition_loss.append(nn.BCELoss()(x, x_target))

        return recognition_loss, generative_outputs
        
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

    def visualize(self, x):
        if not self.using_visdom:
            return

        toshow = x.cpu().data.view(-1, 28, 28).numpy()
        self.vis.images(numpy.split(toshow, toshow.shape[0]))


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
    gflags.DEFINE_float("learning_rate", 0.01, "Learning rate.")

    FLAGS(sys.argv)

    iterations = FLAGS.max_iterations
    it = data_iterator(batch_size=FLAGS.batch_size)
    log_every = FLAGS.log_every
    vis_every = FLAGS.vis_every
    logger = Logger(USING_VISDOM)

    model = HelmholtzMachine()

    print(model)

    opt = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    model.set_verbose(FLAGS.verbose)

    model.train()
    for step in range(iterations):
        
        model.wake()
        recognition_outputs, generative_loss = model.forward(next(it))

        model.sleep()
        recognition_loss, generative_outputs = model.forward(next(it))

        total_loss = 0.0
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
            for i, loss in enumerate(generative_loss):
                print("\tgenerative loss", i, loss.data[0])
            for i, loss in enumerate(recognition_loss):
                print("\trecognition loss", i, loss.data[0])
            sys.stdout.write("\n")
            print("\ttotal loss", total_loss.data[0])
            sys.stdout.write("\n")

        if step % vis_every == 0:
            logger.visualize(generative_outputs[-1])


if __name__ == '__main__':
    main()

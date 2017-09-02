#!/bin/python3

"""
A Helmholtz machine trained with the Wake Sleep algorithm.
"""

import numpy

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import gflags


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

        self.g_bias = nn.Parameter(torch.FloatTensor(self.layers[-1]))

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
            out = torch.bernoulli(x)
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


def fake_data_iterator(batch_size=3, size=784):
    while True:
        yield Variable(torch.from_numpy(numpy.random.randn(batch_size, size)).float())


def main():
    gflags.DEFINE_boolean("verbose", False, "Set to True for additional log output.")
    gflags.DEFINE_integer("max_iterations", 100, "Number of total training steps.")
    gflags.DEFINE_integer("batch_size", 3, "Batch size.")

    FLAGS(sys.argv)

    iterations = FLAGS.max_iterations
    it = fake_data_iterator(batch_size=FLAGS.batch_size)

    model = HelmholtzMachine()

    print(model)

    model.set_verbose(FLAGS.verbose)

    model.train()
    for step in range(iterations):
        print("Step: {}".format(step))

        model.wake()
        recognition_outputs, generative_loss = model.forward(next(it))

        model.sleep()
        recognition_loss, generative_outputs = model.forward(next(it))



if __name__ == '__main__':
    main()

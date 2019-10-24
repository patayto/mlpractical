from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, LayerWithParameters, StochasticLayer, StochasticLayerWithParameters
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import AdamLearningRule
from mlp.optimisers import Optimiser
from collections import defaultdict
import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider, EMNISTDataProvider
import sys

def train_model_and_plot_stats(
		model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True):
	# As well as monitoring the error over training also monitor classification
	# accuracy i.e. proportion of most-probable predicted classes being equal to targets
	data_monitors = {'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

	# Use the created objects to initialise a new Optimiser instance.
	optimiser = Optimiser(
		model, error, learning_rule, train_data, valid_data, data_monitors, notebook=notebook)

	# Run the optimiser for 5 epochs (full passes through the training set)
	# printing statistics every epoch.
	stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

	return optimiser, stats, keys, run_time


# I don't need original model, but it's good for sanity checking that the code is doing what I think it should be
def createModelWithBestParameterSettings(original_model, best_model, final_parameters, best_parameters, verbose=False):
	model_params = [layer.params for layer in original_model.layers if
	                isinstance(layer, LayerWithParameters) or isinstance(layer, StochasticLayerWithParameters)]
	model_params_flat = [item for sublist in model_params for item in sublist]

	if verbose:
		print("Model parameters equal to final model params: {}".format(final_parameters == original_model.params))
		print("Model parameters equal to best model params: {}".format(
			bool(np.prod([(a == b).prod() for (a, b) in zip(best_parameters, original_model.params)]))))
		print("Final parameters equal to best model params: {}".format(
			bool(np.prod([(a == b).prod() for (a, b) in zip(best_parameters, final_parameters)]))))

	reshaped_best_params = [list(a) for a in np.array(best_parameters).reshape(np.array(model_params).shape)]

	layers_with_params = [isinstance(layer, LayerWithParameters) or isinstance(layer, StochasticLayerWithParameters) for
	                      layer in best_model.layers]

	# update parameters of model to the parameters of the best setting of the model
	paramsUpdated = 0  # we have N layers, but only K have params, with K <= N, therefore reshaped_params is of size K, while model.layers is size N, with N-K layers which should be ignored, and K which should be updated
	for layerNum, hasParams in enumerate(layers_with_params):
		if hasParams:
			best_model.layers[layerNum].params = reshaped_best_params[paramsUpdated]
			paramsUpdated += 1

	if verbose:
		print("New model parameters equal to final model params: {}".format(
			bool(np.prod([(a == b).prod() for (a, b) in zip(final_parameters, best_model.params)]))))
		print("New model parameters equal to best model params: {}".format(
			bool(np.prod([(a == b).prod() for (a, b) in zip(best_parameters, best_model.params)]))))
		print("Final parameters equal to best model params: {}".format(
			bool(np.prod([(a == b).prod() for (a, b) in zip(final_parameters, best_parameters)]))))
		print("New model parameters equal to original model parameters: {}".format(
			bool(np.prod([(a == b).prod() for (a, b) in zip(best_model.params, original_model.params)]))))

	return best_model


# evaluate test set accuracy
def acc(y, t):
	return (y.argmax(-1) == t.argmax(-1)).mean()


def calculateAccuracy(model, data, verbose=False):
	accs = []
	for inputs_batch, targets_batch in data:
		activations = model.fprop(inputs_batch)
		accs.append(acc(activations[-1], targets_batch))

	if verbose:
		print("Test set accuracy on best model: {}".format(np.mean(accs)))

	return np.mean(accs)


def createBestModelAndCalculateAccuracy(original_model, best_model, final_parameters, best_parameters, data, batch_size=100):
	# set best model parameters
	createModelWithBestParameterSettings(original_model, best_model, final_parameters, best_parameters)

	# prepare data
	data.reset()
	data.batch_size = batch_size

	# calculate accuracy
	acc = calculateAccuracy(best_model, data, verbose=True)

	return acc

# assumption is that numHiddenLayers >= 1 (else it's not a MultiLayerModel)
def createMultiLayerReLUModel(numHiddenLayers, input_dim, hidden_dim, output_dim, rng):
	weights_init = GlorotUniformInit(rng=rng)
	biases_init = ConstantInit(0.)

	model_layers = [
		AffineLayer(input_dim, hidden_dim, weights_init, biases_init)
	]

	for i in range(numHiddenLayers):
		model_layers.append(ReluLayer())

		if i == numHiddenLayers - 1:
			# final numHiddenLayers
			model_layers.append(AffineLayer(hidden_dim, output_dim, weights_init, biases_init))
		else:
			model_layers.append(AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init))

	return MultipleLayerModel(model_layers)


def NLayerBaselines(numHiddenLayers, rng, seed, train_data, valid_data, test_data, batch_size, num_epochs=100, stats_interval=1):
	# setup hyperparameters
	input_dim, output_dim = 784, 47

	hidden_dims = [32, 64, 128]
	layer_stats = defaultdict(tuple)

	for hidden_dim in hidden_dims:
		print("******* {} hidden units *******".format(hidden_dim))
		# Reset random number generator and data provider states on each run
		# to ensure reproducibility of results
		rng.seed(seed)
		train_data.reset()
		valid_data.reset()

		# Alter data-provider batch size
		train_data.batch_size = batch_size
		valid_data.batch_size = batch_size

		model = createMultiLayerReLUModel(numHiddenLayers, input_dim, hidden_dim, output_dim, rng)

		error = CrossEntropySoftmaxError()
		# Use a basic gradient descent learning rule
		learning_rule = AdamLearningRule()

		# Remember to use notebook=False when you write a script to be run in a terminal
		stats = train_model_and_plot_stats(
			model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=False)

		layer_stats[hidden_dim] = stats

	layer_accs = []

	for (num_units, stats) in layer_stats.items():
		print("1 layer, {} hidden units per layer:".format(num_units))

		# initialise empty model with the same architecture of the trained model
		best_model = createMultiLayerReLUModel(numHiddenLayers, input_dim, hidden_dim, output_dim, rng)

		acc = createBestModelAndCalculateAccuracy(stats[0].model, best_model, stats[0].model.params, stats[0].best_model_params, test_data)
		print()

		layer_accs.append(acc)

	return layer_accs

def printSampleMeanAndSD(samples, num_layers, num_samples):
	samples = np.array(list(samples.values()))

	layer_sizes = {0: 32, 1: 64, 2: 128}

	print("Mean and SD of test set accuracy of {} layer network ({} samples)".format(num_layers, num_samples))

	for (i, num_units) in layer_sizes.items():
		layer_mean = samples[:, i].mean()
		layer_std = samples[:, i].std()
		print("\t{} units per layer: {} ± {}".format(num_units, layer_mean, layer_std))


# The below code will set up the data providers, random number
# generator and logger objects needed for training runs. As
# loading the data from file take a little while you generally
# will probably not want to reload the data providers on
# every training run. If you wish to reset their state you
# should instead use the .reset() method of the data providers.
def main(numSamples):

	oneLayerSamples = defaultdict(list)
	twoLayerSamples = defaultdict(list)
	threeLayerSamples = defaultdict(list)

	for i in range(0, numSamples):
		print("************ Sample {} ************".format(i+1))
		# Seed a random number generator
		rng = np.random.RandomState(i)
		batch_size = 100
		# Set up a logger object to print info about the training run to stdout
		logger = logging.getLogger()
		logger.setLevel(logging.INFO)
		logger.handlers = [logging.StreamHandler()]

		# Create data provider objects for the MNIST data set
		train_data = EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
		valid_data = EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)
		test_data = EMNISTDataProvider('test', batch_size=batch_size, rng=rng)

		oneLayerSample = NLayerBaselines(1, rng, i, train_data, valid_data, test_data, batch_size, num_epochs=100)
		twoLayerSample = NLayerBaselines(2, rng, i, train_data, valid_data, test_data, batch_size, num_epochs=100)
		threeLayerSample = NLayerBaselines(3, rng, i, train_data, valid_data, test_data, batch_size, num_epochs=100)

		oneLayerSamples[i] = oneLayerSample
		twoLayerSamples[i] = twoLayerSample
		threeLayerSamples[i] = threeLayerSample

	printSampleMeanAndSD(oneLayerSamples, 1, numSamples)
	printSampleMeanAndSD(twoLayerSamples, 2, numSamples)
	printSampleMeanAndSD(threeLayerSamples, 3, numSamples)

if __name__ == "__main__":
	numSamples = int(sys.argv[1])
	main(numSamples)

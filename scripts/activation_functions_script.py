from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import AdamLearningRule
from mlp.optimisers import Optimiser
from collections import defaultdict
import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider, EMNISTDataProvider
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import AdamLearningRule
from mlp.optimisers import Optimiser
from mlp.layers import LayerWithParameters, StochasticLayer, StochasticLayerWithParameters
from mlp.layers import LeakyReluLayer, ParametricReluLayer, RandomReluLayer, ExponentialLinearUnitLayer
import sys
import time


input_dim, output_dim, hidden_dim = 784, 47, 128
numHiddenLayers = 3
batch_size = 100

def train_model_and_plot_stats(
		model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=False,
		early_stopping=None):
	# As well as monitoring the error over training also monitor classification
	# accuracy i.e. proportion of most-probable predicted classes being equal to targets
	data_monitors = {'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

	# Use the created objects to initialise a new Optimiser instance.
	optimiser = Optimiser(
		model, error, learning_rule, train_data, valid_data, data_monitors, notebook=notebook)

	# Run the optimiser for 5 epochs (full passes through the training set)
	# printing statistics every epoch.
	if early_stopping is not None:
		stats, keys, run_time = optimiser.train_with_early_stopping(num_epochs=num_epochs, stop_after=early_stopping,
		                                                            stats_interval=stats_interval)
	else:
		stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

	return optimiser, stats, keys, run_time


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


# I don't need original model, but it's good for sanity checking that the code is doing what I think it should be
def createModelWithBestParameterSettings(original_model, best_model, final_parameters, best_parameters, verbose=False,
                                         model_type=None):
	model_params = [layer.params for layer in original_model.layers if
	                isinstance(layer, LayerWithParameters) or isinstance(layer, StochasticLayerWithParameters)]
	model_params_flat = [item for sublist in model_params for item in sublist]

	if verbose:
		print("Model parameters equal to final model params: {}".format(final_parameters == original_model.params))
		print("Model parameters equal to best model params: {}".format(
			bool(np.prod([(a == b).prod() for (a, b) in zip(best_parameters, original_model.params)]))))
		print("Final parameters equal to best model params: {}".format(
			bool(np.prod([(a == b).prod() for (a, b) in zip(best_parameters, final_parameters)]))))

	if model_type == 'PReLU':
		is_affine_layer = True
		layers_processed = 0
		reshaped_best_params = []
		while layers_processed < len(best_parameters):
			if is_affine_layer:
				# take two
				reshaped_best_params.append([best_parameters[layers_processed], best_parameters[layers_processed + 1]])
				layers_processed += 2
				is_affine_layer = False
			else:
				reshaped_best_params.append([best_parameters[layers_processed]])
				layers_processed += 1
				is_affine_layer = True
	else:
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


def calculateAccuracy(model, data, data_type, verbose=False):
	accs = []
	for inputs_batch, targets_batch in data:
		activations = model.fprop(inputs_batch)
		accs.append(acc(activations[-1], targets_batch))

	if verbose:
		print("{} set accuracy on best model: {}".format(data_type, np.mean(accs)))

	return np.mean(accs)


def createBestModelAndCalculateAccuracy(original_model, best_model, final_parameters, best_parameters, data, data_type,
                                        model_type=None):
	# set best model parameters
	createModelWithBestParameterSettings(original_model, best_model, final_parameters, best_parameters,
	                                     model_type=model_type)

	# prepare data
	data.reset()
	data.batch_size = batch_size

	# calculate accuracy
	return calculateAccuracy(best_model, data, data_type, verbose=True)


def createMultiLayerModel(numHiddenLayers, hiddenLayerType, input_dim, hidden_dim, output_dim, rng):
    weights_init = GlorotUniformInit(rng=rng)
    biases_init = ConstantInit(0.)

    model_layers = [
        AffineLayer(input_dim, hidden_dim, weights_init, biases_init)
    ]

    for i in range(numHiddenLayers):
        if hiddenLayerType is not None:
            model_layers.append(hiddenLayerType)

        if i == numHiddenLayers - 1:
            # final numHiddenLayers
            model_layers.append(AffineLayer(hidden_dim, output_dim, weights_init, biases_init))
        else:
            model_layers.append(AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init))

    return MultipleLayerModel(model_layers)


# code common to all models
def trainModel(model,
               train_data,
               valid_data,
               learning_rate=0.01,
               num_epochs=100,
               stats_interval=1,
               batch_size=100,
               hidden_dim=128,
               early_stopping=None):
	input_dim, output_dim = 784, 47

	# reset data providers
	train_data.reset()
	valid_data.reset()

	train_data.batch_size = batch_size
	valid_data.batch_size = batch_size

	# set error function and learning rule
	error = CrossEntropySoftmaxError()
	learning_rule = AdamLearningRule()

	# train model
	model_stats = train_model_and_plot_stats(
		model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=False,
		early_stopping=early_stopping
	)

	return model_stats


def createMultiLayerLReLUModel(numHiddenLayers, alpha, input_dim, hidden_dim, output_dim, batch_size, rng):
	# initialise hyperparameters
	leakiness = np.ones((batch_size, hidden_dim)) * alpha

	hiddenLayerType = LeakyReluLayer(alpha=leakiness)

	return createMultiLayerModel(numHiddenLayers, hiddenLayerType, input_dim, hidden_dim, output_dim, rng)


def gatherLRELUSamples(numSamples, alpha, num_epochs, train_data, valid_data, rng, test_data=None, early_stopping=None):
	lrelu_valid_accs = []

	if test_data is not None:
		lrelu_test_accs = []

	for i in range(0, numSamples):
		print("*********** Sample {} ***********".format(i + 1))
		# reset rng
		rng.seed(i)

		# create model (3 layers)
		lrelu_model = createMultiLayerLReLUModel(numHiddenLayers, alpha, input_dim, hidden_dim, output_dim, batch_size,
		                                         rng)

		leaky_relu_stats = trainModel(lrelu_model, train_data, valid_data,
		                              learning_rate=0.01,
		                              num_epochs=num_epochs,
		                              stats_interval=1,
		                              batch_size=100,
		                              hidden_dim=128,
		                              early_stopping=early_stopping)

		# create empty "best model"
		best_lrelu_model = createMultiLayerLReLUModel(numHiddenLayers, alpha, input_dim, hidden_dim, output_dim,
		                                              batch_size, rng)

		# report best model validation set accuracy
		lrelu_valid_acc = createBestModelAndCalculateAccuracy(leaky_relu_stats[0].model, best_lrelu_model,
		                                                      leaky_relu_stats[0].model.params,
		                                                      leaky_relu_stats[0].best_model_params, valid_data,
		                                                      'Validation')

		lrelu_valid_accs.append(lrelu_valid_acc)

		if test_data is not None:
			lrelu_test_acc = createBestModelAndCalculateAccuracy(leaky_relu_stats[0].model, best_lrelu_model,
			                                                     leaky_relu_stats[0].model.params,
			                                                     leaky_relu_stats[0].best_model_params, test_data,
			                                                     'Test')
			lrelu_test_accs.append(lrelu_test_acc)

	mean = np.array(lrelu_valid_accs).mean()
	sd = np.array(lrelu_valid_accs).std()

	if test_data is not None:
		mean_test = np.array(lrelu_test_accs).mean()
		sd_test = np.array(lrelu_test_accs).std()

		return mean, sd, mean_test, sd_test

	return mean, sd



def createMultiLayerPReLUModel(numHiddenLayers, alpha_init, input_dim, hidden_dim, output_dim, batch_size, rng):
	# initialise hyperparameters
	alpha = np.ones((batch_size, hidden_dim)) * alpha_init

	hiddenLayerType = ParametricReluLayer(alpha)

	return createMultiLayerModel(numHiddenLayers, hiddenLayerType, input_dim, hidden_dim, output_dim, rng)


def gatherPRELUSamples(numSamples, alpha_init, num_epochs, train_data, valid_data, rng, test_data=None, early_stopping=None):
	prelu_valid_accs = []

	if test_data is not None:
		prelu_test_accs = []

	for i in range(0, numSamples):
		print("*********** Sample {} ***********".format(i + 1))
		# reset rng
		rng.seed(i)

		# create model (3 layers)
		prelu_model = createMultiLayerPReLUModel(numHiddenLayers, alpha_init, input_dim, hidden_dim, output_dim,
		                                         batch_size, rng)

		prelu_stats = trainModel(prelu_model, train_data, valid_data,
		                         learning_rate=0.01,
		                         num_epochs=num_epochs,
		                         stats_interval=1,
		                         batch_size=100,
		                         hidden_dim=128, early_stopping=early_stopping)

		# create empty "best model"
		best_prelu_model = createMultiLayerPReLUModel(numHiddenLayers, alpha_init, input_dim, hidden_dim, output_dim,
		                                              batch_size, rng)

		# report best model validation set accuracy
		prelu_valid_acc = createBestModelAndCalculateAccuracy(prelu_stats[0].model, best_prelu_model,
		                                                      prelu_stats[0].model.params,
		                                                      prelu_stats[0].best_model_params, valid_data,
		                                                      'Validation', 'PReLU')

		prelu_valid_accs.append(prelu_valid_acc)

		if test_data is not None:
			prelu_test_acc = createBestModelAndCalculateAccuracy(prelu_stats[0].model, best_prelu_model,
			                                                     prelu_stats[0].model.params,
			                                                     prelu_stats[0].best_model_params, test_data, 'Test', 'PReLU')
			prelu_test_accs.append(prelu_test_acc)

	mean = np.array(prelu_valid_accs).mean()
	sd = np.array(prelu_valid_accs).std()

	if test_data is not None:
		mean_test = np.array(prelu_test_accs).mean()
		sd_test = np.array(prelu_test_accs).std()

		return mean, sd, mean_test, sd_test

	return mean, sd


def createMultiLayerRRELUModel(numHiddenLayers, lower, upper, layer_rng, input_dim, hidden_dim, output_dim, rng):
	# initialise hyperparameters
	hiddenLayerType = RandomReluLayer(rng=layer_rng)

	return createMultiLayerModel(numHiddenLayers, hiddenLayerType, input_dim, hidden_dim, output_dim, rng)


def gatherRRELUSamples(numSamples, layer_rng, num_epochs, train_data, valid_data, rng, lower=0.125, upper=0.3333333333333333, test_data=None,
                       early_stopping=None):
	rrelu_valid_accs = []

	if test_data is not None:
		rrelu_test_accs = []

	for i in range(0, numSamples):
		print("*********** Sample {} ***********".format(i + 1))
		# reset rng
		rng.seed(i)

		# create model (3 layers)
		rrelu_model = createMultiLayerRRELUModel(numHiddenLayers, lower, upper, layer_rng, input_dim, hidden_dim,
		                                         output_dim, rng)

		rrelu_stats = trainModel(rrelu_model, train_data, valid_data,
		                         learning_rate=0.01,
		                         num_epochs=num_epochs,
		                         stats_interval=1,
		                         batch_size=100,
		                         hidden_dim=128, early_stopping=early_stopping)

		# create empty "best model"
		best_rrelu_model = createMultiLayerRRELUModel(numHiddenLayers, lower, upper, layer_rng, input_dim, hidden_dim,
		                                              output_dim, rng)

		# report best model validation set accuracy
		rrelu_valid_acc = createBestModelAndCalculateAccuracy(rrelu_stats[0].model, best_rrelu_model,
		                                                      rrelu_stats[0].model.params,
		                                                      rrelu_stats[0].best_model_params, valid_data,
		                                                      'Validation')

		rrelu_valid_accs.append(rrelu_valid_acc)

		if test_data is not None:
			rrelu_test_acc = createBestModelAndCalculateAccuracy(rrelu_stats[0].model, best_rrelu_model,
			                                                     rrelu_stats[0].model.params,
			                                                     rrelu_stats[0].best_model_params, test_data, 'Test')
			rrelu_test_accs.append(rrelu_test_acc)

	mean = np.array(rrelu_valid_accs).mean()
	sd = np.array(rrelu_valid_accs).std()

	if test_data is not None:
		mean_test = np.array(rrelu_test_accs).mean()
		sd_test = np.array(rrelu_test_accs).std()

		return mean, sd, mean_test, sd_test

	return mean, sd


from mlp.layers import ExponentialLinearUnitLayer


def createMultiLayerELUModel(numHiddenLayers, alpha_init, input_dim, hidden_dim, output_dim, rng):
	# initialise hyperparameters
	alpha = np.ones((batch_size, hidden_dim)) * alpha_init

	hiddenLayerType = ExponentialLinearUnitLayer(alpha)

	return createMultiLayerModel(numHiddenLayers, hiddenLayerType, input_dim, hidden_dim, output_dim, rng)


def gatherELUSamples(numSamples, alpha, num_epochs, train_data, valid_data, rng, test_data=None, early_stopping=None):
	elu_valid_accs = []

	if test_data is not None:
		elu_test_accs = []

	for i in range(0, numSamples):
		print("*********** Sample {} ***********".format(i + 1))
		# reset rng
		rng.seed(i)

		# create model (3 layers)
		elu_model = createMultiLayerELUModel(numHiddenLayers, alpha, input_dim, hidden_dim, output_dim, rng)

		elu_stats = trainModel(elu_model, train_data, valid_data,
		                       learning_rate=0.01,
		                       num_epochs=num_epochs,
		                       stats_interval=1,
		                       batch_size=100,
		                       hidden_dim=128, early_stopping=early_stopping)

		# create empty "best model"
		best_elu_model = createMultiLayerELUModel(numHiddenLayers, alpha, input_dim, hidden_dim, output_dim, rng)

		# report best model validation set accuracy
		elu_valid_acc = createBestModelAndCalculateAccuracy(elu_stats[0].model, best_elu_model,
		                                                    elu_stats[0].model.params, elu_stats[0].best_model_params,
		                                                    valid_data, 'Validation')

		if test_data is not None:
			elu_test_acc = createBestModelAndCalculateAccuracy(elu_stats[0].model, best_elu_model,
			                                                   elu_stats[0].model.params,
			                                                   elu_stats[0].best_model_params, test_data, 'Test')
			elu_test_accs.append(elu_test_acc)

		elu_valid_accs.append(elu_valid_acc)

	mean = np.array(elu_valid_accs).mean()
	sd = np.array(elu_valid_accs).std()

	if test_data is not None:
		mean_test = np.array(elu_test_accs).mean()
		sd_test = np.array(elu_test_accs).std()

		return mean, sd, mean_test, sd_test

	return mean, sd


def createMultiLayerAffineModel(numHiddenLayers, input_dim, hidden_dim, output_dim, rng):
	return createMultiLayerModel(numHiddenLayers + 1, None, input_dim, hidden_dim, output_dim, rng)


def gatherAffineSamples(numSamples, num_epochs, train_data, valid_data, rng, test_data=None, early_stopping=None):
	valid_accs = []

	if test_data is not None:
		test_accs = []

	for i in range(0, numSamples):
		print("*********** Sample {} ***********".format(i + 1))
		# reset rng
		rng.seed(i)

		# create model (3 layers)
		model = createMultiLayerAffineModel(numHiddenLayers, input_dim, hidden_dim, output_dim, rng)

		stats = trainModel(model, train_data, valid_data,
		                   learning_rate=0.01,
		                   num_epochs=num_epochs,
		                   stats_interval=1,
		                   batch_size=batch_size,
		                   hidden_dim=hidden_dim, early_stopping=early_stopping)

		# create empty "best model"
		best_model = createMultiLayerAffineModel(numHiddenLayers, input_dim, hidden_dim, output_dim, rng)

		# report best model validation set accuracy
		valid_acc = createBestModelAndCalculateAccuracy(stats[0].model, best_model, stats[0].model.params,
		                                                stats[0].best_model_params, valid_data, 'Validation')

		if test_data is not None:
			test_acc = createBestModelAndCalculateAccuracy(stats[0].model, best_model, stats[0].model.params,
			                                               stats[0].best_model_params, test_data, 'Test')
			test_accs.append(test_acc)

		valid_accs.append(valid_acc)

	mean = np.array(valid_accs).mean()
	sd = np.array(valid_accs).std()

	if test_data is not None:
		mean_test = np.array(test_accs).mean()
		sd_test = np.array(test_accs).std()

		return mean, sd, mean_test, sd_test

	return mean, sd


def main(model_to_run, num_samples, num_epochs, early_stopping, alpha, all_data):
	# The below code will set up the data providers, random number
	# generator and logger objects needed for training runs. As
	# loading the data from file take a little while you generally
	# will probably not want to reload the data providers on
	# every training run. If you wish to reset their state you
	# should instead use the .reset() method of the data providers.

	# Seed a random number generator
	seed = 11102019
	rng = np.random.RandomState(seed)
	batch_size = 100
	# Set up a logger object to print info about the training run to stdout
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	logger.handlers = [logging.StreamHandler()]

	# Create data provider objects for the MNIST data set
	train_data = EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
	valid_data = EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)
	test_data = EMNISTDataProvider('test', batch_size=batch_size, rng=rng)

	# changing batch size (and number of batches for smaller sets)
	max_num_batches_train = 100
	max_num_batches_valid = 10

	# defaults: batch_size=100, max_num_batches=-1
	train_data_100 = EMNISTDataProvider('train', batch_size=batch_size, max_num_batches=max_num_batches_train, rng=rng)
	valid_data_10 = EMNISTDataProvider('valid', batch_size=batch_size, max_num_batches=max_num_batches_valid, rng=rng)

	if model_to_run == 'lrelu':
		accs = gatherLRELUSamples(num_samples, 0.1, num_epochs, train_data, valid_data, rng, test_data, early_stopping)
	elif model_to_run == 'prelu':
		accs = gatherPRELUSamples(num_samples, 0.1, num_epochs, train_data, valid_data, rng, test_data, early_stopping)
	elif model_to_run == 'rrelu':
		layer_rng = np.random.RandomState(10)
		accs = gatherRRELUSamples(num_samples, layer_rng, num_epochs, train_data, valid_data, rng, test_data=test_data, early_stopping=early_stopping)
	elif model_to_run == 'elu':
		if all_data:
			accs = gatherELUSamples(num_samples, alpha, num_epochs, train_data, valid_data, rng, test_data=test_data, early_stopping=early_stopping)
		else:
			accs = gatherELUSamples(num_samples, alpha, num_epochs, train_data_100, valid_data_10, rng, test_data=test_data, early_stopping=early_stopping)
	elif model_to_run == 'affine':
		accs = gatherAffineSamples(num_samples, num_epochs, train_data, valid_data, rng, test_data=test_data, early_stopping=early_stopping)
	else:
		print("ERROR: MODEL NOT FOUND")

	print()
	print("Validation accuracy of best model: {} (± {})".format(accs[0], accs[1]))
	print("Validation accuracy of best model: {} (± {})".format(accs[2], accs[3]))


if __name__ == "__main__":
	model_to_run = sys.argv[1]
	num_samples = int(sys.argv[2])
	num_epochs = int(sys.argv[3])
	early_stopping = int(sys.argv[4])
	alpha = float(sys.argv[5])
	all_data = bool(sys.argv[6])

	file_name = '{}_{}_{}_{}_{}'.format(model_to_run, num_samples, num_epochs, early_stopping, alpha)

	sys.stderr.write('Training begun...\n')

	# redirect stdout
	# sys.stdout = open(file_name, 'w+')

	print(time.time())
	main(model_to_run, num_samples, num_epochs, early_stopping, alpha, all_data)
	print(time.time())

	sys.stderr.write('Training complete! See {} for output.'.format(file_name))

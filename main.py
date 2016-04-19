import six.moves.cPickle as pickle
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wave
from lstmp import LSTMP
from fuel.datasets.youtube_audio import YouTubeAudio
from fuel.transformers.sequences import Window

# hyper parameters
freq = 16000
hiddenUnits = 512
learningRate = 0.002 # learning rate
length = 32000   # 2 seconds
features = 120 # RNN input feature size

# create LSTM
print("Creating 2-Layer LSTM...")
lstm = LSTMP(features, hiddenUnits, 1)

# extra parameters
trainingSize = length*features # training length
delay = length                 # target delay
stride = 8000                  # sequence stream stride
iterations = 128
gIterations = 128
t_start = 60
g_start = 930

# switch to configure training or audio generation
# 0: generate only; 1: train only; 2: train & generate
training = 0

if training > 0:
	idx = 0
	error = np.array([0])
	minError = np.inf
	vals = []
	
	# retrive datastream
	print("retrieving data...")
	data = YouTubeAudio('XqaJ2Ol5cC4')
	stream = data.get_example_stream()
	data_stream = Window(delay, trainingSize, trainingSize, stride, True, stream)

	print("hidden units:", hiddenUnits)
	print("learning rate:", learningRate)
	print("training size:", trainingSize/freq,'seconds')
	print("iterations: ", iterations)
	
	print("training begin...")
	#Init params crude		
	c0 = np.random.uniform(low=-0.2, high=0.2, size=(512,))
	r0 = np.random.uniform(low=-0.2, high=0.2, size=(256,))
	for batch_stream in data_stream.get_epoch_iterator():
		idx += 1		
		[u, t] = np.array(batch_stream, dtype=theano.config.floatX) # get samples
		if idx > t_start: 
		# start after t seconds
			# do some reshaping magic
			uBatch = np.reshape((u/0x8000), (features,length)).swapaxes(0,1)
			tBatch = (t[-length:]/0x8000)
			# train and find error
			print("\ntraining...")
			[r0, c0, output, error]  = lstm.train(r0, c0, uBatch, tBatch, learningRate)
			# feedback r0, c0 from previous r_last, c_last
			# forcing initial r0, c0 to "match" previous training LSTM connections
			#lstm.r0 = r0#theano.shared(np.array(r))
			#lstm.c0 = c0#theano.shared(np.array(c))
			vals.append(error)
			print ("Cost:", error, "at iteration:",idx-t_start)	
			print(output.flatten())		
			if error<minError:
				minError = error
				print("LOWEST ERROR:", minError)
				# SAVE MODEL
				f = open('best_model.pkl', 'wb')
				pickle.dump(lstm.params, f)
				f.close()	
			if idx % 5 == 0:
				plt.plot(vals)
				plt.savefig('loss.png')	
				plt.clf()
				f = open('5_model.pkl', 'wb')
				pickle.dump(lstm.params, f)
				f.close()
		# End somewhere
		if idx>(t_start+iterations): break
	print("Total sequence trained:", (idx-t_start)*(stride/freq), "seconds")
	# PLOTTING LOSS
	plt.plot(vals)
	plt.savefig('loss.png')			
	f = open('loss.pkl', 'wb')
	pickle.dump(vals, f)
	f.close()
if training != 1:
	f = open('best_model.pkl', 'rb') 
	# load params from file
	lstm.params = pickle.load(f)
	f.close()
	start = 0
	idx = 0
	vals=[]

	# retrive datastream
	print("retrieving data...")
	data = YouTubeAudio('XqaJ2Ol5cC4')
	stream = data.get_example_stream()
	data_stream = Window(delay, trainingSize, trainingSize, stride, True, stream)
	
	print("iterations: ", gIterations)
	
	print("generation begin...")
	#Init params crude		
	c0 = np.random.uniform(low=-0.25, high=0.25, size=(512,))
	r0 = np.random.uniform(low=-0.25, high=0.25, size=(256,))
	for batch_stream in data_stream.get_epoch_iterator():
		idx += 1
		if idx > g_start:
			if start != 1:
				# get one batch as seed sequence
				[u, t] = np.array(batch_stream, dtype=theano.config.floatX) # get samples		
				start = 1
				
			# do some reshaping magic
			uBatch = np.reshape((u/0x8000), (features,length)).swapaxes(0,1)
			
			# generate 1 new batch of data
			[r0, c0, prediction] = lstm.predict(r0, c0, uBatch)
			# feedback r0, c0 from previous r_last, c_last
			# forcing initial r0, c0 to "match" previous training LSTM connections
			#lstm.r0 = theano.shared(np.array(r))
			#lstm.c0 = theano.shared(np.array(c))
			##print(r.shape, c.shape)
			print(prediction.flatten(), prediction.shape)
			for item in prediction.flatten():
				vals.append(np.asscalar(item))
				
			# update u by removing a block from u and appending prediction
			u = np.append(u[length:], prediction, axis=0)
			print ("Iteration:", idx-g_start)	
			
		# End somewhere
		if idx>(g_start+gIterations): break
		
	print("Total sequence size generated:", (idx-g_start)*(delay/freq), "seconds")
	
	f = open('audio.pkl', 'wb')
	pickle.dump(vals, f)
	f.close()
	plt.clf()
	plt.plot(vals)
	plt.savefig('audio.png')
	wave.write('audio.wav', freq, np.asarray(vals, dtype=theano.config.floatX))
			

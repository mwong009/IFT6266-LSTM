import six.moves.cPickle as pickle
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wave
from lstmor import LSTMOR
from lstmp import LSTMP
from fuel.datasets.youtube_audio import YouTubeAudio
from fuel.transformers.sequences import Window

# hyper parameters
freq = 16000
hiddenUnits = 300
batchSize = 40000   # no. of timesteps
stride = batchSize  # window stride
miniBatches = 40  # batches per iteration
sequenceSize = batchSize*miniBatches
learningRate = 0.002 # learning rate
vals = []
minError = np.inf
idx = 0
iterations = 256
gIterations = 64
tStartTime = 30
gStartTime = 1200

# create LSTM
print("Creating 2-Layer LSTMOR...")
lstm = LSTMP(miniBatches, hiddenUnits, miniBatches)

# switch this to configure training or audio generation
# 0: generate only; 1: train only; 2: train & generate
training = 0

if training > 0:
	# retrive datastream
	print("retrieving data...")
	data = YouTubeAudio('XqaJ2Ol5cC4')
	stream = data.get_example_stream()
	data_stream = Window(stride, sequenceSize, sequenceSize, True, stream)
	
	print("Input Size:", batchSize)
	print("minibatches:", miniBatches) 
	print("stride:", stride)
	print("hidden units:", hiddenUnits)
	print("learning rate:", learningRate)
	print("sequence size:", sequenceSize, '(',sequenceSize/freq,'s)' )
	print("iterations:", iterations, ", training begin...")
	for batch_stream in data_stream.get_epoch_iterator():
		# Start somewhere (after 1 minute)
		if idx>=(tStartTime*freq):
			[u, t] = np.array(batch_stream, dtype=theano.config.floatX)
			# do some reshaping magic
			[uBatch, tBatch] = np.reshape([(u/0x8000), (t/0x8000)], (2,miniBatches,batchSize)).swapaxes(1,2)
			# train and find error
			print("\ntraining...")
			error  = lstm.train(uBatch, tBatch, learningRate)
			vals.append(np.asarray(error))
			print ("Cost:", error, "at iteration:",idx-(tStartTime*freq))			
			if error<minError:
				minError = error
				print("LOWEST ERROR:", minError)
				# LOWEST ERROR MODEL
				f = open('LSTM_MODEL_SIZE'+str(batchSize)+'_'+str(sequenceSize/freq)+'s_LR'
	                     +str(learningRate)+'_HU'+str(hiddenUnits)+'.pkl', 'wb')
				pickle.dump(lstm.params, f)
				f.close()			
		# End somewhere
		if idx>=(tStartTime*freq+iterations): break # iterations
		idx = idx + 1
	print("Total sequence trained:", (idx-(tStartTime*freq))*(stride/freq), "seconds")
	# saving and printing
	plt.plot(vals)
	plt.savefig('LSTM_LOSS_PLOT_SIZE'+str(batchSize)+'_'+str(sequenceSize/freq)+'s_LR'
	            +str(learningRate)+'_HU'+str(hiddenUnits)+'.png')			
	f = open('LSTM_LOSS_PLOT_SIZE'+str(batchSize)+'_'+str(sequenceSize/freq)+'s_LR'
	            +str(learningRate)+'_HU'+str(hiddenUnits)+'_vals.pkl', 'wb')
	pickle.dump(vals, f)
	f.close()

if training != 1:
	# load parameters
	f = open('LSTM_MODEL_SIZE'+str(batchSize)+'_'+str(sequenceSize/freq)+'s_LR'
	         +str(learningRate)+'_HU'+str(hiddenUnits)+'.pkl', 'rb') 
	lstm.params = pickle.load(f) #load params from file
	f.close()
	start = 0
	idx = 1
	vals=[]
	# retrive datastream
	print("retrieving data...")
	data = YouTubeAudio('XqaJ2Ol5cC4')
	stream = data.get_example_stream()
	data_stream = Window(stride, sequenceSize, sequenceSize, True, stream)
	print("Input Size:", batchSize)
	print("minibatches:", miniBatches) 
	print("Sequence length:", sequenceSize/freq, "s")
	print("stride:", stride)
	print("hidden units:", hiddenUnits)
	print("learning rate:", learningRate)
	print("iterations:", gIterations, ", generation begin...")
	for batch_stream in data_stream.get_epoch_iterator():
		# Start somewhere
		if idx>=(gStartTime*freq):
			if start == 0:
				# get N batch as seed sequence
				u, t = batch_stream
				u = np.array(u/(0x8000), dtype=theano.config.floatX)		
				start = 1
			# do some reshaping magic
			uBatch = np.reshape(u, (miniBatches, batchSize)).swapaxes(0,1)		
			# generate N batch of data
			prediction = np.reshape(lstm.predict(uBatch).swapaxes(0,1), u.shape)
			# drop N-1 batches, keep batch N
			prediction = prediction[(sequenceSize-stride):]
			print(prediction, prediction.shape)
			for item in prediction:
				vals.append(np.asscalar(item))
			# update u by removing a block from u and appending a block from prediction
			u = np.append(u[stride:], prediction, axis=0)
			print ("Iteration:", idx-(gStartTime*freq))	
			
		# End somewhere
		if idx>=(gStartTime*freq+gIterations):break # iterations
		idx = idx + 1
		
	print("Total sequence size generated:", (idx-(gStartTime*freq))*(stride/freq), "seconds")
	
	f = open('LSTM_GEN_AUDIO'+str(batchSize)+'_'+str(sequenceSize/freq)+'s_LR'
	         +str(learningRate)+'_HU'+str(hiddenUnits)+'.pkl', 'wb')
	pickle.dump(vals, f)
	f.close()
	plt.clf()
	plt.plot(vals)
	plt.savefig('LSTM_GEN_PLOT'+str(batchSize)+'_'+str(sequenceSize/freq)+'s_LR'
	         +str(learningRate)+'_HU'+str(hiddenUnits)+'.png')
	wave.write('LSTM_GEN_AUDIO'+str(batchSize)+'_'+str(sequenceSize/freq)+'s_LR'
	         +str(learningRate)+'_HU'+str(hiddenUnits)+'.wav', 
	         freq, np.asarray(vals, dtype=theano.config.floatX))
			

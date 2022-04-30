
import pyomnidata
pyomnidata.UpdateLocalData()

from sys import stdout, maxsize
import torch
import numpy as np
from numpy.lib import recfunctions as rfn
import matplotlib.pyplot as plt # This will probably be removed, as I would like to see these plots later.
import sklearn.metrics as sm
from ignite.contrib.metrics.regression import MeanAbsoluteRelativeError as mare

import networks

import warnings
warnings.filterwarnings("error")


def getHighLowData(yearsLowList, yearsHighList, inputNumbers, targetNumbers, forecastOffset, datapointOffset, inputHours, **datakwargs):
	'''
	a wrapper for getData that returns data from the high part of a solar cycle and the low part of the solar cycle.
	'''
	lowInputs = torch.zeros((0, inputHours, len(inputNumbers)))
	lowTargets = np.zeros((0, len(targetNumbers)))


	for years in yearsLowList:
		newIns, newTargs = detData(years, inputNumbers, targetNumbers, forecastOffset, datapointOffset, inputHours, 'cuda')
		lowInputs = torch.cat((lowInputs, newIns))
		lowTargets = np.concatenate((lowTargets, newTargs))


	highInputs = torch.zeros((0, inputHours, len(inputNumbers)))
	highTargets = np.zeros((0, len(targetNumbers)))


	for years in yearsHighList:
		newIns, newTargs = detData(years, inputNumbers, targetNumbers, forecastOffset, datapointOffset, inputHours, 'cuda')
		highInputs = torch.cat((highInputs, newIns))
		highTargets = np.concatenate((highTargets, newTargs))

	return highInputs, highTargets, lowInputs, lowTargets

class normStruct:
	'''
	This class is used in physics based loss to un-normalize the data to recover the E < V x B relationship.
	The default values correspoond to no normalization, and are updated upon normalization. 
	'''
	# These are gonna need better neames, but they'll have to do for now
	def __init__(self, inputs=0, device = 'cuda'):
		if inputs == 0:
			self.lr = 0 # shift before multiplying
			self.mul = 1 # multiplicative factor
			self.ud = 0 # shift after multiplying
		else:
			self.lr = torch.zeros((inputs), device=device)
			self.mul = torch.ones((inputs), device=device)
			self.ud = torch.zeros((inputs), device=device)

def getData(years, inputNumbers, targetNumbers, forecastOffset, datapointOffset, inputHours, device, norm='none', frames=None):
	omnidata = pyomnidata.GetOMNI(years)
	numHours = int(omnidata[-1][2]) - int(omnidata[0][2]) # data[i][2] is UTC. This operation lets us get how many days are in our datatset.
	history = torch.zeros(numHours, len(inputNumbers), device=device) # We want each hour as a datapoint, with an average of the points inside
	hour = 0
	dataIterator = 0
	targets = []
	lastBad = 0
	badDict = {}
	isNaN = []
	norms = normStruct(inputs=len(inputNumbers), device = device)

	for i in range(numHours):
		hourList = [] # define a list to hold the data before averaging
		trgList = []
		while int(omnidata[dataIterator][1]) == hour and dataIterator < len(omnidata): # get all the data in this hour
			inputList = []
			hourTrgs = []
			for j in inputNumbers: # I couldn't figure out how to access multiple entries at the same time, so I appended them in a list
				inputList.append(omnidata[dataIterator][j]) # and put that list as an element of hourList
			hourList.append(inputList) # put the desired data in the holding list
			for j in targetNumbers:
				hourTrgs.append(omnidata[dataIterator][j])
			trgList.append(hourTrgs)
			dataIterator += 1 # move to the next 5 minute increment

		# If there are NANs in the data, nanmean will throw an error. Catch this and put the NANs in history to be dealt with later.
		try:
			hourInfo = np.nanmean(hourList, axis=0)
			targets.append(np.nanmean(trgList, axis=0))
			for j in range(len(inputNumbers)):
				history[i,j] = float(hourInfo[j]) # assign our new averaged data to a history tensor. Probably should figure out why 'float' is necessary, but that's a problem for future rob.	
			isNaN.append(False)
		except RuntimeWarning:
			history[i,:] = float('NaN')
	#		 if i <= len(targets): This line might be useful later. Depends on how hte error checking works.
			targets.append([float('nan') for _ in targetNumbers])
			isNaN.append(True)
		hour = (hour+1)%24 # move to the next hour, resetting if we go over a day

	if frames: 
		history = history[frames[0]:frames[1]] # frames allows us to only take part of the data
		isNaN = [False for _ in range(history.shape[0])]

	windowLength = forecastOffset + inputHours
	isValidStart =  []
	# Check for any NANs - those screw with the neural nets.
	for i in range(len(isNaN) - windowLength):
		#		 Check our inputs for NaNs			 Check our targets for NaNs
		isValidStart.append(not any(isNaN[i:i+inputHours]) and not np.isnan(targets[i+windowLength][:]).any())

# The next nine lines could use a refactor
	# make sure there is an offset between valid data start points
	for i in range(len(isValidStart)):
		prev = min(i, datapointOffset) # this line prevents us checking outside the bounds of our array
		if any(isValidStart[i-prev:i]): # check back far enough to know if this point is in the offset for another point
			isValidStart[i] = False
	# in hindsight I could have done this in the previous for loop, but too late now.
	starts = []
	for i in range(len(isValidStart)):
		if isValidStart[i]:
			starts.append(i)


	# The dataset we want to regress on should take every valid starting number, then be the correct length and width.
	data = torch.zeros(len(starts), inputHours, len(inputNumbers), device=device) 
	ys = np.zeros((len(starts), len(targetNumbers)), dtype=float)
	for i in range(len(starts)):
		data[i,:,:] = history[starts[i]:starts[i] + inputHours, :] # honestly, I don't know what happened here.
		ys[i,:] = targets[starts[i] + windowLength]

	ys = ys.astype('float32') # this shouldn't need to be here, but here we are
	# perform normalization
	if norm.__class__.__name__ == 'normStruct': # this indicates that a model has already been trained with a given normalization, and we should reuse that.
		norms = norm
		for i in range(len(inputNumbers)):
			data[:,:,i] -= norm.ud[i]
			data[:,:,i] /= norm.mul[i]
			data[:,:,i] += norm.lr[i]
		for i in range(len(targetNumbers)):
			ys[:,i] = (ys[:,i] - norm.ud[i].cpu().detach().numpy())/norm.mul[i].detach().cpu().numpy() + norm.lr[i].cpu().detach().numpy()
	elif norm=='0mean':
		for i in range(len(inputNumbers)):
			norms.ud[i]  = torch.mean(data[:,:,i])
			data[:,:,i] -= norms.ud[i]
			norms.mul[i]  = torch.std(data[:,:,i])
			data[:,:,i] /= norms.mul[i]
		for i in range(len(targetNumbers)):
			ys[:,i] = (ys[:,i] - norms.ud[i].cpu().detach().numpy())/norms.mul[i].detach().cpu().numpy()
	elif '-' in norm:
		low, high = norm.split('-')
		if low == '0': low = 0.000000000000001 # We don't want divide by zero errors when we calculate PCE
		else: low = float(low)
		norms.lr[:] = low
		high = float(high)
		for i in range(len(inputNumbers)):
			norms.ud[i]  = torch.min(data[:,:,i])
			data[:,:,i] -= norms.ud[i]
			norms.mul[i]  = torch.max(data[:,:,i])/ (high - low)
			data[:,:,i] /= norms.mul[i]
			norms.lr[i] = low
			data[:,:,i] += low
		for i in range(len(targetNumbers)):
			ys[:,i] -= norms.ud[i].detach().cpu().numpy()
			ys[:,i] /= norms.mul[i].detach().cpu().numpy()
			ys[:,i] += norms.lr[i].detach().cpu().numpy()
	elif 'max' in norm:
		newmax = float(norm.strip('abcdefghijklmnopqrstuvwxyz,.:;'))
		for i in range(len(inputNumbers)):
			norms.mul[i] = torch.max(torch.abs(data[:,:,i]))/newmax
			data[:,:,i] /= norms.mul[i]
		for i in range(len(targetNumbers)):
			ys[:,i] /= norms.mul[i].detach().cpu().numpy()
	return data, ys, norms

def percentError(yhat, target):
	retval = []
	for i in range(len(target)):
		retval.append(torch.mean(abs((target[i]-yhat[i])/target[i])))
	return retval

def dictToList(grid):
	entries = 1
	for _, value in grid.items():
		entries *= max(len(value), 1)
	retval = [{} for _ in range(entries)]

	rotationFactor = 1
	for key, value in grid.items():
		for i in range(entries):
			retval[i][key] = value[int(i/rotationFactor) % len(value)]
		rotationFactor *= len(value)
	return retval


def trainer(model, data1, data2, val, test, opt, lfc, epochs, device, verboseEpochs=-1, returnModel = False, trainable=True):

	trainLoss = []
	trainPCE = []
	valLoss = []
	valPCE = []

	if trainable:
		for epoch in range(epochs):
			if verboseEpochs > 0 and (not (epoch+1) % verboseEpochs): verbose = True
			else: verbose = False
			model.train()
			batch_loss = []
			batch_PCE = []
	# iterate through both training sets
			for (xtrain, ytrain) in data1:
				xtrain = xtrain.to(device)
				ytrain = ytrain.to(device)
				opt.zero_grad()
				output = model(xtrain)
				output = output.to(device)
				loss = torch.zeros(output.shape, dtype=torch.float).to(device)
				loss = lfc(output, ytrain)
				loss = loss.type(torch.float)
				loss.backward()
				opt.step()
				batch_loss.append(loss)
				for pce in percentError(output, ytrain): batch_PCE.append(pce)

			for (xtrain, ytrain) in data2:
				xtrain = xtrain.to(device)
				ytrain = ytrain.to(device)
				opt.zero_grad()
				output = model(xtrain)
				output = output.to(device)
				loss = lfc(output, ytrain)
				loss.backward()
				opt.step()
				batch_loss.append(loss)
				for pce in percentError(output, ytrain): batch_PCE.append(pce)


			if verbose: print(f'The training loss for epoch  {epoch+1}/{epochs} was {torch.mean(torch.Tensor(batch_loss))}')
			trainLoss.append(torch.mean(torch.Tensor(batch_loss)).item())
			tnsr = torch.zeros((len(batch_PCE)))
			for i in range(len(batch_PCE)):
				entry = torch.mean(batch_PCE[i]).item()
				tnsr[i] = entry
			trainPCE.append(torch.mean(tnsr).item())

			batch_loss = []
			batch_PCE = []
			model.eval()
			hasVal = False
			for (xval, yval) in val:
				hasVal = True
				xval = xval.to(device)
				yval = yval.to(device)
				output = model(xval).to(device)
				loss = lfc(output, yval)
				batch_loss.append(loss.item())
				for pce in percentError(output, yval): batch_PCE.append(pce)
			if verbose and hasVal: print(f'The validation loss for epoch {epoch+1}/{epochs} was {torch.mean(torch.Tensor(batch_loss))}')
			valLoss.append(torch.mean(torch.Tensor(batch_loss)).item())
			tnsr = torch.zeros(len(batch_PCE))
			for i in range(len(batch_PCE)): tnsr[i] = torch.mean(batch_PCE[i]).item()
			valPCE.append(torch.mean(torch.Tensor(tnsr)).item())
		
	testLoss = []
	testPCE = []
	testR2 = []

	for (xtest, ytest) in test:
		# Model is still in eval mode from the end of the for loop, so we do not have to switch.
		output = model(xtest.to(device)).to(device)
		ytest = ytest.to(device)
		loss = lfc(output, ytest)
		testLoss.append(loss)
		for pce in percentError(output, ytest): testPCE.append(pce)
		testR2.append(sm.r2_score(ytest.detach().cpu(), output.detach().cpu()))
	testLoss = torch.mean(torch.Tensor(testLoss)).item()
	tnsr = torch.zeros((len(testPCE)))
	if testR2: r2 = np.mean(testR2)
	else: r2 = 0
	for i in range(len(testPCE)): tnsr[i] = torch.mean(testPCE[i]).item()
	print(f'The test loss of this iteration is {testLoss}. This is a {torch.mean(tnsr) * 100}% error.')
	if returnModel: 
		return trainLoss, trainPCE, testLoss, testPCE, model
	return trainLoss, trainPCE, valLoss, valPCE, testLoss, testPCE, r2


def gridSearch3CVMSE(x,y, net, epochs, batch, grid, device, *netargs, lfc = torch.nn.MSELoss(), lr=0.0001, file=None, verboseEpochs=-1, trainable=True, **netkwargs):
	numPoints = x.shape[0]
	ziplist = list(zip(x, y))
	j = int(numPoints*0.3)
	test = int(numPoints*0.1)

	train1, train2, train3, test = torch.utils.data.random_split(ziplist, [numPoints-2*j-test, j, j, test])

	train1 = torch.utils.data.DataLoader(train1, batch_size = batch)
	train2 = torch.utils.data.DataLoader(train2, batch_size = batch)
	train3 = torch.utils.data.DataLoader(train3, batch_size = batch)
	# does batch size matter for testing? - YES - BIG TIME - OTHERWISE R2 IS OFF
	test   = torch.utils.data.DataLoader(test,   batch_size = maxsize)

	maxR2 = -100000000
	maxR2Args = []
	minLoss = 10000000000 #float('inf')
	minLossArgs = []
	minPCE = 100000000000 #float('inf')
	minPCEArgs = []
	# Do each of the grid things
	for gridnetkwargs in dictToList(grid):
		if 'num_layers' in gridnetkwargs and gridnetkwargs['num_layers'] == 1 and 'gruDrop' in gridnetkwargs and gridnetkwargs['gruDrop'] != 0:
			print('Mismatch between num_layers and gruDrop. Skipping this round')
			continue
		print(f'args: {gridnetkwargs}', end='\n')
		testLossList = []
		testPCEList = []
		testR2 = []

		trainLoss = np.zeros((epochs, 3))
		valLoss = np.zeros((epochs, 3))
		trainPCE = np.zeros((epochs, 3))
		valPCE =  np.zeros((epochs, 3))


		# we train all three networks
		model1 = net(*netargs, **gridnetkwargs, **netkwargs)
		opt = torch.optim.Adam(model1.parameters(), lr=lr)
		model1 = model1.to(device)
		trainLoss[:,1], trainPCE[:,1], valLoss[:,1], valPCE[:,1], testLoss, testPCE, r2 = trainer(model1, train2, train3, train1, test, opt, lfc, epochs, device, verboseEpochs=verboseEpochs, trainable=trainable)
		del model1
		testLossList.append(testLoss)
		testPCEList.append(testPCE)
		testR2.append(r2)

		model2 = net(*netargs, **gridnetkwargs, **netkwargs)
		model2 = model2.to(device)
		opt = torch.optim.Adam(model2.parameters(), lr=lr)
		trainLoss[:,2], trainPCE[:,2], valLoss[:,2], valPCE[:,2], testLoss, testPCE, r2 = trainer(model2, train3, train1, train2, test, opt, lfc, epochs, device, verboseEpochs=verboseEpochs, trainable=trainable)
		del model2
		testLossList.append(testLoss)
		testPCEList.append(testPCE)
		testR2.append(r2)

		model3 = net(*netargs, **gridnetkwargs, **netkwargs)
		model3 = model3.to(device)
		opt = torch.optim.Adam(model3.parameters(), lr=lr)
		trainLoss[:,0], trainPCE[:,0], valLoss[:,0], valPCE[:,0], testLoss, testPCE, r2 = trainer(model3, train1, train2, train3, test, opt, lfc, epochs, device, verboseEpochs=verboseEpochs, trainable=trainable)
		del model3
		testLossList.append(testLoss)
		testPCEList.append(testPCE)
		testR2.append(r2)

		testLoss = np.mean(testLossList)
		testPCE = np.mean([t.detach().cpu() for t in testPCE])
		testR2 = np.mean(testR2)

		if testLoss < minLoss:
			minLoss = testLoss
			minLossArgs = gridnetkwargs
		if testPCE < minPCE:
			minPCE = testPCE
			minPCEArgs = gridnetkwargs
		if testR2 > maxR2:
			maxR2 = testR2
			maxR2Args = gridnetkwargs

		if file:
			printLocation = open(file, mode='a')
			print(f'training loss:', file=printLocation)
			for r in trainLoss: print(r, file=printLocation)
			print(f'training PCE:', file=printLocation)
			for r in trainPCE: print(r, file=printLocation)
			print(f'validation loss:', file=printLocation)
			for r in valLoss: print(r, file=printLocation)
			print(f'validation PCE:', file=printLocation)
			for r in valPCE: print(r, file=printLocation)
			print(f'test RMSE: {testLoss} test PCE: {testPCE}, test R2: {testR2}, args: {gridnetkwargs}', file=printLocation, flush=True)
			printLocation.close()
		else:
			print('===========================================================')
			print(f'training loss: {trainLoss}')
			print(f'training PCE: {trainPCE}')
			print(f'validation loss: {valLoss}')
			print(f'validation PCE: {valPCE}')
		print(f'test RMSE: {testLoss} test PCE: {testPCE}, testR2: {testR2}')


	return minLossArgs, minLoss, minPCEArgs, minPCE


# def trainer(model, data1, data2, val, test, opt, lfc, epochs, device, verboseEpochs=-1):
# def gridSearch3CVMSE(x,y, net, epochs, batch, grid, device, *netargs, lfc = torch.nn.MSELoss(), lr=0.0001, file=None, verboseEpochs=-1, **netkwargs):


def finalTraining(x,y, net, epochs, batch, device, filename, *netargs, lfc = torch.nn.MSELoss(), lr=0.0001, verboseEpochs=None, **netkwargs):
	print(filename)
	if verboseEpochs == None:
		verboseEpochs = int(epochs/5)+1

	numPoints = x.shape[0]
	ziplist = list(zip(x, y))
	test = int(numPoints*0.1)

	train, test = torch.utils.data.random_split(ziplist, [numPoints-test, test])

	train = torch.utils.data.DataLoader(train, batch_size = batch)
	# I don't know if batch size actually matters for testing (I'm a software guy, and this feels like a hardware question), but I want to print line by line later, so it's 1 here.
	test   = torch.utils.data.DataLoader(test,  batch_size = 1)


	model = net(*netargs, **netkwargs)
	opt = torch.optim.Adam(model.parameters(), lr=lr)
	model = model.to(device)

	# This next line is what we in the business like to call "refinancing your tech debt". Due to me assuming I would only use trainer for 3 fold validation, 
	# unwisely made it so that the data for each fold and the validation were a requirement. Since I would like to train on all but the test set for this final bit,
	# I now have to use empty lists to bypass the second training and validation loops. 
	trainLoss, trainPCE, testLoss, testPCE, model = trainer(model, train, [], [], test, opt, lfc, epochs, device, verboseEpochs=verboseEpochs, returnModel = True)
	# model should be passed by reference in the line above, right? 
	file = open(filename, mode='w')
	for (xtest, ytest) in test:
		xtest = xtest.to(device)
		output = model(xtest).to(device)
		ytest = ytest.to(device)
		print('Actual: ', ytest.tolist(), ' Predicted: ', output.tolist(), file=file)
	file.close()
	return None


def highLowTraining(highInputs, highTargets, lowInputs, lowTargets, net, epochs, batch, device, filename, *netargs, lfc = torch.nn.MSELoss(), lr = 0.0001, verboseEpochs=None, **netkwargs):
	if verboseEpochs == None:
		verboseEpochs = int(epochs/5)+1

	highzip = list(zip(highInputs,highTargets))
	high = torch.utils.data.DataLoader(highzip, batch_size=batch, shuffle=True)

	lowzip = list(zip(lowInputs, lowTargets))
	low = torch.utils.data.DataLoader(lowzip, batch_size=batch, shuffle=True)

	model = net(*netargs, **netkwargs)
	opt = torch.optim.Adam(model.parameters(), lr=lr)
	model = model.to(device)

	trainLoss, trainPCE, testLoss, testPCE = trainer(model, low, [], [], high, opt, lfc, epochs, device, verboseEpochs=verboseEpochs)
	# model should be passed by reference in the line above, right? 
	file = open('trainLowGuessHigh' + filename, mode='w')
	for (xtest, ytest) in test:
		xtest = xtest.to(device)
		output = model(xtest).to(device)
		print('Actual: ', ytest[0], ' Predicted: ', output[0], ' Loss: ', lfc(ytest[0], output[0]), ' PCE: ', pce(output[0], ytest[0]), file=file)
	file.close()




	model = net(*netargs, **netkwargs)
	opt = torch.optim.Adam(model.parameters(), lr=lr)
	model = model.to(device)

	trainLoss, trainPCE, testLoss, testPCE = trainer(model, high, [], [], low, opt, lfc, epochs, device, verboseEpochs=verboseEpochs)
	# model should be passed by reference in the line above, right? 
	file = open('trainHighGuessLow' + filename, mode='w')
	for (xtest, ytest) in test:
		xtest = xtest.to(device)
		output = model(xtest).to(device)
		ytest = ytest.to(device)
		outprint = output[0].to('cuda')
		yprint = ytest[0].to('cuda')
		l = lfc(yprint, outprint)
		print('Actual: ', yprint, ' Predicted: ', outprint, ' Loss: ', l, ' PCE: ', percentError(outprint, yprint), file=file)
	file.close()

	final

# def finalTraining(x,y, net, epochs, batch, device, filename, *netargs, lfc = torch.nn.MSELoss(), lr=0.0001, verboseEpochs=None, **netkwargs):
# def getData(years, inputNumbers, targetNumbers, forecastOffset, datapointOffset, inputHours, device, norm='none'):
#	return data, ys, norms

# def trainer(model, data1, data2, val, test, opt, lfc, epochs, device, verboseEpochs=-1, returnModel = False):
# 	return trainLoss, trainPCE, valLoss, valPCE, testLoss, testPCE, r2
# 	if returnModel: return trainLoss, trainPCE, testLoss, testPCE, model

# 			printLocation = open(file, mode='a')
#			print(f'training loss:', file=printLocation)
#			for r in trainLoss: print(r, file=printLocation)


# This function trains a model on all data available, then tests that model on the longest available subset. This should be given the inputs, but for right now it's just going to be hard coded.
def getPredictions(net, epochs, batch, device, filename, inputs, targets, forecastOffset, datapointOffset, numInputHours, *netargs, trainYearStart=1981, trainYearEnd=2021, lfc = torch.nn.MSELoss(), lr = 0.0001, verboseEpochs=-1, datanorm='None', **netkwargs):
	# get the data
	x,y,norm = getData((trainYearStart,trainYearEnd), inputs, targets, forecastOffset, datapointOffset, numInputHours, device, norm=datanorm)
	ziplist = list(zip(x,y))
	data = torch.utils.data.DataLoader(ziplist, batch_size = batch)
	lfc = networks.ohmsLoss(norm, physWeight = 0.0001)

	model = net(*netargs, **netkwargs)
	model.to(device)
	# model is a mutable datatype, so it is passed by reference, and model will be trained after this line.
	_, _, _, _, trained = trainer(model, data, [], [], [], torch.optim.Adam(model.parameters(), lr=lr), lfc, epochs, device, verboseEpochs = verboseEpochs, returnModel = True)

	# load the correct data
	start = 5592
	end = 7181
	x,y,norm = getData(1999, inputs, targets, forecastOffset, 1, numInputHours, device, norm=norm, frames=(start, end))
	ziplist = list(zip(x,y))
	# the next two variables should respond to the data, but I'm not that smart, so they are hard coded to be the longest stretch with 27,13,14,15,21,22,23,46,47 not NaN.
	ziplist = ziplist

	data = torch.utils.data.DataLoader(ziplist, batch_size = 1)

	yhat = torch.ones((y.shape[0],y.shape[1]))
	for i, dato in enumerate(data):
		xi, _ = dato
		output = trained(xi)
		yhat[i,:] = output

#	filename += 'predictions.txt'
	printFile = open(filename, mode='w')
	# I should probably print the inputs used here, but I'm lazy, and I don't want to refactor my results grapher.
	print('Actual:', file=printFile)
	for r in y: 
		for e in r: 
			print(e.item(), end=',', file=printFile)
		print('', file=printFile)
	print('predicted:', file = printFile)
	for r in yhat:
		for e in r: 
			print(e.item(), end=',', file=printFile)
		print('', file=printFile)
	metric = mare()
	metric._update([yhat,torch.tensor(y)])
	factors = 1
	for f in yhat.shape[1:]: factors *= f
	m = metric.compute()
	rmse = torch.sqrt(torch.nn.MSELoss()(torch.tensor(y),yhat))
	print(torch.mean(yhat))
	print(np.mean(y))
	r2 = sm.r2_score(y, yhat.detach().cpu())
	print(f'RMSE: {rmse}, average PCE per datum: {m}, average PCE per individual element (x_i_j): {m/factors}, R2: {r2}', file=printFile)
	print(f'RMSE: {rmse}, average PCE per datum: {m}, average PCE per individual element (x_i_j): {m/factors}, R2: {r2}')
	metric.reset()
	printFile.close()
	return None









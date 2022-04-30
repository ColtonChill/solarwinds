
import torch

def newSizer(orig, size, stride, pad):
	'''
	This is a function that calculates the size of one side of an output from an convolutional or pooling layer.
	All the arguments should be integers. 
	'''
	if stride == None: stride = size
	return int((orig + 2*pad - size)/stride + 1)


class baseNet(torch.nn.Module):
	def __init__(self):
		super(baseNet, self).__init__()
		self.layer = torch.nn.Linear(8, 12) # this doesn't learrn anytihng, and it shouldnt
		pass

	def forward(self, x):
		return x[:,-1,:7] # I really should have code to get this 7, but it's because our inputs contain the 7 outputs 12 hours ago, and two extra proton flux fields.

class CnnNet(torch.nn.Module):
	def __init__(self, targetLen=5, inputHours=24, numInputs=7, kernel1width=5, kernel1channels=10, mpool = 2, kernel2width=5, kernel2channels=10, apool=3, dense=1024, dense2=2048):
		super(CnnNet, self).__init__()
		# note: tuples in the 'size' argument in torch allow for non-square kernels, which are necessary for time based CNNs.
		self.conv1 = torch.nn.Conv2d(1,kernel1channels, (kernel1width, 1))
		n = newSizer(inputHours, kernel1width, 1, 0)
		self.mp = torch.nn.MaxPool2d((mpool, 1)) 
		n = newSizer(n, mpool, mpool, 0)
		self.conv2 = torch.nn.Conv2d(kernel1channels, kernel2channels, (kernel2width, numInputs))
		n = newSizer(n, kernel2width, 1, 0)
		self.avp = torch.nn.AvgPool2d((apool, 1))
		self.vectorSize = kernel2channels * newSizer(n, apool, apool, 0)
		self.relu = torch.nn.ReLU()
		
		self.fc1 = torch.nn.Linear(self.vectorSize, dense)
		self.fc2 = torch.nn.Linear(dense, dense2)
		self.fc3 = torch.nn.Linear(dense2, targetLen)

		self.inputHours = inputHours
		self.numInputs = numInputs
		self.targetLen = targetLen


	def forward(self, x):
		x = x.reshape(-1,1, self.inputHours, self.numInputs)
		x = self.relu(self.mp(self.conv1(x)))
		x = self.relu(self.avp(self.conv2(x)))
		x = self.fc3(self.fc2(self.relu(self.fc1(x.reshape((-1,1,self.vectorSize))))))
		return x.reshape(-1, self.targetLen)


class LSTM(torch.nn.Module):

	def __init__(self, num_classes = 4, input_size = 7, seq_length = 24, hidden_size = 8, num_layers = 4, dense = [1024, 2048]):
		super(LSTM, self).__init__()
		
		self.num_classes = num_classes
		self.num_layers = num_layers
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.seq_length = seq_length
		
		self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
		denseLayers = [seq_length * hidden_size, *dense, num_classes]
		# See gruNet for explanation of this line
		self.denseLayers = torch.nn.Sequential(*[torch.nn.Sequential(torch.nn.Linear(denseLayers[i-1], denseLayers[i]), torch.nn.ReLU()) for i in range(1, len(denseLayers)-1)], 
																	torch.nn.Linear(denseLayers[-2], denseLayers[-1]))

#		dense = [self.vectorSize, *dense, numOutputs]
#		self.denseLayers = torch.nn.Sequential(*[torch.nn.Sequential(torch.nn.Linear(dense[i-1], dense[i]), torch.nn.ReLU()) for i in range(1,len(dense)-1)], torch.nn.Linear(dense[-2], dense[-1]))



	def forward(self, x):
		bs = x.shape[0]
		output, (h_out, _) = self.lstm(x) #, (h_0, c_0)
		out = self.denseLayers(output.reshape(bs, self.seq_length * self.hidden_size))
		return out


class gruNet(torch.nn.Module):
	def __init__(self, numInputs, numOutputs, seqLen, hidden_size=7, num_layers=1, gruDrop=0.1, dense = [1024,2048]):
		super(gruNet, self).__init__()
		self.gru = torch.nn.GRU(input_size=numInputs, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=gruDrop, bidirectional=False)
		self.outSize = hidden_size * seqLen
		dense = [self.outSize, *dense, numOutputs]
		# This line serves to make the number of layers a hyperparameter as well as their width
		self.denseLayers = torch.nn.Sequential(
								*[torch.nn.Sequential(torch.nn.Linear(dense[i-1], dense[i]), torch.nn.ReLU()) for i in range(1, len(dense)-1)],
								torch.nn.Linear(dense[-2], dense[-1]))
		# we use a sequential module to bind all the dense layers to just one variable to allow us to call them later
		# The next line makes a linear layer that takes the prior layer's output size, it's output size, and fits a layer between them with an activation function (ReLU).
		# The final line is one more linear layer without an activation function to allow us to guess negative values as well as positive ones.


	def forward(self, x):
		bs = x.shape[0]
		output, _ = self.gru(x)
		out = self.denseLayers(output.reshape(bs, self.outSize))
		return out


class ResBlock(torch.nn.Module):
	def __init__(self, numInputs, numChannels=3, width = 3, af = torch.nn.ReLU):
		super(ResBlock, self).__init__()
		assert width % 2 # if it is even, we can't pad out to keep the same shape.
		padWidth = int((width-1)/2)
		assert numInputs % 2
		padHeight = int((numInputs-1)/2)
		self.conv1 = torch.nn.Conv2d(numChannels, numChannels, (width,1), padding=(padWidth,0))
		self.conv2 = torch.nn.Conv2d(numChannels, numChannels, (1, numInputs), padding=(0,padHeight))
		self.af = af()
		self.bn = torch.nn.BatchNorm2d(numChannels)
		
	def forward(self, x):
		yhat = self.conv2(x)
		return self.bn(self.af(self.conv1(yhat))) + x

class resCnnNet(torch.nn.Module):
	def __init__(self, numInputs, inputHours, numOutputs, numChannels = 3, af = torch.nn.ReLU,
				  resBlockWidth = 3, numBlocks = 7, dense = [128,256]):
		super(resCnnNet, self).__init__()
		self.hours = inputHours
		self.numInputs = numInputs
		self.first = torch.nn.Conv2d(1, numChannels, 1)
		self.blocks = torch.nn.Sequential(
				  *[ResBlock(numInputs, numChannels = numChannels, width = resBlockWidth, af = af) 
				  for _ in range(numBlocks)])
		# This assumes that dense is a list of values for dense layers.
		self.vectorSize = numChannels * numInputs * inputHours 
		dense = [self.vectorSize, *dense, numOutputs]
		self.denseLayers = torch.nn.Sequential(*[
					torch.nn.Sequential(torch.nn.Linear(dense[i-1], dense[i]), torch.nn.ReLU()) for i in range(1,len(dense)-1)
					], 
				torch.nn.Linear(dense[-2], dense[-1]))
		
	def forward(self, x):
		# This first one doesn't have a skip connection... Is that going to be a problem?
		x = self.first(x.reshape(-1,1,self.hours, self.numInputs))
		x = self.blocks(x)
		x = x.reshape(-1, self.vectorSize)
		x = self.denseLayers(x)
		return x
		




class rotateResBlock(torch.nn.Module):
	def __init__(self, numInputs, width = 3, af = torch.nn.ReLU):
		super(rotateResBlock, self).__init__()
		assert width % 2 # if it is even, we can't pad out to keep the same shape.
		padWidth = int((width-1)/2)
		assert numInputs % 2
		padHeight = int((numInputs-1)/2)
		numChannels = 1 # This could probably be a hyper parameter, but I'm experimenting
		self.conv1 = torch.nn.Conv2d(1, numChannels, (width,1), padding=(padWidth,0))
		self.conv2 = torch.nn.Conv2d(numChannels,numInputs, (1, numInputs), padding=(0,0))
		self.af = af()
		self.bn = torch.nn.BatchNorm2d(numChannels)
		
	def forward(self, x):
		yhat = self.conv1(x)
		# we need this transpose operation to 'rotate' our input planes to match up with x again
		yhat = self.conv2(x).transpose(1,3)
		return self.bn(self.af(yhat)) + x

class rotateRCN(torch.nn.Module):
	def __init__(self, numInputs, inputHours, numOutputs, af = torch.nn.ReLU,
				  resBlockWidth = 3, numBlocks = 7, dense = [128,256]):
		super(rotateRCN, self).__init__()
		self.hours = inputHours
		self.numInputs = numInputs
		self.first = torch.nn.Conv2d(1, 1, 1)
		self.blocks = torch.nn.Sequential(
				  *[rotateResBlock(numInputs, width = resBlockWidth, af = af) 
				  for _ in range(numBlocks)])
		# This assumes that dense is a list of values for dense layers.
		self.vectorSize = 1 * numInputs * inputHours 
		dense = [self.vectorSize, *dense, numOutputs]
		self.denseLayers = torch.nn.Sequential(*[
						torch.nn.Sequential(torch.nn.Linear(dense[i-1], dense[i]), torch.nn.ReLU()) for i in range(1,len(dense)-1)
						], 
					torch.nn.Linear(dense[-2], dense[-1]))
		
	def forward(self, x):
		# This first one doesn't have a skip connection... Is that going to be a problem?
		x = self.first(x.reshape(-1,1,self.hours, self.numInputs))
		x = self.blocks(x)
		x = x.reshape(-1, self.vectorSize)
		x = self.denseLayers(x)
		return x
		







class ohmsLoss(torch.nn.Module):
	# Ohm's law for an ideal plasma states that E + v x B = 0, where E is the electric field, v is the velocity field, and B is the magnetic field.
		# If the plasma is less than ideal, then the zero becomes rho J, where rho is the electrical resistivity and J is the electric current field.
		# For this dataset, alpha |E| <= ||v x B||, where alpha is a constant.


	# Using this assumes that targets are [27,13,14,15,21,22,23]
	# This alpha value corresponds to non-normalized data. 
		# in a perfect world, this would just represent the factor due to units between these E and v x B, but we use a small fudge factor as well.
		# This alpha was calulated by using the jupyter notebook alphaFinder, for the given error.
	def __init__(self, norms, physWeight=.25, alpha = 0.001, exponent=1):
		super(ohmsLoss, self).__init__()
		self.weight=physWeight
		self.mse = torch.nn.MSELoss()
		self.relu = torch.nn.ReLU()
		self.alpha = alpha
		self.exponent = exponent
#		self.error = error
		self.norms = norms
		# this line is experimental
		self.factor = norms.mul[0] # this corresponds to the electric field value, which is on the same order of magnitude as our loss. 

	def forward(self, preds, target):
		# we need to un normalize, so we add the lower part for max-min, then we multiply for max, maxmin, and 0mean, then we add the final bit for all but max. 
		# No normalization has 0, 1 and 0 for these values, respectively, as does any normalization that does not use that given term.
		input = (preds[:,0:7] - self.norms.lr[0:7])
		input *= self.norms.mul[0:7] 
		input += self.norms.ud[0:7]
		x = input[:,2] * input[:,6] - input[:,3] * input[:,5]
		y = input[:,3] * input[:,4] - input[:,1] * input[:,6]
		z = input[:,1] * input[:,5] - input[:,2] * input[:,4]
		norm = torch.sqrt(x**2 + y**2 + z**2)

		# for the quation to hold, v x B and E must be in opposite directions, so we do that here. E is taken care of with torch.abs, and norm cannot be negative.
		# Applyint the negative to one of these two shows what would happen if they were in opposite directions.
		err = self.relu(torch.abs(input[:,0]) - self.alpha * norm)

		return (1-self.weight) * self.mse(preds, target)  + self.weight * (torch.mean(err))**self.exponent /self.factor

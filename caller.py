
import torch
import argparse

import cfv
import networks


parser = argparse.ArgumentParser()


# Data parameters
# defaults are the beginning and end of solar cycle 24, as found on wikipedia.
parser.add_argument('--yearStart', type=int, default=2009, help='The first year of data to train on.')
parser.add_argument('--yearEnd', type=int, default= 2019, help='The final year of data to train on.')
parser.add_argument('--normalization', default='0mean', help='Type of normalization: \n\t\t 0mean: zero mean and unit deviation. \n\t\t 0-1: min set to zero, max set to one. \n\t\t n-m: 0 to 1, but from n to m, m<n.  none: no normalization.\n\t\t MaxN: divide everything by the fartest entry from zero, multiply by max.\n Choose multple by separating with commas. ')
parser.add_argument('-u', '--full', action='store_true', help='Use the full dataset - from 1981 to 2021. Overrides yearStart and yearEnd args.')

# define what inputs we want to consider for our model - names listed in following cell
parser.add_argument('--inputs', default='27,13,14,15,21,22,23,46,47', help='a list of values corresponding to the positions of inputs in the omni data.')
parser.add_argument('--targets', default='27,13,14,15,21,22,23', help='entris for the regression targets') # I should probably not let this be an argument, since the loss function depends on all these
parser.add_argument('--forecastOffset', type=int, default=12, help='integer number of hours between the final input datapoint and the forcasted value')
parser.add_argument('--inputHours', type=int, default=24, help='integer number of hours of data in the input space')
parser.add_argument('--datapointOffset', type=int, default=36, help='integer number of hours between the starts of two datapoints - forecastOffset + inputHours gives no overlap in datapoints.')

# ML parameters
parser.add_argument('--device', default ='cuda:0' if torch.cuda.is_available() else 'cpu', help = 'Device to utilize for GPU acceleration - should be \'cuda:x\' where x is the 0 indexed GPU.')
parser.add_argument('--batchSize', type=int, default=128, help='amount of datapoints in a single training instance')
parser.add_argument('--epochs', type=int, default=2000, help='number of times to train and update a network')
parser.add_argument('--verboseEpochs', type=int, default=-1, help='how many epochs between print statements to update on progress. set negative to not print updates.')
parser.add_argument('--lr', type=float, default=1, help='Learning rate for the network.')
parser.add_argument('--dest', default='results/', help='path to directory for results')

# Loss Function Parameters
parser.add_argument('--lam', default='0', help="Value between 0 and 1 to weight between RMSE and Ohm's law based loss. 0 is entirely RMSE, 1 is entirely Ohm's law based. Separate with commas to try multiple.")

# Network parameters
parser.add_argument('--netNumber', type=int, default=-1, help='Which net to run. R2 net = 0, LSTM = 1, CnnNet = 2, ResCnnNet = 3, RotateRCN = 4, GRU = 5, all = -1.')
parser.add_argument('-f', '--final', action='store_true', help='Train a single instance of the supplied nets, then print their output to a file for comparision.')
parser.add_argument('-p', '--predict', action='store_true', help='Train a single instance of the suppliet net, then print the longest forecast available.')

args = parser.parse_args()
if args.full: 
	parser.yearStart = 1981
	parser.yearEnd = 2021
norms = args.normalization.split(',')
lams = args.lam.split(',')
lams = [float(l) for l in lams]
inputs = [int(i) for i in args.inputs.split(',')]
targets = [int(i) for i in args.targets.split(',')]
# Before we use Ohm's law loss, make sure it has what it needs
if len(targets) < 7: ohms = False
else: ohms = (targets[:7] == [27,13,14,15,21,22,23])

if not ohms:
	print('Using standard MSE loss, as either the correct inputs were not in for its use, or they were in the wrong order.')
	lams = ['none']

filename = str(args.yearStart) + '-' + str(args.yearEnd)

ohmDict = { 'none': {'alpha':0.0008555, 'error':4.25},
		'0mean':{'alpha':0.38, 'error':9},
		'max1':{'alpha':0.155, 'error':0.09},
		'max100':{'alpha':0.0016, 'error':6.5}
	}


if __name__ == '__main__':
	LSTMkwargs = {
		'num_classes' : len(targets),
		'input_size' : len(inputs),
		'seq_length' : args.inputHours
		}
	cnnkwargs = {
		'inputHours' : args.inputHours,
		'numInputs' : len(inputs),
		'targetLen' : len(targets)
		}
	RCNkwargs = {
		'numInputs': len(inputs),
		'inputHours': args.inputHours,
		'numOutputs': len(targets)
		}
	rotatekwargs = {
		'numInputs': len(inputs),
		'inputHours': args.inputHours,
		'numOutputs': len(targets)
		}
	grukwargs = {
		'numInputs': len(inputs),
		'numOutputs': len(targets),
		'seqLen' : args.inputHours
		}
	# in hindsight, this should be in a different file.
	for norm in norms:
#		hasMSE = False
#		if ohms and (norm not in ohmDict):
#			print(f'Normalzation scheme {norm} is not supported. Try one of the following:', *ohmDict)
#			lam = ['none']
#		elif ohms:
		lam = lams
		for i in range(len(lam)):
#			if ohms:
			x,y,normInfo = cfv.getData([args.yearStart, args.yearEnd], inputs, targets, args.forecastOffset, args.datapointOffset, args.inputHours, args.device, norm=norm)
			lfc = networks.ohmsLoss(normInfo, physWeight=lam[i])
			baseFile = norm + filename + 'lam' + str(lam[i]) + '.txt'

			print('Data Loaded. Beginning grid search.')

			if (not args.final) and (not args.predict):
				LSTMgrid = {
					'hidden_size' : [8, 16],
					'num_layers' : [4, 8],
					'dense' : [[2048,2048], [256,512,128]]
					}
				cnnGrid = {
					'kernel1width' : [5],
					'kernel1channels' : [15, 20],
					'mpool' : [2],
					'kernel2width' : [5],
					'kernel2channels' : [15, 20],
					'apool' : [2],
					'dense' : [2048],
					'dense2' : [2048]
					}
				RCNgrid = {
					'numChannels' : [5],
					'resBlockWidth' : [3,5,7],
					'numBlocks' : [20, 30],
					'dense' : [[256,512,128], [128,256,512,128]],
					'af': [torch.nn.ReLU, torch.nn.ELU]
					}
				rotateGrid = {
					'resBlockWidth' : [3,5,7],
					'numBlocks' : [20,30],
					'dense' : [[256,512,128], [128,256,512,128]],
					'af': [torch.nn.ReLU, torch.nn.ELU]
					}
				gruGrid = {
					'hidden_size': [8,15],
					'num_layers' : [2,3],
					'gruDrop' : [0.01, 0.1],
					'dense' : [[256,512,128]]
					}


				if args.netNumber < 0 or args.netNumber == 0:
					lmla, lml, lmpa, lmp = cfv.gridSearch3CVMSE(x,y, networks.baseNet, args.epochs, args.batchSize, {}, 'cpu', lfc=lfc, lr = 1, file=args.dest + 'baseNet'+baseFile, trainable=False)
					print(f'Minimum loss achieved by baseNet: {lml}. Args: {lmla}')
					print(f'Minimum percent error achieved by baseNet: {lmp}. Args: {lmpa}')
					LSTMfile = open(args.dest + 'baseNet'+baseFile, mode = 'a')
					print(f'Minimum loss achieved by baseNet: {lml}. Args: {lmla}', file=LSTMfile)
					print(f'Minimum percent error achieved by baseNet: {lmp}. Args: {lmpa}', file=LSTMfile)
					LSTMfile.close()



				if args.netNumber < 0 or args.netNumber == 1:
					lr = args.lr if args.lr != 1 else 0.0001
					lmla, lml, lmpa, lmp = cfv.gridSearch3CVMSE(x,y, networks.LSTM, args.epochs, args.batchSize, LSTMgrid, args.device, lfc=lfc, lr = lr, file=args.dest + 'LSTM'+baseFile, **LSTMkwargs)
					print(f'Minimum loss achieved by LSTM: {lml}. Args: {lmla}')
					print(f'Minimum percent error achieved by LSTM: {lmp}. Args: {lmpa}')
					LSTMfile = open(args.dest + 'LSTM'+baseFile, mode = 'a')
					print(f'Minimum loss achieved by LSTM: {lml}. Args: {lmla}', file=LSTMfile)
					print(f'Minimum percent error achieved by LSTM: {lmp}. Args: {lmpa}', file=LSTMfile)
					LSTMfile.close()

				if args.netNumber < 0 or args.netNumber == 2:
					lr = args.lr if args.lr != 1 else 0.00002
					cmla, cml, cmpa, cmp = cfv.gridSearch3CVMSE(x,y, networks.CnnNet, args.epochs, args.batchSize, cnnGrid, args.device, lfc=lfc, lr = lr, file=args.dest + 'cnnNet'+baseFile, **cnnkwargs)
					cnnFile = open(args.dest + 'cnnNet'+baseFile, mode='a')
					print(f'Minimum loss achieved by cnnNet: {cml}. Args: {cmla}')
					print(f'Minimum percent error achieved by cnnNet: {cmp}. Args: {cmpa}')
					print(f'Minimum loss achieved by cnnNet: {cml}. Args: {cmla}', file=cnnFile)
					print(f'Minimum percent error achieved by cnnNet: {cmp}. Args: {cmpa}', file=cnnFile)
					cnnFile.close()

				if args.netNumber < 0 or args.netNumber == 3:
					lr = args.lr if args.lr != 1 else 0.00009
					rcmla, rcml, rcmpa, rcmp = cfv.gridSearch3CVMSE(x,y, networks.resCnnNet, args.epochs, args.batchSize, RCNgrid, args.device, lfc=lfc, lr = lr, file=args.dest + 'RCNwithActivation'+baseFile, **RCNkwargs)
					cnnFile = open(args.dest + 'RCNwithActivation'+baseFile, mode='a')
					print(f'Minimum loss achieved by resCnnNet: {rcml}. Args: {rcmla}')
					print(f'Minimum percent error achieved by resCnnNet: {rcmp}. Args: {rcmpa}')
					print(f'Minimum loss achieved by resCnnNet: {rcml}. Args: {rcmla}', file=cnnFile)
					print(f'Minimum percent error achieved by resCnnNet: {rcmp}. Args: {rcmpa}', file=cnnFile)
					cnnFile.close()


				if args.netNumber < 0 or args.netNumber == 4:
					lr = args.lr if args.lr != 1 else 0.00009
					rcmla, rcml, rcmpa, rcmp = cfv.gridSearch3CVMSE(x,y, networks.rotateRCN, args.epochs, args.batchSize, rotateGrid, args.device, lfc=lfc, lr = lr, file=args.dest + 'rotateRCN'+baseFile, **rotatekwargs)

					cnnFile = open(args.dest + 'rotateRCN'+baseFile, mode='a')
					print(f'Minimum loss achieved by rotateRCN: {rcml}. Args: {rcmla}')
					print(f'Minimum percent error achieved by rotateRCN: {rcmp}. Args: {rcmpa}')
					print(f'Minimum loss achieved by rotateRCN: {rcml}. Args: {rcmla}', file=cnnFile)
					print(f'Minimum percent error achieved by rotateRCN: {rcmp}. Args: {rcmpa}', file=cnnFile)
					cnnFile.close()


				if args.netNumber < 0 or args.netNumber == 5:
					lr = args.lr if args.lr != 1 else 0.0001
					lmla, lml, lmpa, lmp = cfv.gridSearch3CVMSE(x,y, networks.gruNet, args.epochs, args.batchSize, gruGrid, args.device, lfc=lfc, lr = lr, file=args.dest + 'GRU'+baseFile, **grukwargs)
					print(f'Minimum loss achieved by GRU: {lml}. Args: {lmla}')
					print(f'Minimum percent error achieved by GRU: {lmp}. Args: {lmpa}')
					LSTMfile = open(args.dest + 'GRU'+baseFile, mode = 'a')
					print(f'Minimum loss achieved by GRU: {lml}. Args: {lmla}', file=LSTMfile)
					print(f'Minimum percent error achieved by GRU: {lmp}. Args: {lmpa}', file=LSTMfile)
					LSTMfile.close()

			elif args.final:
				LSTMkwargs = { **LSTMkwargs,
					'hidden_size' : 8,
					'num_layers' : 4,
					'dense' : [2048,2048]
					}
				RCNkwargs = { **RCNkwargs,
					'numChannels' : 5,
					'resBlockWidth' : 5,
					'numBlocks' : 20,
					'dense' : [128,512,1024,2048],
					'af': torch.nn.ReLU
					}
				cnnkwargs = { **cnnkwargs, 
					'kernel1width' : 5,
					'kernel1channels' : 20,
					'mpool' : 2,
					'kernel2width' : 5,
					'kernel2channels' : 20,
					'apool' : 2,
					'dense' : 2048,
					'dense2' : 2048
					}


				if args.netNumber < 0 or args.netNumber == 1:
					cfv.finalTraining(x,y, networks.LSTM, args.epochs, args.batchSize, args.device, args.dest + 'LSTMtestOutputs'+baseFile, verboseEpochs = args.verboseEpochs, 
							lr = args.lr, **LSTMkwargs)
				if args.netNumber < 0 or args.netNumber == 2:
					cfv.finalTraining(x,y, networks.CnnNet, args.epochs, args.batchSize, args.device, args.dest + 'cnntestOutputs'+baseFile, verboseEpochs = args.verboseEpochs, lr = args.lr, **cnnkwargs)
				if args.netNumber < 0 or args.netNumber == 3:
					print('Now training resCNN')
					cfv.finalTraining(x,y, networks.resCnnNet, args.epochs, args.batchSize, args.device, args.dest + 'resCnntestOutputs'+baseFile, verboseEpochs = args.verboseEpochs, 
							lr = args.lr, **RCNkwargs)
			elif args.predict:
				LSTMkwargs = { **LSTMkwargs,
					'hidden_size' : 8,
					'num_layers' : 4,
					'dense' : [2048,2048]
					}
				RCNkwargs = { **RCNkwargs,
					'numChannels' : 5,
					'resBlockWidth' : 5,
					'numBlocks' : 20,
					'dense' : [128,512,1024,2048],
					'af': torch.nn.ReLU
					}
				cnnkwargs = { **cnnkwargs, 
					'kernel1width' : 5,
					'kernel1channels' : 20,
					'mpool' : 2,
					'kernel2width' : 5,
					'kernel2channels' : 20,
					'apool' : 2,
					'dense' : 2048,
					'dense2' : 2048
					}

				predArgs1 = [args.epochs, args.batchSize, args.device]
				predArgs2 = [inputs, targets, args.forecastOffset, args.datapointOffset, args.inputHours]
				standardKwargs = {'trainYearStart' : args.yearStart,
							'trainYearEnd' : args.yearEnd,
							'lfc' : lfc,
							'verboseEpochs' : args.verboseEpochs,
							'datanorm' : norm
						}
				if args.netNumber < 0 or args.netNumber == 1:
					lr = args.lr if args.lr != 1 else 0.0001
					cfv.getPredictions(networks.LSTM, *predArgs1, args.dest + 'LTSMforecast.txt', *predArgs2, lr=lr, **standardKwargs, **LSTMkwargs)
				if args.netNumber < 0 or args.netNumber == 2:
					lr = args.lr if args.lr != 1 else 0.00002
					cfv.getPredictions(networks.CnnNet, *predArgs1, args.dest + 'CNNforecast.txt', *predArgs2, lr=lr, **standardKwargs, **cnnkwargs)
				if args.netNumber < 0 or args.netNumber == 3:
					lr = args.lr if args.lr != 1 else 0.00009
					cfv.getPredictions(networks.resCnnNet, *predArgs1, args.dest + 'RCNforecast.txt', *predArgs2, lr=lr, **standardKwargs, **RCNkwargs)
				if args.netNumber < 0 or args.netNumber == 4:
					lr = args.lr if args.lr != 1 else 0.00009 #TODO
					cfv.getPredictions(networks.rotateRCN, *predArgs1, args.dest + 'rotateForecast.txt', *predArgs2, lr=lr, **standardKwargs, **RCNkwargs)


# def getPredictions(net, epochs, batch, device, filename, inputs, targets, forecastOffset, datapointOffset, inputHours, *netargs, trainYearStart=1981, trainYearEnd=2021, lfc = torch.nn.MSELoss(), lr = 0.0001, ve>
#                   per  [         args1       ]   per     [                         args2                             ] per-ish  {                   kwargs                                        } per         








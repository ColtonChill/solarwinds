import argparse
from matplotlib import pyplot as plt
import numpy as np

def analyzeLines(l):
	if '{' in l[0]: return l, 0
	# The training regimen always starts with training loss
	increment = l[0].startswith('training loss: [')
	values = np.zeros((len(l), 3))
	title = None
	for i, line in enumerate(l):
		if ':' in line:
			title, line = line.split(':')
		line = line.strip('[] \n')
		numbers = [float(v) for v in line.split()]
		values[i,:] = numbers
	return [values, title], increment


def readfile(fileName):
	file = open(fileName, mode='r')
	entries = []
	lines = []
	i = -1 # I'd use enumerate here, but I don't want it to read the entire file at once.
	levels = 0 # this will keep track of how many 
	for line in file:
		# All lines that start an array have a colon
		if ':' in line:
			# Make sure this isn't the first line
			if len(lines) != 0:
				# Analyze the lines
				l, increment = analyzeLines(lines)
				# increment if this is the beginning of the next array
				i += increment
				if increment:
					entries.append([])
				# Put the analyzed data into our list
				entries[i].append(l)
			lines = []
		lines.append(line)
	# include the final line
	l, _ = analyzeLines(lines)
	entries[i].append(l)

	return entries

def grapher(entries, bestOnly=False):
	for entry in entries:
		ax1 = plt.subplot(2,1,1)
		ax1.plot(entry[0][0], label=entry[0][1])
		ax1.plot(entry[1][0], label=entry[1][1])
		ax2 = plt.subplot(2,1,2)
		ax2.plot(entry[2][0], label=entry[2][1])
		ax2.plot(entry[3][0], label=entry[3][1])
		plt.title(entry[4])
		plt.legend
		plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('fileName', help='Name of the file to graph.')
parser.add_argument('--BestOnly', '-b', action='store_true', help='Only graph the losses of the lowest grid params.')
args = parser.parse_args()

e = readfile(args.fileName)
grapher(e)

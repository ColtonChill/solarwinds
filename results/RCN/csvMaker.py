

from sys import argv

outfile = 'resultCSV.csv'
out = open(outfile, mode='w')
print('startyear;endyear;normalization;learningRate;lambdaValue;PCE;R2;ResBlockWidth;numBlocks;dense;af',file=out)

for file in argv:
	if file.endswith('.txt'):
		print(file)
		lr, norm = file.split('/', 1)
		norm = norm.split('lam')[0]
		end = norm[-4:]
		start = norm[-9:-5]
		norm = norm[17:-9]
		lam = float ('0.' + file.split('.')[-2])
		f= open(file, mode='r')
		for line in f:
			if 'test R2: 0.' in line: # or 'test PCE: 0.' in line:  # this left due to pce favoring 100-1000 exclusively
				words = line.split()
				criteria = [words[5].strip(','), words[8].strip(',')]
				args = line.split('args: ')[-1]
				# remove the curly braces and the leading '
				args = str(args[2:-2])
				args = args.split("'")
				args = [ *[a.strip(': ,') for a in args[1::2]], args[-2].strip(': ,')]
				args = args[-6:]
				args.pop(-2)
				args.pop(-2)
				print(start, end, norm, lr, lam, *criteria, *args, file=out, sep=';')
		print('', end='', file=out, flush=True)
		f.close()

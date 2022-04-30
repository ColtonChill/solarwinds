

from sys import argv

x = float('inf')

for file in argv[1:]:
	print(file)
	file = open(file, mode='r')
	for line in file:
		if 'Minimum loss achieved by ' in line:
			line = line.split(':')[1]
			num = line.strip(' .Args')
			x = min(x, float(num))
	file.close()

print(x)

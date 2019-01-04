##LESSSE
##10 November 2018
##gmidi
##____________
## methods for sparse array disk representation
##____________

import numpy as np

def save(file,arr,compress=True):
	indices = np.asarray([i for i in zip(*np.nonzero(arr))])
	values = arr[np.nonzero(arr)]
	shape = np.asarray(arr.shape)
	if compress:
		np.savez_compressed(file,shape=shape,indices=indices,values=values)
	else: 		
		np.savez(file,shape=shape,indices=indices,values=values)

def load(file):
	data = np.load(file)
	array = np.zeros(tuple(data['shape']))
	for i in zip(data['indices'],data['values']):
		array[tuple(i[0])]=i[1]
	return array
		

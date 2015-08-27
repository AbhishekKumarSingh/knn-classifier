#!/usr/bin/python
import csv
import math
import random
import matplotlib.pyplot as plt


def load_dataset(filename):
	''' Reads a csv file and store data in a list dataset where
	    item at each index is a list containing info of an instance
	'''
	with open(filename, 'r') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		del dataset[-1]
		return dataset

def kfold_validation(dataset, fold, shuffle=False):
	''' Returns validation, training dataset for k iterations
	''' 
	if shuffle:
		# shuffle dataset to ensure randomness
		random.shuffle(dataset)
	for iteration in range(0, fold):
		valid_set = [item for index, item in enumerate(dataset) if index%fold == iteration]
		train_set = [item for index, item in enumerate(dataset) if index%fold != iteration]
		yield valid_set, train_set

def euclidean_distance(inst1, inst2, nf):
	''' Returns euclidean_distance between two instances
		nf represents number of features in instance
	'''
	diff_sq_sum = 0
	for i in range(0, nf):
		diff_sq_sum = diff_sq_sum + pow(float(inst1[i]) - float(inst2[i]), 2)
	res = math.sqrt(diff_sq_sum)
	return res

def calculate_dists(inst, trainingset, nf):
	''' Returns a list containing euclidean distances calculated
		from intance to all other training set points
	'''
	size = len(trainingset)
	dist = []
	for i in range(0, size):
		edistance = euclidean_distance(inst, trainingset[i], nf)
		info = (edistance, trainingset[i][-1])
		dist.append(info)
	return dist

def knn_identify(k, dist):
	''' Given a matrix containing euclidean distances it returns
	 	class label by considering majority vote in k nearest neighbours
	'''
	dist.sort(key=lambda x: x[0])
	class_freq = {}
	for i in range(0, k):
		# print dist, i
		cls = dist[i][1]
		class_freq[cls] = class_freq.get(cls, 0) + 1
	label = max(class_freq, key=class_freq.get)
	return label

def get_Allknn_acc_for_kfold(dataset, fold, kstart, kend, nf):
	''' for a given fold it returns a list containing
		mean accuracy for k->1:5
	'''
	ksize = kend - kstart + 1
	correct = [[] for i in range(ksize)]
	wrong = [[] for i in range(ksize)]
	test_set_sz = 0
	for test_set, train_set in kfold_validation(dataset, fold, True):
		test_set_sz += len(test_set)
		for inst in test_set:
			edist = calculate_dists(inst, train_set, nf)
			label = []
			for k in range(kstart, kend + 1):
				label.append(knn_identify(k, edist))
			for i in range(0, ksize):
				if label[i] == inst[-1]:
					correct[i].append(inst) 
				else:
					wrong[i].append(inst)
	accuracy = [(float(len(correct[i]))/test_set_sz)*100 for i in xrange(ksize)]
	return accuracy

def plot_graph(mean_accuracy, kstart, kend, sd, ptitle):
	''' plots error bar graph. k values on x axis and
		mean accuracy on y axis.
	'''
	fig, axs = plt.subplots(nrows=2, ncols=2)
	y = mean_accuracy[0]
	x = range(kstart, kend+1)
	ax = axs[0,0]
	ax.errorbar(x, y, yerr=sd[0], fmt='o')
	ax.set_xlabel("k        ")
	ax.set_ylabel("Mean accuracy")
	ax.set_title("2 fold KNN ")

	y = mean_accuracy[1]
	x = range(kstart, kend+1)
	ax = axs[0,1]
	ax.errorbar(x, y, yerr=sd[1], fmt='o')
	ax.set_title("3 fold KNN ")
	ax.set_xlabel("k        ")
	ax.set_ylabel("Mean accuracy")

	y = mean_accuracy[2]
	x = range(kstart, kend+1)
	ax = axs[1,0]
	ax.errorbar(x, y, yerr=sd[2], fmt='o')
	ax.set_title("4 fold KNN ")
	ax.set_xlabel("k        ")
	ax.set_ylabel("Mean accuracy")

	y = mean_accuracy[3]
	x = range(kstart, kend+1)
	ax = axs[1,1]
	ax.errorbar(x, y, yerr=sd[3], fmt='o')
	ax.set_title("5 fold KNN ")
	ax.set_xlabel("k         ")
	ax.set_ylabel("Mean accuracy")

	fig.suptitle(ptitle)
	plt.show()
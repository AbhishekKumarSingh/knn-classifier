#!/usr/bin/python
from os import sys
import numpy
from util import load_dataset, get_Allknn_acc_for_kfold, plot_graph

def registry(filename,nf, ptitle, kfstart=2, kfend=5, kstart=1, kend=5):
	''' starts the project. For each fold it calculates mean accuracy,
		standard deviation and plot the corresponding graph.
	'''
	dataset = load_dataset(filename)
	kf_accuracy = []
	for kf in range(kfstart, kfend+1):
		kf_accuracy.append(get_Allknn_acc_for_kfold(dataset, kf, kstart, kend, nf))
	kf_mean_acc = [sum(acclist)/len(acclist) for acclist in kf_accuracy]
	sd = [numpy.std(acclist) for acclist in kf_accuracy]

	for kf, acclist in zip(range(kfstart, kfend+1),kf_accuracy):
		print kf, "fold validation ===> accuracy of", sum(acclist)/len(acclist)
	# print kf_mean_acc
	mean_sd = sum(sd)/len(sd)
	mean_acc = sum(kf_mean_acc)/len(kf_mean_acc)
	print "Mean accuracy : ", mean_acc
	print "Mean S.D : ", mean_sd
	plot_graph(kf_accuracy, kstart, kend,sd, ptitle)


if __name__ == '__main__':
	filename, nf, ptitle, = [arg for arg in sys.argv[1:4]]
	registry(filename, int(nf), ptitle)



# dataset = [[3, 104, 'R'], [2, 100, 'R'], [1, 81, 'R'], [101, 10, 'A'], [90, 5, 'A'], [98, 2, 'A']]
# dataset = load_dataset('bezdekIris.data')
# print get_Allknn_acc_for_kfold(dataset, 2, 1, 5, 4)
#print knn_classfier('bezdekIris.data', 2, 4)
#registry('optdigits.tra', 2, 5, 1, 5, 64)
#registry('pima-indians-diabetes.data', 2, 5, 1, 5, 8, 'KNN on diabetes dataset')


#registry('optdigits.tra', 2, 5, 1, 5, 64, 'KNN on Optical digits dataset')
#registry('glass.data', 2, 5, 1, 5, 10, 'KNN on glass dataset')
# registry('haberman.data', 2, 5, 1, 5, 3, 'KNN on habermann dataset')
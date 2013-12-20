import pickle
import os 

def main():

	os.chdir("TrainINKML_v3")
	X = pickle.load(open("train3X.p",'rb'))
	y = pickle.load(open("train3y.p",'rb'))

	print len(X), len(y)

	# fox x1,y1 in zip(X,y):
	# 	print len(x1), y1



	# pca = PCA(n_components=35)
	# pca = pca.fit(X)

	# pickle.dump( pca, open( "spatialpca.p", "wb" ) )

	# pcaX = pca.transform(X)
	# yes, no = 0.0, 0.0 

	# spatialSVM = svm.SVC()
	# spatialSVM = spatialSVM.fit(pcaX, y)	

	# pickle.dump( spatialSVM, open( "spatialsvm.p", "wb" ) )



if __name__ == '__main__':
	main()
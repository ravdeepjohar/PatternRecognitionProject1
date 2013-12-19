# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import cPickle as pickle
from scipy.misc import comb
import numpy as np
import os
import matplotlib.pyplot as plt
from pylab import *
from sklearn.decomposition import PCA
from sklearn import svm


indexFile = open("index.csv")
classDict = {}
numRadii = 15
numTheta = 20

relationToClass = {'R':1,'A':2,'B':2,'I':3,'Sup':4,'Sub':5}

for line in indexFile:
	 classDict[line.split(",")[0]]= int((line.split(",")[1]).strip())
		
indexFile.close()
		
def bernstein_poly(i, n, t):
		"""
		 The Bernstein polynomial of n, i as a function of t
		"""
		return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def getxycor(root):
	xcor = []
	ycor = []
	for neighbor in root.iter('{http://www.w3.org/2003/InkML}trace'):
		t = neighbor.text.split(",")
		x = []
		y = []
		for cor in t:
			if cor.startswith(" "):
				cor = cor.strip()
			x.append(float(cor.split(" ")[0]))
			y.append(float(cor.split(" ")[1]))
		xcor.append(x)
		ycor.append(y)
	return xcor,ycor

def getSymbolIndices(root):
	symbols = []
	indices = []
	for neighbor in root.findall('{http://www.w3.org/2003/InkML}traceGroup'):
		 for n in neighbor.findall('{http://www.w3.org/2003/InkML}traceGroup'):
			  for n2 in n.iter('{http://www.w3.org/2003/InkML}annotation'):
				  symbols.append(n2.text)
				  tempind = []
				  for n in n.iter('{http://www.w3.org/2003/InkML}traceView'):
					  tempind.append(int(n.attrib["traceDataRef"]))
			  indices.append(tempind)
	return symbols, indices



def getsymbolspairs(noofstrokes):
	strokepairs = []
	for i in range(noofstrokes):
		for j in range(i+1,noofstrokes):
			strokepairs.append([i,j])

	return strokepairs

def createPoints():
	
	fTrain = open("train.txt", 'rb')
	#fTest = open("test2.txt",'rb')

	os.chdir("TrainINKML_v3")

	X, y = [], []
	testX, testy = [], []

	for line in fTrain:

		inkmlfilelocation =  line[:-1].strip()
		if inkmlfilelocation.endswith(".inkml"):
			#inkmlfilelocation = "TrainINKML_v3/expressmatch/101_alfonso.inkml"
			tree = ET.parse(inkmlfilelocation)
			
			root = tree.getroot() 

			xcor,ycor = getxycor(root) 
			symbol, indices = getSymbolIndices(root)
		   
			inkmlfile = os.path.basename(inkmlfilelocation)
			base = os.path.splitext(inkmlfile)[0]
			lgfilepath = "../all_lg/"+base+".lg"

			relations = get_spatialRelations(lgfilepath)
			symbolpairs = getsymbolspairs(len(symbol))
		 

			indices = sorted(indices, key=lambda x: x[0], reverse=False)

			#print indices

			#print symbolpairs		

			for sympair in symbolpairs:
				

				firstsymbol = sympair[0]
				secondsymbol = sympair[1]

				#print indices[firstsymbol], indices[secondsymbol]

				stroke1 = indices[firstsymbol][-1]
				stroke2 = indices[secondsymbol][-0]

				if (stroke1, stroke2) in relations:
					#print (stroke1, stroke2),  relations[(stroke1, stroke2)] 

					ind = [indices[firstsymbol],indices[secondsymbol]]

					#print ind
					symbolpointsX,symbolpointsY = shiftPoints_SegmentStrokes(ind,xcor,ycor) 
					symbolpointsX,symbolpointsY = normalizedPoints_SegmentStrokes(ind,symbolpointsX,symbolpointsY)  
					bezierpointsX, bezierpointsY = getBezier_SegmentationStrokes(indices,symbolpointsX,symbolpointsY)    

					grid=calculateShapeFeatures(bezierpointsX, bezierpointsY, numRadii, numTheta)

					featurelist = []
					for g in grid:
						for val in g:
							featurelist.append(val)

					#print len(featurelist),  getRelationClass(relations[(stroke1, stroke2)])
					#exit()
					X.append(featurelist)
					y.append(getRelationClass(relations[(stroke1, stroke2)]))


	pca = PCA(n_components=100)
	pca = pca.fit(X)

	pickle.dump( pca, open( "spatialpca.p", "wb" ) )

	pcaX = pca.transform(X)
	yes, no = 0.0, 0.0 

	spatialSVM = svm.SVC()
	spatialSVM = spatialSVM.fit(pcaX, y)

	pickle.dump( spatialSVM, open( "spatialsvm.p", "wb" ) )
	

	# for line in fTest:

	# 	inkmlfilelocation =  line[:-1].strip()
	# 	if inkmlfilelocation.endswith(".inkml"):
	# 		#inkmlfilelocation = "TrainINKML_v3/expressmatch/101_alfonso.inkml"
	# 		tree = ET.parse(inkmlfilelocation)
			
	# 		root = tree.getroot() 

	# 		xcor,ycor = getxycor(root) 
	# 		symbol, indices = getSymbolIndices(root)
		   
	# 		inkmlfile = os.path.basename(inkmlfilelocation)
	# 		base = os.path.splitext(inkmlfile)[0]
	# 		lgfilepath = "../all_lg/"+base+".lg"

	# 		relations = get_spatialRelations(lgfilepath)
	# 		symbolpairs = getsymbolspairs(len(symbol))
		 

	# 		indices = sorted(indices, key=lambda x: x[0], reverse=False)

	# 		#print indices

	# 		#print symbolpairs		

	# 		for sympair in symbolpairs:
				

	# 			firstsymbol = sympair[0]
	# 			secondsymbol = sympair[1]

	# 			#print indices[firstsymbol], indices[secondsymbol]

	# 			stroke1 = indices[firstsymbol][-1]
	# 			stroke2 = indices[secondsymbol][-0]

	# 			if (stroke1, stroke2) in relations:
	# 				#print (stroke1, stroke2),  relations[(stroke1, stroke2)] 

	# 				ind = [indices[firstsymbol],indices[secondsymbol]]

	# 				#print ind
	# 				symbolpointsX,symbolpointsY = shiftPoints_SegmentStrokes(ind,xcor,ycor) 
	# 				symbolpointsX,symbolpointsY = normalizedPoints_SegmentStrokes(ind,symbolpointsX,symbolpointsY)  
	# 				bezierpointsX, bezierpointsY = getBezier_SegmentationStrokes(indices,symbolpointsX,symbolpointsY)    

	# 				grid=calculateShapeFeatures(bezierpointsX, bezierpointsY, numRadii, numTheta)

	# 				featurelist = []
	# 				for g in grid:
	# 					for val in g:
	# 						featurelist.append(val)

	# 				#print len(featurelist),  getRelationClass(relations[(stroke1, stroke2)])
	# 				#exit()
	# 				pcaTextX = pca.transform([featurelist])
	# 				testy = getRelationClass(relations[(stroke1, stroke2)])
					
	# 				val = spatialSVM.predict(pcaTextX[0])[0]

	# 				if (val == testy):
	# 					yes += 1

	# 				else:

	# 					no += 1




	# print "Accuracy:" + str((yes/float(yes+no)*100))		
	# print len(X), len (y)
	
	




def getRelationClass(relation):

	global relationToClass

	return relationToClass[relation]
	
				  
def normalizedPoints_SegmentStrokes(indices,symbolpointsX,symbolpointsY):

	xmin,ymin = 999999, 999999
	xmax,ymax = -999999, -999999

	for x,y in zip(symbolpointsX, symbolpointsY):   

		newx = []
		newy = []
		for x1,y1 in zip(x,y):
			for x2,y2 in zip(x1,y1):
				newx.append(x2)
				newy.append(y2)

		lxmin,lymin,lxmax,lymax = getMinMax(newx,newy)

		if lxmin<xmin:
			xmin=lxmin
		if lymin<ymin:
			ymin=lymin
		if lxmax>xmax:
			xmax=lxmax
		if lymax>ymax:
			ymax=lymax

	for x,y in zip(symbolpointsX, symbolpointsY):

		testx = []
		testy = []
		

		for x1,y1 in zip(x,y):

			tempy = y1

			if (ymax-ymin) == 0 :
				tempy[:] = [ 0 for y in tempy]
			else:
				tempy[:] = [ ((y - ymin) / (ymax-ymin)) for y in tempy]

			y1 = tempy


			tempx = x1

			if (xmax-xmin) == 0:
				tempx = [ 0 for xp in tempx]
			else:
				tempx[:] = [((xp- xmin) / (xmax-xmin))  for xp in tempx]

			x1 = tempx

	# plt.figure()       
	# for x,y in zip(symbolpointsX, symbolpointsY):     
	  
	#   for x1,y1 in zip(x,y):
			
	# 	plt.scatter(x1,y1)
	# plt.show() 
			
		

	return symbolpointsX,symbolpointsY

def shiftPoints_SegmentStrokes(indices,xcor,ycor):

	symbolpointsX = []
	symbolpointsY = []

	xmin = 9999999
	ymax = -999999

	
	for sym in range(len(indices)):
		
		newx = []
		newy = []

		
		for x in indices[sym]:
			for xp, yp in zip(xcor[x],ycor[x]):
				newx.append(xp)
				newy.append(yp)

		bbx,bby = getBoundingBox(newx,newy)

		if bbx[3] < xmin:        
			xmin = bbx[3]
		if bby[3] > ymax:
			ymax = bby[3]

	for sym in range(len(indices)):

		templistX = []
		templistY = []

						
		for x in indices[sym]:
		

			tempy = ycor[x]
			ty = [y - ymax for y in tempy]
			
			tempx = xcor[x]
			tx = [xp - xmin for xp in tempx]        
					
			
			templistX.append(tx)
			templistY.append(ty)
				
		
		symbolpointsX.append(templistX)
		symbolpointsY.append(templistY)

	
	# plt.figure()
	# for x,y in zip(symbolpointsX, symbolpointsY):     
	 
	#   for x1,y1 in zip(x,y):
	#       plt.scatter(x1,y1)
	# plt.show()
	

	return symbolpointsX,symbolpointsY    
							
#Function to calculate Euclidean distance between two points    
def calculateEuclidean(a, b):
	return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def calculateShapeFeatures(bezierPointsX, bezierPointsY, radialOffset, angleOffset):
	centerOfMass1 = [0, 0]
	centerOfMass2 = [0, 0]
	center = [0, 0]
	
	maxPoint = [0, 0]
	distance = 0.0
	
	bezierPointsX1 = []
	bezierPointsX2 = []
	bezierPointsY1 = []
	bezierPointsY2 = []
	
	for i in bezierPointsX[0]:
		bezierPointsX1 = bezierPointsX1 + i
	for i in bezierPointsX[1]:
		bezierPointsX2 = bezierPointsX2 + i
	for i in bezierPointsY[0]:
		bezierPointsY1 = bezierPointsY1 + i
	for i in bezierPointsY[1]:
		bezierPointsY2 = bezierPointsY2 + i
	
	
	for x, y in zip(bezierPointsX1, bezierPointsY1):
		centerOfMass1[0] += x
		centerOfMass1[1] += y
	
	centerOfMass1[0] /= len(bezierPointsX1)
	centerOfMass1[1] /= len(bezierPointsY1)
	
	for x, y in zip(bezierPointsX2, bezierPointsY2):
		centerOfMass2[0] += x
		centerOfMass2[1] += y
	
	centerOfMass2[0] /= len(bezierPointsX2)
	centerOfMass2[1] /= len(bezierPointsY2)
	
	center[0] = (centerOfMass1[0] + centerOfMass2[0]) / 2
	center[1] = (centerOfMass1[1] + centerOfMass2[1]) / 2
	
#	for i in range(len(bezierPointsX1)):
#		bezierPointsX1[i] -= center[0]
#	for i in range(len(bezierPointsX2)):
#		bezierPointsX2[i] -= center[0]
#	for i in range(len(bezierPointsY1)):
#		bezierPointsY1[i] -= center[1]
#	for i in range(len(bezierPointsY2)):
#		bezierPointsY2[i] -= center[1]

	for x1, y1, x2, y2 in zip(bezierPointsX1, bezierPointsY1, bezierPointsX2, bezierPointsY2):
		d = calculateEuclidean(center, [x1, y1])
		if d > distance:
			distance = d
			maxPoint = [x1, y1]
		
		d = calculateEuclidean(center, [x2, y2])
		if d > distance:
			distance = d
			maxPoint = [x2, y2]
		
	
	scatter(bezierPointsX1 + bezierPointsX2, bezierPointsY1 + bezierPointsY2)
	scatter(centerOfMass1[0], centerOfMass1[1], color = 'red')
	scatter(centerOfMass2[0], centerOfMass2[1], color = 'green')
	scatter(center[0], center[1], color = 'black')
	
	radius = 0.0
	nextRadius = 0.0
	deltaRadius = distance/radialOffset
	deltaTheta = (2 * np.pi) / angleOffset
	
	grid = []	
	
	for i in range(radialOffset):
		radius = nextRadius
		nextRadius += deltaRadius
		theta = 0.0
		nextTheta = 0.0
		row = []
		xcor = []
		ycor = []
		for j in range(angleOffset):
			theta = nextTheta
			nextTheta += deltaTheta
			numleftpoints = 0
			numrightpoints = 0
			xcor.append(nextRadius*cos(theta) + center[0])
			ycor.append(nextRadius*sin(theta) + center[1])
			for x, y in zip(bezierPointsX1, bezierPointsY1):
				pdist = calculateEuclidean(center, [x, y])
				ptheta = myAtan2(y, x)
				if (pdist >= radius and pdist < nextRadius and ptheta >= theta and ptheta < nextTheta):
					numleftpoints += 1
					
			for x, y in zip(bezierPointsX2, bezierPointsY2):
				pdist = calculateEuclidean(center, [x, y])
				ptheta = myAtan2(y, x)
				if (pdist >= radius and pdist < nextRadius and ptheta >= theta and ptheta < nextTheta):
					numrightpoints += 1
			
			if (numleftpoints > numrightpoints):
				row.append(-1)
			elif(numleftpoints == 0 and numrightpoints == 0):
				row.append(0)
			elif(numleftpoints <= numrightpoints):
				row.append(1)
		
		xcor.append(nextRadius*cos(nextTheta) + center[0])
		ycor.append(nextRadius*sin(nextTheta) + center[1])
		plt.plot(xcor, ycor)
		grid.append(row)

	#plt.show()
	
	return grid
	
				
			
#map results from arctan2 between 0 and 2*pi
def myAtan2(y, x)	:
	result = np.arctan2(y, x)
	if result >= 0:
		return result
	else:
		return 2*np.pi + result

def getBezier_SegmentationStrokes(indices,symbolpointsX,symbolpointsY):
	bezierpointsX = []
	bezierpointsY = []
	totalPoints = 34
	# create points for symbols using bezier 

	for x,y in zip(symbolpointsX, symbolpointsY):   

		newx = []
		newy = []

		itr = 1
		sums = 0
		length = len(x)    
		nTimes = int(totalPoints/length)

		templistX = []
		templistY = []

		for x1,y1 in zip(x,y):
			
			if(itr==length):
				nTimes = totalPoints - sums
		
			nPoints = len(x1)
			xPoints = np.array(x1)
			yPoints = np.array(y1)

			t = np.linspace(0.0, 1.0, nTimes)           
			polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t)
			 for i in range(0, nPoints)   ])
			
			
			templistX.append((np.dot(xPoints, polynomial_array)).reshape(-1,).tolist())
			templistY.append((np.dot(yPoints, polynomial_array)).reshape(-1,).tolist())
		   
			itr = itr + 1
			sums = sums + nTimes

		bezierpointsX.append(templistX)
		bezierpointsY.append(templistY)

#	plt.figure()
#	for x,y in zip(bezierpointsX, bezierpointsY):        
#	   
#		for x1,y1 in zip(x,y):
#			plt.scatter(x1,y1)
#			
#	plt.show()
	return bezierpointsX,bezierpointsY


def get_spatialRelations(lgfilepath):

	flgopen = open(lgfilepath,'rb')

	relations = {}

	spatial_relations = ["R","A","B","I","Sup","Sub"]

	for line in flgopen:
		splits =  line.split(",")

		if splits[0].startswith("E"):

			relation = splits[3].replace(" ","")
			if [e in relation for e in spatial_relations if e in relation]:

				relations[tuple((int(splits[1]),int(splits[2])))] = relation   

	flgopen.close()             

	return relations

def getBoundingBox(x,y):
	
	xmin,ymin,xmax,ymax = getMinMax(x,y)

	bbx = []
	bby = []
	bbx.append(xmin)
	bby.append(ymin)
	bbx.append(xmax)
	bby.append(ymin)
	bbx.append(xmax)
	bby.append(ymax)
	bbx.append(xmin)    
	bby.append(ymax)
	bbx.append(xmin)
	bby.append(ymin)

	return bbx,bby

def getMinMax(x,y):
	xmin = min(x)
	ymin = min(y)
	xmax = max(x)
	ymax = max(y)

	return xmin,ymin,xmax,ymax


def main():	
	createPoints()
	#calculateShapeFeatures(0, 1, 5, 5)
	#print getRelationClass("A")
	
	
		
if __name__ == '__main__':
	main()
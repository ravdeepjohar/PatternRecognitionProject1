from __future__ import division
import xml.etree.ElementTree as ET
import numpy as np
from sets import Set
from scipy.misc import comb
import matplotlib.pyplot as plt
import math, csv, os, random
import csv
import os
import pickle



test = []
train = []
validate = []

splitsSymbol = {}

vertices = []
graph = []
edges = []

indexFile = open("index.csv")
classDict = {}
classDict2 = {}

for line in indexFile:
	 classDict[line.split(",")[0]]= int((line.split(",")[1]).strip())

indexFile.close()
indexFile = open("index.csv")	
for line in indexFile:
     classDict2[int((line.split(",")[1]).strip())]= line.split(",")[0]
indexFile.close()
		
trainFeaturesX = []
trainFeaturesY = []
segmentTrainFeaturesX = []
segmentTrainFeaturesY = []
def main():
	X, y= [], []
	X2, y2 = [], []
	from sklearn import svm

	
	# feature_functions = []


	print "Start Feature extraction"
	
	fTest = open("test.txt", 'rb')
	os.chdir("TrainINKML_v3")
	'''fTrainXCSV = open('TrainDataX.csv', 'rb')
	fTrainYCSV = open('TrainDataY.csv', 'rb')
	fSegmentTrainXCSV = open('SegmentTrainDataX.csv', 'rb')
	fSegmentTrainYCSV = open('SegmentTrainDataY.csv', 'rb')
	
		
	
	for line in fTrainXCSV:
		temp = line.strip().split(',')
		temp2 = []
		for i in temp:
			temp2.append(float(i))
		trainFeaturesX.append(temp2)
	for line in fTrainYCSV:
		temp = line.strip().split(',')
		temp2 = []
		for i in temp:
			temp2.append(float(i))
		trainFeaturesY.append(temp2)
	for line in fSegmentTrainXCSV:
		temp = line.strip().split(',')
		temp2 = []
		for i in temp:
			temp2.append(float(i))
		segmentTrainFeaturesX.append(temp2)
	for line in fSegmentTrainYCSV:
		temp = line.strip().split(',')
		temp2 = []
		for i in temp:
			temp2.append(float(i))
		segmentTrainFeaturesY.append(temp2)
		
	symbolSVM = svm.SVC()
	symbolSVM = symbolSVM.fit(trainFeaturesX, trainFeaturesY)
	
	segmentSVM = svm.SVC()
	segmentSVM = segmentSVM.fit(segmentTrainFeaturesX, segmentTrainFeaturesY)
	'''
	symbolSVMFile = 	open('symbolSVM', 'rb')
	segmentSVMFile = open('segmentSVM', 'rb')
	
			
	segmentSVM = pickle.load(segmentSVMFile)
	symbolSVM = pickle.load(symbolSVMFile)
	
	symbolSVMFile.close()
	segmentSVMFile.close()
							
	
	
	#line = fTest.readline()
	
	#tree = ET.parse(line[:-1].strip())
	tree = ET.parse("expressmatch/65_alfonso.inkml")
	
	root = tree.getroot() 
	#tempx,tempy = extract_features(root)
	tempx2,labels = extract_features_Segmentation(root)
	
	
	#print indices2
	predictedlabels = []
	

	for tx,ty in zip(tempx2,labels):		
		predictedlabels.append(segmentSVM.predict(tx)[0])
	
	print predictedlabels
		
	symbolindices = getSymbolsPairs(predictedlabels)
	
	print symbolindices
		
	tempx, indices = test_extract_features(root,symbolindices)
	
	print len(tempx)
	classifiedSymbols = []	
	
	for tx in tempx:
		
		#predictedlabels.append(symbolSVM.predict(tx)[0])
		classifiedSymbols.append(symbolSVM.predict(tx)[0])
		#print symbolSVM.predict(tx)[0]
	print classifiedSymbols
	
	generate_lg_file(classifiedSymbols, symbolindices)
	fTest.close()
	
def generate_lg_file(symbolList, indices):
	if not os.path.exists('nnlg'):
		os.makedirs('nnlg')
		
	#symbolList, indices = getSymbolIndices(root)
	#symbolIndices = []	
	#for i in symbolList:
	#	symbolIndices.append(classDict[i])
		
	s = 0
	lg = []
	edges = []
	for i in range(len(symbolList)):
		sym = classDict2[symbolList[i]]
		strokes = indices[i]
		
		if(len(strokes) > 1):
			multiStrokes = []
			for i in range(len(strokes)):
				tempLG = []
				tempLG.append('N')
				tempLG.append(' ' + str(s))
				multiStrokes.append(s)
				s = s + 1
				tempLG.append(' ' + str(sym))
				tempLG.append(' 1.0')
				lg.append(tempLG)
			for i in range(len(multiStrokes)):
				for j in range(i, len(multiStrokes)):
					if i != j:
						tempEdge = []
						tempEdge.append('E')
						tempEdge.append(' ' + str(multiStrokes[i]))
						tempEdge.append(' ' + str(multiStrokes[j]))
						tempEdge.append(' *')
						tempEdge.append(' 1.0')
						edges.append(tempEdge)
						
						tempEdge = []
						tempEdge.append('E')
						tempEdge.append(' ' + str(multiStrokes[j]))
						tempEdge.append(' ' + str(multiStrokes[i]))
						tempEdge.append(' *')
						tempEdge.append(' 1.0')
						edges.append(tempEdge)
		else:
			tempLG = []
			tempLG.append('N')
			tempLG.append(' ' + str(s))
			s = s + 1
			tempLG.append(' ' + str(sym))
			tempLG.append(' 1.0')
			lg.append(tempLG)
			
	lgFile = open('nnlg/test.lg', 'wb')
	
	c = csv.writer(lgFile)
	
	for i in range(len(lg)):
		c.writerow(lg[i])
	
	#lgFile.write('\n')
	
	for i in range(len(edges)):
		c.writerow(edges[i])
	
	lgFile.close()

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

def getLabel(symbol):
	if symbol == ',':
		symbol = 'COMMA'
	return classDict[symbol]


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

def shiftPoints(indices,xcor,ycor):
	
	for sym in range(len(indices)):
		newx = []
		newy = []
		for x in indices[sym]:
			for xp, yp in zip(xcor[x],ycor[x]):
				newx.append(xp)
				newy.append(yp)

		bbx,bby = getBoundingBox(newx,newy)
		xmin = bbx[3]
		ymax = bby[3]

		testx = []
		testy = []
		for x in indices[sym]:
			tempy = ycor[x]
			tempy[:] = [y - ymax for y in tempy]
			ycor[x] = tempy


			tempx = xcor[x]
			tempx[:] = [xp - xmin for xp in tempx]
			xcor[x] = tempx

			for xp, yp in zip(xcor[x],ycor[x]):
				testx.append(xp)
				testy.append(yp)

		# plt.figure()
		# plt.scatter(testx,testy)
		# plt.show()

	return xcor,ycor

	

	
def normalizedPoints(indices,xcor,ycor):

	for sym in range(len(indices)):
		newx = []
		newy = []

		for x in indices[sym]:
			for xp, yp in zip(xcor[x],ycor[x]):
				newx.append(xp)
				newy.append(yp)

		xmin,ymin,xmax,ymax = getMinMax(newx,newy)
		
		testx = []
		testy = []
		
		for x in indices[sym]:

			tempy = ycor[x]

			if (ymax-ymin) == 0 :
				tempy[:] = [ 0 for y in tempy]
			else:
				tempy[:] = [ ((y - ymin) / (ymax-ymin)) for y in tempy]

			ycor[x] = tempy


			tempx = xcor[x]

			if (xmax-xmin) == 0:
				tempx = [ 0 for xp in tempx]
			else:
				tempx[:] = [((xp- xmin) / (xmax-xmin))  for xp in tempx]

			xcor[x] = tempx

			
			for xp, yp in zip(xcor[x],ycor[x]):
				testx.append(xp)
				testy.append(yp)

		# plt.figure()
		# plt.scatter(testx,testy)
		# plt.show()

	return xcor,ycor

def bernstein_poly(i, n, t):
		"""
		 The Bernstein polynomial of n, i as a function of t
		"""
		return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def getBezier(indices,xcor,ycor):
	bezierpoints = []
	totalPoints = 34
	# create points for symbols using bezier 
	
	for sym in range(len(indices)):
		itr = 1
		sums = 0
		length = len(indices[sym])    
		nTimes = int(totalPoints/length)
			
		temp = []    
		
		for x in indices[sym]:
			
			if(itr==length):
				nTimes = totalPoints - sums
			
			nPoints = len(xcor[x])
			xPoints = np.array(xcor[x])
			yPoints = np.array(ycor[x])

			if len(xPoints) > 6:
				index= [0,1,len(xPoints)-2,len(xPoints)-1]
				xPoints = np.delete(xPoints,index)
				yPoints = np.delete(yPoints,index)
				nPoints = nPoints - 4

			t = np.linspace(0.0, 1.0, nTimes)           
			polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t)
			 for i in range(0, nPoints)   ])
			
			xvals = (np.dot(xPoints, polynomial_array)).reshape(-1,).tolist()
			yvals = (np.dot(yPoints, polynomial_array)).reshape(-1,).tolist()
		   
			cor = []
			for x,y in zip(xvals,yvals):           
				cor.append([x,y])        
			
			temp.append(cor)
		   
			#plt.plot(xvals, yvals)

			itr = itr + 1
			sums = sums + nTimes
		#plt.show() 
		bezierpoints.append(temp)
	
	return bezierpoints

def extract_features_Segmentation(root):

	xcor,ycor = getxycor(root) 
	symbol, indices = getSymbolIndices(root)

	# for x,y in zip(symbol,indices):
	# 	print x, y 

	indices, labels = getstrokepairs(indices)

	xcor,ycor = shiftPoints(indices,xcor,ycor)
	xcor,ycor = normalizedPoints(indices,xcor,ycor)
	bezierpoints = getBezier(indices,xcor,ycor)

	features = []
	

	for sym, j in zip(labels, range(len(labels))):

		feature = []

		for x in bezierpoints[j]:
			for xp in x:
				feature.append(xp[0])

		for x in bezierpoints[j]:
			for xp in x:
				feature.append(xp[1])

		features.append(feature)
		
	#print labels, indices

	# for x,y in zip(features,labels):
	# 	print x ,y 
	# 	print " "

	# print " "


	return features, labels
	
def getSymbolsPairs(labels):
	#indices = [1, 0, 1, 0, 0, 1, 1, 0, 1]
	
	numberofstrokes = len(labels)
	strokes = []
	symbolList = []
	visited = []
	for i in range(numberofstrokes):
		strokes.append([i, i+1])
	
	#print strokes[0][0]
	for i in range(numberofstrokes):
		if labels[i] == 0:
			if strokes[i][0] not in visited:
				symbolList.append([strokes[i][0]])
				visited.append(strokes[i][0])
			if strokes[i][1] not in visited:
				symbolList.append([strokes[i][1]])
				visited.append(strokes[i][1])
		elif labels[i] == 1:
			if len(symbolList) > 0:
				temp = symbolList.pop()
				temp.append(strokes[i][1])
				visited.append(strokes[i][1])
			else:
				temp = []
				temp.append(strokes[i][0])
				temp.append(strokes[i][1])
				visited.append(strokes[i][0])
				visited.append(strokes[i][1])
			
			symbolList.append(temp)
			
			
	
	return symbolList
	
def getstrokepairs(indices):
	#print indices
	numberofstrokes = 0

	for x in indices:
		numberofstrokes += len(x)

	strokepairs = dict()
	sp = []
	labels = []
	

	for i in range(numberofstrokes-1):

		strokepairs[tuple([i,i+1])] = 0 
		sp.append([i,i+1])
		labels.append(0)
		
	

	
	for x in indices:
		if(len(x)>1):
			for p in range(len(x)-1):
				for l in range(len(sp)):
					if([x[p],x[p+1]]==sp[l]):
						 labels[l]= 1
	#print labels
	#print sp
	'''newindices = []
	newsymbols = []
	for x in strokepairs:
		newindices.append(list(x))
		newsymbols.append(strokepairs[x])'''


	return sp,labels#newindices , newsymbols

def test_extract_features(root,indices):

	xcor,ycor = getxycor(root) 
	xcor,ycor = shiftPoints(indices,xcor,ycor)
	xcor,ycor = normalizedPoints(indices,xcor,ycor)
	bezierpoints = getBezier(indices,xcor,ycor)

	features = []
	

	for  j in range(len(indices)):

		feature = []

		for x in bezierpoints[j]:
			for xp in x:
				feature.append(xp[0])

		for x in bezierpoints[j]:
			for xp in x:
				feature.append(xp[1])

		features.append(feature)
		
		

	return features, indices


def extract_features(root):

	xcor,ycor = getxycor(root) 
	symbol, indices = getSymbolIndices(root)
	xcor,ycor = shiftPoints(symbol,indices,xcor,ycor)
	xcor,ycor = normalizedPoints(symbol,indices,xcor,ycor)
	bezierpoints = getBezier(symbol,indices,xcor,ycor)

	features = []
	labels = []

	for sym, j in zip(symbol, range(len(symbol))):

		feature = []

		for x in bezierpoints[j]:
			for xp in x:
				feature.append(xp[0])

		for x in bezierpoints[j]:
			for xp in x:
				feature.append(xp[1])

		features.append(feature)
		
		labels.append(getLabel(sym))

	return features, labels, indices

def getMinMax(x,y):
	xmin = min(x)
	ymin = min(y)
	xmax = max(x)
	ymax = max(y)

	return xmin,ymin,xmax,ymax

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


	
def getCenter(x,y):
	
	xmin,ymin,xmax,ymax = getMinMax(x,y)

	bbx,bby = getBoundingBox(x,y)
	return [(((xmax-xmin)/2.0)+xmin),(((ymax-ymin)/2.0)+ymin)]

def distance(a,b):

	return sum([(a[i]-b[i])**2 for i in range(len(a))])**0.5

def boundingbox(xcor, ycor):

	vertices = []
	for ind in range(len(xcor)):
		x = xcor[ind]
		y = ycor[ind]
		vertices.append(Vertex(ind))
		vertices[ind].center = getCenter(x,y)
		bbx,bby = getBoundingBox(x,y)
		#plt.plot(x,y)
		#plt.scatter(vertices[ind].center[0],vertices[ind].center[1])

	for v in range(len(xcor)):
		temp = []
		for i in range(len(xcor)):
			temp.append(None)
		graph.append(temp)

	for ind in range(len(xcor)):

		for ind2 in range(len(xcor)):
			if ind2 != ind:
				dist = distance(vertices[ind].center,vertices[ind2].center)
				vertices[ind].adjacent.append([vertices[ind2],dist])
				# #connect(vertices[ind], vertices[ind2], dist)
				graph[ind][ind2] = dist
	
	
	'''mst = prims(vertices)

	for x in mst:
		print x 

	print len(mst)'''

	pltx = []
	plty = []

	for v in mst:

		v1 = v[0]
		v2 = v[1]
		
		pltx.append(v1.center[0])
		plty.append(v1.center[1])

		pltx.append(v2.center[0])
		plty.append(v2.center[1])

	# for v in mst:

	# 	v1 = v[0]
	# 	v2 = v[1]
		
	# 	pltx.append(vertices[v1].center[0])
	# 	plty.append(vertices[v1].center[1])

	# 	pltx.append(vertices[v2].center[0])
	# 	plty.append(vertices[v2].center[1])


	#plt.plot(pltx,plty)


	#plt.show()

	return bbx,bby





if __name__ == '__main__':
	main()

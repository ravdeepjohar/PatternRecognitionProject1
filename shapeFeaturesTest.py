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
import csv


indexFile = open("index.csv")
classDict = {}
classDict2 = {}
numRadii = 10
numTheta = 15

relationToClass = {'R':1,'A':2,'B':2,'I':3,'Sup':4,'Sub':5}
reverseRelation = {1:'R', 2:'A', 2:'B', 3:'I', 4:'Sup', 5:'Sub'}

for line in indexFile:
	classDict[line.split(",")[0]]= int((line.split(",")[1]).strip())
	classDict2[int((line.split(",")[1]).strip())]= line.split(",")[0]
		
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

def shiftPoints_SegmentStrokes(indices,xcor,ycor):

	symbolpointsX = []
	symbolpointsY = []

	
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

	

	# for x,y in zip(symbolpointsX, symbolpointsY):		
	# 	plt.figure()
	# 	for x1,y1 in zip(x,y):
	# 		plt.scatter(x1,y1)
	# 	plt.show()


	return symbolpointsX,symbolpointsY

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

def normalizedPoints_SegmentStrokes(indices,symbolpointsX,symbolpointsY):

	for x,y in zip(symbolpointsX, symbolpointsY):	

		newx = []
		newy = []
		for x1,y1 in zip(x,y):
			for x2,y2 in zip(x1,y1):
				newx.append(x2)
				newy.append(y2)

		xmin,ymin,xmax,ymax = getMinMax(newx,newy)


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

			
	# for x,y in zip(symbolpointsX, symbolpointsY):		
	# 	#plt.figure()
	# 	for x1,y1 in zip(x,y):
			
	# 		#plt.scatter(x1,y1)
	# 	#plt.show()	
			
		

	return symbolpointsX,symbolpointsY

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


	'''for x,y in zip(bezierpointsX, bezierpointsY):		
		plt.figure()
		for x1,y1 in zip(x,y):
			plt.scatter(x1,y1)
			
		plt.show()'''
	
	return bezierpointsX,bezierpointsY

def calculate_bounding_box_distance(x, y):
	centers = []
	for x1, y1 in zip(x, y):
		
		centers.append(getCenter(x1, y1))
		
	'''plt.figure()
	for x1,y1 in zip(x,y):
		plt.scatter(x1,y1)
		bla1,bla2 = getCenter(x1, y1)
		plt.scatter(bla1,bla2, color = 'r')
			
	plt.show()'''
	d = distance(centers[0], centers[1])
	
	return d
	
def calculate_average_center_distance(x, y):
	centers = []
	for x1, y1 in zip(x, y):
		
		centers.append(average(x1, y1))
				

	d = distance(centers[0], centers[1])
	
	return d

def calculate_max_distance(x, y):
	maxDistance = -1.0
	for x1, y1 in zip(x[0], y[0]):
		for x2, y2 in zip(x[1], y[1]):
			d = distance([x1, y1], [x2, y2])
			if d > maxDistance:
				maxDistance = d	
	return maxDistance


def calculate_two_stroke_distance(x, y):
	first_stroke_end_x = x[0][len(x[0]) - 1]
	first_stroke_end_y = x[0][len(x[0]) - 1]
	second_stroke_start_x = x[1][0]
	second_stroke_start_y = y[1][0]
	
	
	'''plt.figure()
	for x1,y1 in zip(x,y):
		plt.scatter(x1,y1)
		bla1,bla2 = getCenter(x1, y1)
		plt.scatter(first_stroke_end_x,first_stroke_end_y, color = 'r')
		plt.scatter(second_stroke_start_x, second_stroke_start_y, color = 'r')
			
	plt.show()'''
	
	offset = second_stroke_start_x - first_stroke_end_x
	return offset

def calculate_bounding_box_vertical_distance(x, y):
	centers = []
	for x1, y1 in zip(x, y):
		
		centers.append(getCenter(x1, y1))
		
	'''plt.figure()
	for x1,y1 in zip(x,y):
		plt.scatter(x1,y1)
		bla1,bla2 = getCenter(x1, y1)
		plt.scatter(bla1,bla2, color = 'r')
			
	plt.show()'''
	d = centers[1][1] - centers[0][1]
	
	return d
	
def calculate_writing_slope(x, y):
	first_stroke_end_x = x[0][len(x[0]) - 1]
	first_stroke_end_y = x[0][len(x[0]) - 1]
	second_stroke_start_x = x[1][0]
	second_stroke_start_y = y[1][0]
	
	vector = [second_stroke_start_x - first_stroke_end_x, second_stroke_start_y-first_stroke_end_y]
	horizontal = [1.0, 0.0]
	
	dotProd = vector[0] * horizontal[0] + vector[1] * horizontal[1]
	magHorizontal = math.sqrt(horizontal[0]**2 + horizontal[1]**2)
	magVector = math.sqrt(vector[0]**2 + vector[1]**2)
	
	denom = (magHorizontal*magVector)
	if denom == 0:
		denom = 0.001
	cosTheta = dotProd/denom
	theta = math.acos(cosTheta)
	
	'''plt.figure()
	for x1,y1 in zip(x,y):
		plt.scatter(x1,y1)
		bla1,bla2 = getCenter(x1, y1)
		plt.scatter(first_stroke_end_x,first_stroke_end_y, color = 'r')
		plt.scatter(second_stroke_start_x, second_stroke_start_y, color = 'r')
			
	plt.show()'''
	
	return theta
	
def calculate_writing_curvature(x, y):
	first_stroke_start_x = x[0][0]
	first_stroke_start_y = y[0][0]
	first_stroke_end_x = x[0][len(x[0]) - 1]
	first_stroke_end_y = x[0][len(x[0]) - 1]
	second_stroke_start_x = x[1][0]
	second_stroke_start_y = y[1][0]
	second_stroke_end_x = x[1][len(x[1]) - 1]
	second_stroke_end_y = y[1][len(y[1]) - 1]
	
	first_stroke_vector = [first_stroke_end_x - first_stroke_start_x, first_stroke_end_y - first_stroke_start_y]
	second_stroke_vector = [second_stroke_end_x - second_stroke_start_x, second_stroke_end_y - second_stroke_start_y]
	
	dotProd = first_stroke_vector[0] * second_stroke_vector[0] + first_stroke_vector[1] * second_stroke_vector[1]
	first_vector_magnitude = math.sqrt(first_stroke_vector[0]**2 + first_stroke_vector[1]**2)
	second_vector_magnitude = math.sqrt(second_stroke_vector[0]**2 + second_stroke_vector[1]**2)
	
	denom = (first_vector_magnitude*second_vector_magnitude)
	if denom == 0:
		denom = 0.001
	cosTheta = dotProd/denom
	
	#cosTheta should be in range [-1, 1]
	if(cosTheta >= 1.0):
		cosTheta = 0.9999
	elif(cosTheta <= -1.0):
		cosTheta = -0.9999
	theta = math.acos(cosTheta)
	
	
	return theta

def calculate_bounding_box_size_difference(x, y):
	sizes = []
	for x1, y1 in zip(x, y):
		boundinBoxX, boundingBoxY = getBoundingBox(x1, y1)
		width = distance([boundinBoxX[0], boundingBoxY[0]], [boundinBoxX[1], boundingBoxY[1]])
		height = distance([boundinBoxX[1], boundingBoxY[1]], [boundinBoxX[2], boundingBoxY[2]])
		sizes.append(width*height)
	
	diff = sizes[1] - sizes[0]
	return diff



def extract_features_Segmentation(root):

	xcor,ycor = getxycor(root) 
	symbol, indices = getSymbolIndices(root)

	#for x,y in zip(symbol,indices):
		#print x, y 

	indices, labels = getstrokepairs(indices)
	
	#print indices

	xcor,ycor = shiftPoints_SegmentStrokes(indices,xcor,ycor)
	xcor,ycor = normalizedPoints_SegmentStrokes(indices,xcor,ycor)
	xcor,ycor = getBezier_SegmentationStrokes(indices,xcor,ycor)
	
	features = []
	
		
	for x, y in zip(xcor, ycor):
		feature = []
		feature.append(calculate_bounding_box_distance(x, y))
		feature.append(calculate_average_center_distance(x,y))
		feature.append(calculate_max_distance(x, y))
		feature.append(calculate_two_stroke_distance(x, y))
		feature.append(calculate_bounding_box_vertical_distance(x, y))
		feature.append(calculate_writing_slope(x, y))
		feature.append(calculate_writing_curvature(x, y))
		feature.append(calculate_bounding_box_size_difference(x, y))

		features.append(feature)

	

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


def calculate_aspect_ratios(bezierpoints):
	aspect_ratio = []
		
	for i in range(len(bezierpoints)):
		max_width = 0.0
		max_height = 0.0
		min_width = 0.0
		min_height = 0.0
		
		for j in range(len(bezierpoints[i])):
			for coord in bezierpoints[i][j]:
				if(max_width == 0.0):
					max_width = coord[0]
				elif(max_width < coord[0]):
					max_width = coord[0]
				
				if(max_height == 0.0):
					max_height = coord[1]
				elif(max_height < coord[1]):
					max_height = coord[1]
				
				if(min_width == 0.0):
					min_width = coord[0]
				elif(min_width > coord[0]):
					min_width = coord[0]
				
				if(min_height == 0.0):
					min_height = coord[1]
				elif(min_height > coord[1]):
					min_height = coord[1]
		
		if(max_height - min_height > 0):
			aspect = (max_width - min_width)/(max_height - min_height)
		else:
			aspect = (max_width - min_width)
		aspect_ratio.append(aspect)
	return aspect_ratio

def calculate_covariance(bezierpoints):
	covariance = []
	for i in range(len(bezierpoints)):
		tempYList = []
		tempXList = []
		for j in range(len(bezierpoints[i])):
			for coord in bezierpoints[i][j]:
				tempXList.append(coord[0])
				tempYList.append(coord[1])
		mat = []
		xMean = 0.0
		yMean = 0.0
		for j in range(len(tempXList)):
			xMean = tempXList[j]
			yMean = tempYList[j]
			mat.append([tempXList[j], tempYList[j]])
		
		xMean = xMean / len(tempXList)
		yMean = yMean / len(tempYList)
		
		mat = np.matrix(mat)
		mean = np.matrix([xMean, yMean])
		
		covarianceMatrix = np.matrix([[0.0, 0.0], [0.0, 0.0]])
		
		covarianceMatrix = (mat[0] - mean).T * (mat[0] - mean)
		for k in range(len(mat)):
			covarianceMatrix = covarianceMatrix + (mat[k] - mean).T * (mat[k] - mean)
		
		covariance.append(covarianceMatrix)
	return covariance
	
def calculate_vicinity_slope(bezierpoints):
	
	vicinity_slope = []
	
	for i in range(len(bezierpoints)):
		tempList = []
		for j in range(len(bezierpoints[i])):
			for coord in bezierpoints[i][j]:
				tempList.append(coord)
		sumTheta = 0.0
		for k in range(3, len(tempList)-3):
			horizontal = [1.0, 0.0]
			vec = [tempList[k+3][0] - tempList[k-3][0], tempList[k+3][1] - tempList[k-3][1]]
			magHorizonal = np.sqrt(horizontal[0] * horizontal[0] + horizontal[1] * horizontal[1])
			magVec = np.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
			
			denom = (magVec * magHorizonal)
			if denom <= 0:
				denom = 0.0001
			cosTheta = (horizontal[0] * vec[0] + horizontal[1] * vec[1])/denom
			
			if(cosTheta >= 1.0):
				cosTheta = 0.999
			if(cosTheta <= -1.0):
				cosTheta = -0.999
			
			theta = math.acos(cosTheta)
			sumTheta += theta
		
		vicinity_slope.append(sumTheta)
	return vicinity_slope
	
def calculate_curvature(bezierpoints):
	curvature = []
	
	for i in range(len(bezierpoints)):
		tempList = []
		for j in range(len(bezierpoints[i])):
			for coord in bezierpoints[i][j]:
				tempList.append(coord)
		sumTheta = 0.0
		for k in range(3, len(tempList) - 3):
			vec1 = [tempList[k-3][0] - tempList[k][0], tempList[k-3][1] - tempList[k][1]]
			mag1 = np.sqrt(vec1[0] * vec1[0] + vec1[1] * vec1[1])
			vec2 = [tempList[k][0] - tempList[k+3][0], tempList[k][1] - tempList[k+3][1]]
			mag2 = np.sqrt(vec2[0] * vec2[0] + vec2[1] * vec2[1])
			

			denom = (mag1 * mag2)
			if denom <= 0:
				denom = 0.0001
			cosTheta = (vec1[0] * vec2[0] + vec1[1] * vec2[1])/denom
			if(cosTheta >= 1.0):
				cosTheta = 0.999
			if(cosTheta <= -1.0):
				cosTheta = -0.999
				
			
			theta = math.acos(cosTheta)
			sumTheta += theta
			
		
		curvature.append(sumTheta)
	return curvature

def calculate_number_of_strokes(indices):
	numStrokes = []
	for i in range(len(indices)):
		numStrokes.append(len(indices[i]))
	return numStrokes

def extract_features(root,indices):

	xcor,ycor = getxycor(root) 
	symbol, _ = getSymbolIndices(root)
	shiftedxcor,shiftedycor = shiftPoints(indices,xcor,ycor)
	normalizedxcor,normalizedycor = normalizedPoints(indices,shiftedxcor,shiftedycor)
	bezierpoints = getBezier(indices,normalizedxcor,normalizedycor)
	
	
	features = []
	labels = []
	
	#symbol features
	
	aspect_ratios = calculate_aspect_ratios(bezierpoints)
	covariance = calculate_covariance(bezierpoints)
	vicinity_slope = calculate_vicinity_slope(bezierpoints)
	curvature = calculate_curvature(bezierpoints)
	numStrokes = calculate_number_of_strokes(indices)
	

	for sym, j in zip(indices, range(len(indices))):

		feature = []

		for x in bezierpoints[j]:
			for xp in x:
				feature.append(xp[0])

		for x in bezierpoints[j]:
			for xp in x:
				feature.append(xp[1])

		feature.append(aspect_ratios[j])
		feature.append(vicinity_slope[j])
		feature.append(curvature[j])
		feature.append(numStrokes[j])

		tempCovList = covariance[j].reshape(-1).tolist()

		for k in range(len(tempCovList[0])):
			
			feature.append(tempCovList[0][k])
		
		features.append(feature)
		
		#labels.append(getLabel(sym))

	return features, _


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

def average(x,y):
	
	return sum(x1 for x1 in x)/len(x), sum(y1 for y1 in y)/len(y)

	

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


def createPoints():
	
	fTest = open("test2.txt",'rb')



	os.chdir("TrainINKML_v3")

	symbolSVMFile = open('symbolSVMAll', 'rb')
	segmentSVMFile = open('segmentSVMAll', 'rb')
	
	segmentSVM = pickle.load(segmentSVMFile)
	symbolSVM = pickle.load(symbolSVMFile)
	
	symbolSVMFile.close()
	segmentSVMFile.close()
	
	#X3, y3 = [], []
	testX, testy = [], []

	pca = pickle.load(open( "spatialpca.p", "rb" ))

	
	yes, no = 0.0, 0.0 	

	spatialSVM = pickle.load(open( "spatialsvm.p", "rb" ) )

	for line in fTest:

		inkmlfilelocation =  line[:-1].strip()
		if inkmlfilelocation.endswith(".inkml"):
			#inkmlfilelocation = "TrainINKML_v3/expressmatch/101_alfonso.inkml"
			tree = ET.parse(inkmlfilelocation)
			
			root = tree.getroot() 

			xcor,ycor = getxycor(root) 
			symbol, indices = getSymbolIndices(root)
			
			
			tempx2,labels = extract_features_Segmentation(root)
		
		
			#print indices2
			predictedlabels = []
			
			#print labels
			for tx,ty in zip(tempx2,labels):		
				predictedlabels.append(segmentSVM.predict(tx)[0])
			
			#print predictedlabels
	
			for l1, l2 in zip(labels, predictedlabels):
				if (l1 == l2):
					yes += 1
	
				else:
	
					no += 1
	
	
				
			symbolindices = getSymbolsPairs(predictedlabels)
			
			#print symbolindices
				
			tempx, labels = extract_features(root,symbolindices)
			
			#print len(tempx)
			classifiedSymbols = []	
			
			for tx in tempx:
				classifiedSymbols.append(symbolSVM.predict(tx)[0])
				
	
			#print classifiedSymbols
			
			#inkmlfile = os.path.basename(inkmlfilelocation)
			#base = os.path.splitext(inkmlfile)[0]
			#lgfilepath = "../all_lg/"+base+".lg"

			#relations = get_spatialRelations(lgfilepath)
			symbolpairs = getsymbolspairs(len(symbolindices))
		 

			indices = sorted(symbolindices, key=lambda x: x[0], reverse=False)

			#print indices

			#print symbolpairs		
			vals = []
			for sympair in symbolpairs:
				
				
				firstsymbol = sympair[0]
				secondsymbol = sympair[1]

				#print indices[firstsymbol], indices[secondsymbol]

				stroke1 = indices[firstsymbol][-1]
				stroke2 = indices[secondsymbol][-0]

				#if (stroke1, stroke2) in relations:
				#print (stroke1, stroke2),  relations[(stroke1, stroke2)] 

				ind = [indices[firstsymbol],indices[secondsymbol]]

				#print ind
				symbolpointsX,symbolpointsY = shiftPoints_ParserStrokes(ind,xcor,ycor) 
				symbolpointsX,symbolpointsY = normalizedPoints_ParserStrokes(ind,symbolpointsX,symbolpointsY)  
				bezierpointsX, bezierpointsY = getBezier_ParserStrokes(indices,symbolpointsX,symbolpointsY)    

				grid=calculateShapeFeatures(bezierpointsX, bezierpointsY, numRadii, numTheta)

				#print len(featurelist),  getRelationClass(relations[(stroke1, stroke2)])
				#exit()
				pcaTextX = pca.transform([grid])
				#testy = getRelationClass(relations[(stroke1, stroke2)])
				
				val = spatialSVM.predict(pcaTextX[0])[0]
				#print (stroke1, stroke2),  relations[(stroke1, stroke2)], val
				vals.append(val)
					
			generate_LG(symbolindices, symbolpairs, classifiedSymbols, vals)			
	
	
	

def generate_LG(symbolindices, symbolpairs, classifiedsymbols, vals):
	
	if not os.path.exists('nnlg'):
		os.makedirs('nnlg')	
	
	lg = []
	#print vals
	#print symbolpairs
	print symbolindices
#	for i in range(len(classifiedsymbols)):
#		print classDict2[classifiedsymbols[i]], symbolindices[i]
	edges = []
	starsymbols = []
	
	for i, j in zip(symbolindices, classifiedsymbols):
		#print i, classDict2[j]
		for stroke in i:
			templg = []
			templg.append('N')
			templg.append(' ' + str(stroke))
			templg.append(' ' + str(classDict2[j]))
			templg.append(' 1.0')
			lg.append(templg)
			
	
	for sympair, val in zip(symbolpairs, vals):
		sym1 = symbolindices[sympair[0]]
		sym2 = symbolindices[sympair[1]]
		
		if(sympair[0] not in starsymbols):
			starsymbols.append(sympair[0])
			for i in range(len(sym1)):
				for j in range(i, len(sym1)):
					if i != j:
						tempEdge = []
						tempEdge.append('E')
						tempEdge.append(' ' + str(sym1[i]))
						tempEdge.append(' ' + str(sym1[j]))
						tempEdge.append(' *')
						tempEdge.append(' 1.0')
						edges.append(tempEdge)
						
						tempEdge = []
						tempEdge.append('E')
						tempEdge.append(' ' + str(sym1[j]))
						tempEdge.append(' ' + str(sym1[i]))
						tempEdge.append(' *')
						tempEdge.append(' 1.0')
						edges.append(tempEdge)
		if(sympair[1] not in starsymbols):
			starsymbols.append(sympair[1])
			for i in range(len(sym2)):
				for j in range(i, len(sym2)):
					if i != j:
						tempEdge = []
						tempEdge.append('E')
						tempEdge.append(' ' + str(sym2[i]))
						tempEdge.append(' ' + str(sym2[j]))
						tempEdge.append(' *')
						tempEdge.append(' 1.0')
						edges.append(tempEdge)
						
						tempEdge = []
						tempEdge.append('E')
						tempEdge.append(' ' + str(sym2[j]))
						tempEdge.append(' ' + str(sym2[i]))
						tempEdge.append(' *')
						tempEdge.append(' 1.0')
						edges.append(tempEdge)
						
		print sym1, sym2, val
		for s1 in sym1:
			for s2 in sym2:
				tempEdge = []
				tempEdge.append('E')
				tempEdge.append(' ' + str(s1))
				tempEdge.append(' ' + str(s2))
				if val == 1:
					tempEdge.append(' R')
				elif val == 2:
					tempEdge.append(' B')
				elif val == 3:
					tempEdge.append(' I')
				elif val == 4:
					tempEdge.append(' Sup')
				elif val == 5:
					tempEdge.append(' Sub')
				tempEdge.append(' 1.0')
				
				if val < 6:
					edges.append(tempEdge)
		
	edges = sorted(edges, key=lambda x: int(x[1]), reverse = False)
	
	lgFile = open('nnlg/test.lg', 'wb')
	
	c = csv.writer(lgFile)
	
	for i in range(len(lg)):
		c.writerow(lg[i])
	for i in range(len(edges)):
		c.writerow(edges[i])
	
	lgFile.close()
			
		
#	for i in symbolpairs:
#		sym1 = indices[i[0]]
#		sym2 = indices[i[1]]
#		print sym1, sym2
		

def getLabel(symbol):
	if symbol == ',':
		symbol = 'COMMA'
	return classDict[symbol]

def getRelationClass(relation):

	global relationToClass

	return relationToClass[relation]
	
				  
def normalizedPoints_ParserStrokes(indices,symbolpointsX,symbolpointsY):

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

def shiftPoints_ParserStrokes(indices,xcor,ycor):

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
		
	
	# scatter(bezierPointsX1 + bezierPointsX2, bezierPointsY1 + bezierPointsY2)
	# scatter(centerOfMass1[0], centerOfMass1[1], color = 'red')
	# scatter(centerOfMass2[0], centerOfMass2[1], color = 'green')
	# scatter(center[0], center[1], color = 'black')
	
	radius = 0.0
	nextRadius = 0.0
	deltaRadius = distance/radialOffset
	deltaTheta = (2 * np.pi) / angleOffset
	
	grid = []	
	
	#print distance
	#print deltaRadius
	#print deltaTheta * 180 / (2*np.pi)	
	
	plotTheta = []
	plotRadius = []
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
			plotTheta.append(theta)
			plotRadius.append(radius)
			xcor.append(nextRadius*cos(theta) + center[0])
			ycor.append(nextRadius*sin(theta) + center[1])
			for x, y in zip(bezierPointsX1, bezierPointsY1):
				pdist = calculateEuclidean(center, [x, y])
				ptheta = myAtan2(y - center[1], x - center[0])
				if (pdist >= radius and pdist < nextRadius and ptheta >= theta and ptheta < nextTheta):
					numleftpoints += 1
					
			for x, y in zip(bezierPointsX2, bezierPointsY2):
				pdist = calculateEuclidean(center, [x, y])
				ptheta = myAtan2(y - center[1], x - center[0])
				if (pdist >= radius and pdist < nextRadius and ptheta >= theta and ptheta < nextTheta):
					numrightpoints += 1
			
			if (numleftpoints > numrightpoints):
				grid.append(-1)
			elif(numleftpoints == 0 and numrightpoints == 0):
				grid.append(0)
			elif(numleftpoints <= numrightpoints):
				grid.append(1)
	#print grid
#-------------------comment from here to disable plotting-------------------------
	# 	xcor.append(nextRadius*cos(nextTheta) + center[0])
	# 	ycor.append(nextRadius*sin(nextTheta) + center[1])
	# 	plt.plot(xcor, ycor)
	# 	#grid.append(row)
	
		
	# theta = 0.0
	# for i in range(angleOffset):
	# 	plt.plot([center[0], distance*cos(deltaTheta*i) + center[0]], [center[1], distance*sin(deltaTheta*i)+center[1]])
		
	
	# fig = plt.figure()
	# ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
	# color = ['lightgreen','white','red']
	# for i in xrange(angleOffset * radialOffset):
	# 	c = 'white'
	# 	if grid[i] == 1:
	# 		c = color[2]
	# 	elif grid[i] == -1:
	# 		c = color[0]
   
	# 	ax.bar(i * 2 * np.pi / angleOffset, 1, width=2 * np.pi / angleOffset, bottom=i / angleOffset,
	#            color=c, edgecolor = c)
	# plt.ylim(0,radialOffset)
	# ax.set_yticks([])
	# plt.show()
		
	return grid
	
				
			
#map results from arctan2 between 0 and 2*pi
def myAtan2(y, x)	:
	result = np.arctan2(y, x)
	if result >= 0:
		return result
	else:
		return 2*np.pi + result

def getBezier_ParserStrokes(indices,symbolpointsX,symbolpointsY):
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
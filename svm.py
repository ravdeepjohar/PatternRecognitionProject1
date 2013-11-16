from __future__ import division
import xml.etree.ElementTree as ET
import numpy as np
from scipy.misc import comb
import matplotlib.pyplot as plt
import math, csv, os, random

test = []
train = []
validate = []

splitsSymbol = {}

indexFile = open("index.csv")
classDict = {}

for line in indexFile:
	 classDict[line.split(",")[0]]= int((line.split(",")[1]).strip())


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

def getBezier(symbols,indices,xcor,ycor):
	bezierpoints = []
	totalPoints = 15
	# create points for symbols using bezier 
	
	for sym in range(len(symbols)):
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
		   
			plt.plot(xvals, yvals)

			itr = itr + 1
			sums = sums + nTimes
			  
		bezierpoints.append(temp)
	
	return bezierpoints


def main():
	os.chdir("TrainINKML_v3")
	from sklearn import svm

	X, y= [], []
	testX, testy = [], []
	# feature_functions = []

	for user in os.listdir("."):
		for inkmls in os.listdir(user):
			if inkmls.endswith(".inkml"):
				filelocation = user+"/"+inkmls
				#tree = ET.parse("expressmatch/101_alfonso.inkml")
				tree = ET.parse(filelocation)
				root = tree.getroot() 
				tempx,tempy = extract_features(root)

				splitValue = split(root)

				if splitValue == 2 or splitValue == 1:
					X.append(tempx)
					y.append(tempy)
				else:
					testX.append(tempx)
					testy.append(tempy)

	#X,y = extract_features(root)
	
	clf = svm.SVC()
	clf = clf.fit(X, y)

	#testX = [[1, 0.68, 35]]
	#testy = [[79]]
	yes, no = 0.0, 0.0 

	for x,y in zip(testX,testy):
		answer = clf.predict(x)[0]

		if (answer == y):
			yes += 1
		else:
			no += 0 


		print answer

	print "Accuracy:" + str((yes/float(len(testX)))*100)

	# for x in range(len(X)):
	# 	print X[x],y[x]

def extract_features(root):

	xcor,ycor = getxycor(root)				
	symbol, indices = getSymbolIndices(root)
	
	bezierpoints = getBezier(symbol,indices,xcor,ycor)
	
	for sym, j in zip(symbol, range(len(symbol))):
		feature = []
		feature.append(numberOfStrokes(sym))
		feature.append(aspectRatio(bezierpoints,j))
		feature.append(numberofcors(xcor,indices[j]))



	return feature,getLabel(sym)

	
				
def getLabel(symbol):
	if symbol == ',':
		symbol = 'COMMA'
	return classDict[symbol]


def numberOfStrokes(indices):
	return len(indices)

def numberofcors(xcor,indices):
	no = 0 
	for x in indices:
		no += len(xcor[x])
	return no
def aspectRatio(bezierpoints,i):

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

	return aspect

def split(root):

	global splitsSymbol
	for neighbor in root.findall('{http://www.w3.org/2003/InkML}annotation'):
		if neighbor.attrib['type']=="truth":
						
			if neighbor.text not in splitsSymbol:
				rand = random.randint(1,3)
				splitsSymbol[neighbor.text] = rand
				return rand
				
			if splitsSymbol[neighbor.text] == 1:

				splitsSymbol[neighbor.text] = 2

				return 1
				
			elif splitsSymbol[neighbor.text] == 2:
				
				splitsSymbol[neighbor.text] = 3

				return 2
				
			elif splitsSymbol[neighbor.text] == 3:
				
				splitsSymbol[neighbor.text] = 1
				return 3 

	return 1



if __name__ == '__main__':
	main()
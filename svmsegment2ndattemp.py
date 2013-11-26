from __future__ import division
import xml.etree.ElementTree as ET
import numpy as np
from sets import Set
from scipy.misc import comb
import matplotlib.pyplot as plt
import math, csv, os, random


DEBUG = 0
INFINITY = 100000.0


test = []
train = []
validate = []

splitsSymbol = {}

vertices = []
graph = []
edges = []

indexFile = open("index.csv")
classDict = {}

for line in indexFile:
	 classDict[line.split(",")[0]]= int((line.split(",")[1]).strip())

class Vertex(object):
	def __init__(self, label):
		self.label = label
		self.adjacent = []
		self.center = []

	def __repr__(self):
		return str(self.label)

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


def main():
	os.chdir("train")
	from sklearn import svm

	X, y= [], []
	testX, testy = [], []
	# feature_functions = []

	print "Start Feature extraction"

	for user in os.listdir("."):	
	  for inkmls in os.listdir(user):
		  if inkmls.endswith(".inkml"):
				filelocation = user+"/"+inkmls
				#tree = ET.parse("expressmatch/101_alfonso.inkml")
				tree = ET.parse(filelocation)
				root = tree.getroot() 
				#tempx,tempy = extract_features(root)
				tempx,tempy = extract_features_Segmentation(root)
				

				for tx,ty in zip(tempx,tempy):
					
					splitValue = split(ty)

					if splitValue == 2 or splitValue == 1:
						X.append(tx)
						y.append(ty)
					
					else:							
				
						testX.append(tx)
						testy.append(ty)
				  
	#X,y = extract_features(root)

	print "Start Training on" + str(len(X)) + "symbols"
	
	clf = svm.SVC()
	clf = clf.fit(X, y)

	print "Start Testing on " + str(len(testX)) + "symbols"

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
#def get_min(vect):

def shiftPoints(symbols,indices,xcor,ycor):
	
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

	

	
def normalizedPoints(symbols,indices,xcor,ycor):

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

def getBezier(symbols,indices,xcor,ycor):
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
		   
			plt.plot(xvals, yvals)

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


	xcor,ycor = shiftPoints(symbol,indices,xcor,ycor)
	xcor,ycor = normalizedPoints(symbol,indices,xcor,ycor)
	bezierpoints = getBezier(symbol,indices,xcor,ycor)

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

def getstrokepairs(indices):

	numberofstrokes = 0

	for x in indices:
		numberofstrokes += len(x)

	strokepairs = dict()

	for i in range(numberofstrokes-1):

		strokepairs[tuple([i,i+1])] = 0 

	
	for x in indices:
		if(len(x)>1):
			for p in range(len(x)-1):
				strokepairs[tuple([x[p],x[p+1]])] = 1

	newindices = []
	newsymbols = []
	for x in strokepairs:
		newindices.append(list(x))
		newsymbols.append(strokepairs[x])


	return newindices , newsymbols




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

	return features, labels

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
		plt.plot(x,y)
		plt.scatter(vertices[ind].center[0],vertices[ind].center[1])

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
	
	
	mst = prims(vertices)

	for x in mst:
		print x 

	print len(mst)

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


	plt.plot(pltx,plty)


	#plt.show()

	return bbx,bby

def prims(vertices):

	mst = []
	visited = Set([])

	u = vertices[0]
	visited.add(u) 

	for i in range(len(vertices)-1):
		minv = 9999999
		minVertex =  Vertex('test')

		for adj in u.adjacent:
			if adj[0] not in visited and adj[1] < minv:
				minVertex = adj[0]
				minv = adj[1]

		mst.append([u,minVertex,minv])		
		visited.add(minVertex)

		u = minVertex

	return mst

def split(symbol):

	global splitsSymbol	
						
	if symbol not in splitsSymbol:
		rand = random.randint(1,3)
		splitsSymbol[symbol] = rand
		return rand
		
	if splitsSymbol[symbol] == 1:

		splitsSymbol[symbol] = 2

		return 2
		
	elif splitsSymbol[symbol] == 2:
		
		splitsSymbol[symbol] = 3

		return 3
		
	elif splitsSymbol[symbol] == 3:
		
		splitsSymbol[symbol] = 1

		return 1 

	return 1



if __name__ == '__main__':
	main()






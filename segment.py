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

class Vertex(object):
	def __init__(self, label):
		self.label = label
		self.adjacent = []
		self.center = []

	def __repr__(self):
		return str(self.label)


def main():

	os.chdir("TrainINKML_v3")
	from sklearn import svm

	X, y= [], []
	testX, testy = [], []
	# feature_functions = []

	print "Start Feature extraction"

	# for user in os.listdir("."):
	#   for inkmls in os.listdir(user):
	# 	  if inkmls.endswith(".inkml"):
				# filelocation = user+"/"+inkmls
	tree = ET.parse("expressmatch/101_alfonso.inkml")
				# tree = ET.parse(filelocation)

	root = tree.getroot()
	extract_features(root) 

#def get_min(vect):


def extract_features(root):

	xcor,ycor = getxycor(root) 
	boundingbox(xcor, ycor)

	
def getCenter(x,y):
	
	xmin = min(x)
	ymin = min(y)
	xmax = max(x)
	ymax = max(y)

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
	plt.plot(bbx,bby)

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


	plt.show()

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




if __name__ == '__main__':
	main()






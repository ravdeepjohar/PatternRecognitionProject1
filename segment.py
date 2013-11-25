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

MAX_STROKES = 5
combinations = []

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
	tree = ET.parse("expressmatch/65_alfonso.inkml")
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
	
	#create a dictionary for the generated minimal spanning tree
	
	searchTree = {}
	for e in mst:
		src = e[0].label
		dest = e[1].label
		
		if src in searchTree:
			searchTree[src] = searchTree[src] + [dest]
		else:
			searchTree[src] = [dest]
		
		
		if dest in searchTree:
			searchTree[dest] = searchTree[dest] + [src]
		else:
			searchTree[dest] = [src]
	
	#print the minimal spanning tree dictionary
	for x in searchTree:
		print str(x) + ':' + str(searchTree[x])
	
	#apply iterative deepening search on each node as root of the tree and at depth levels
	#ranging from 0 - MAX_STROKES, generating all possible combinations of strokes as 
	#symbols
	for i in range(len(searchTree)):
		iterative_deepening(searchTree, i)

	plt.plot(pltx,plty)


	plt.show()
	
def iterative_deepening(edges, root):
	parent = []
	
	#take one node as root in the spanning tree and apply depth limited search for 
	#different depth levels to generate stroke combinations. 
	for k in range(MAX_STROKES):
		visited = []
		fringe = []
		fringe.append(edges.keys()[root])
		for i in range(len(edges)):
			parent.append(None)
		
		depth_limited_search(edges, parent, visited, fringe, k)


#depth limited search, goes to a certain depth and backtracks the
#path from child node to root creating a stroke combination
	
def depth_limited_search(edges, parent, visited, fringe, depth):
	
	
	node = fringe.pop()
	visited.append(node)
	
	if depth == 0:
		temp = []
		while node != None:
			temp.append(node)
			node = parent[node]
		if temp not in combinations:
			combinations.append(temp)
	else:
		children = edges[node]
		for child in children:
			if child not in visited:
				fringe.append(child)
				parent[child] = node
				depth_limited_search(edges, parent, visited, fringe, depth-1)
				
	
	
		
	
	

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






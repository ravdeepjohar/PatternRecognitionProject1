# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from scipy.misc import comb
import matplotlib.pyplot as plt

indexFile = open("index.csv")
classDict = {}
bezierpoints = []

for line in indexFile:
	 classDict[line.split(",")[0]]= int((line.split(",")[1]).strip())
		
indexFile.close()
		
def bernstein_poly(i, n, t):
        """
         The Bernstein polynomial of n, i as a function of t
        """
        return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def createPoints():
    tree = ET.parse("TrainINKML_v3/expressmatch/101_alfonso.inkml")
    #tree = ET.parse('../expressmatch/79_edwin.inkml')
    root = tree.getroot()               
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
    
    ymin=min(min(ycor))
    xmin=min(min(xcor))
    ymax=max(max(ycor))
    xmax=max(max(xcor))
        
    symbols = []
    indcies = []
    for neighbor in root.findall('{http://www.w3.org/2003/InkML}traceGroup'):
         for n in neighbor.findall('{http://www.w3.org/2003/InkML}traceGroup'):
              for n2 in n.iter('{http://www.w3.org/2003/InkML}annotation'):
                  symbols.append(n2.text)
                  tempind = []
                  for n in n.iter('{http://www.w3.org/2003/InkML}traceView'):
                      tempind.append(int(n.attrib["traceDataRef"]))
              indcies.append(tempind)
              
     # Bezier Curve
    
    newxcor = []
    newycor = [] 
    
    totalPoints = 15
    
    
    
    # create points for symbols using bezier 
    
    for sym in range(len(symbols)):
        itr = 1
        sums = 0
        length = len(indcies[sym])    
        nTimes = int(totalPoints/length)
	
		#initialized cor outside loop instead of inside
        cor = []    
		#removed temp = [] from here
        
        for x in indcies[sym]:
            
            if(itr==length):
                nTimes = totalPoints - sums
             
            nPoints = len(xcor[x])
            xPoints = np.array(xcor[x])
            yPoints = np.array(ycor[x])
            
            t = np.linspace(0.0, 1.0, nTimes)
            
            polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
            
            xvals = (np.dot(xPoints, polynomial_array)).reshape(-1,).tolist()
            yvals = (np.dot(yPoints, polynomial_array)).reshape(-1,).tolist()
           
		#directly adds a list of points into bezier list
		#instead of adding cor to temp
            #cor = []
            for x,y in zip(xvals,yvals):           
                cor.append([x,y])        
            
            #temp.append(cor)
           
            #plt.plot(xvals, yvals)
            itr = itr + 1
            sums = sums + nTimes
              
        bezierpoints.append(cor)
								
#Function to calculate Euclidean distance between two points    
def calculateEuclidean(a, b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def calculateShapeFeatures(i, j, radialOffset, angleOffset):
	centerOfMass1 = [0, 0]
	centerOfMass2 = [0, 0]
	center = [0, 0]
	
	maxPoint = [0, 0]
	distance = 0.0
	
	for point1, point2 in zip(bezierpoints[i], bezierpoints[j]):
		centerOfMass1 = [centerOfMass1[0] + point1[0], centerOfMass1[1] + point1[1]]
		centerOfMass2 = [centerOfMass2[0] + point2[0], centerOfMass2[1] + point2[1]]
	centerOfMass1 = [centerOfMass1[0] / len(bezierpoints[i]), centerOfMass1[1] / len(bezierpoints[i])]
	centerOfMass2 = [centerOfMass2[0] / len(bezierpoints[j]), centerOfMass2[1] / len(bezierpoints[j])]
	center = [(centerOfMass1[0] + centerOfMass2[0])/2, (centerOfMass1[1] + centerOfMass2[1])/2]
		
	for point1, point2 in zip(bezierpoints[i], bezierpoints[j]):
		d = calculateEuclidean(center, point1)
		if d > distance:
			distance = d
			maxPoint = point1
		
		d = calculateEuclidean(center, point2)
		if d > distance:
			distance
			maxPoint = point2
	print center
	print distance
	print maxPoint
	
	r = np.arange(0, distance, 0.01)
	theta = 2 * np.pi * r
	
	ax = plt.subplot(111, polar=True)
	#ax.plot(theta, r, color='r', linewidth=3)
	ax.scatter(maxPoint[0], maxPoint[1])
	ax.set_rmax(distance)
	ax.grid(True)
	
	ax.set_title("A line plot on a polar axis", va='bottom')
	plt.show()
	
		
def main():	
	createPoints()
	calculateShapeFeatures(0, 1, 5, 5)
	
	
		
if __name__ == '__main__':
	main()
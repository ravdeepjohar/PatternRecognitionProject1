from __future__ import division
import xml.etree.ElementTree as ET
import numpy as np
from scipy.misc import comb
import matplotlib.pyplot as plt
import math
import csv


f = open('one.txt')
csvFile = open("one.csv", "wb")



indexFile = open("index.csv")
classDict = {}

for line in indexFile:
     classDict[line.split(",")[0]]= int((line.split(",")[1]).strip())
     

def bernstein_poly(i, n, t):
        """
         The Bernstein polynomial of n, i as a function of t
        """
        return comb(n, i) * ( t**(n-i) ) * (1 - t)**i
        
for line in f:  
    
        
    tree = ET.parse(line[:-1])
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
    
    totalPoints = 50
    
    bezierpoints = []
    
    # create points for symbols using bezier 
    
    for sym in range(len(symbols)):
        itr = 1
        sums = 0
        length = len(indcies[sym])    
        nTimes = int(totalPoints/length)
            
        temp = []    
        
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
           
            cor = []
            for x,y in zip(xvals,yvals):           
                cor.append([x,y])        
            
            temp.append(cor)
           
            plt.plot(xvals, yvals)
            itr = itr + 1
            sums = sums + nTimes
              
        bezierpoints.append(temp)
        
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
        
   # Quadrants 
    normalized_Y = []
    normalized_X = []
    quadrant = []
    
    for i in range(len(bezierpoints)):
        tempYList = []
        tempXList = []
        for j in range(len(bezierpoints[i])):
            for coord in bezierpoints[i][j]:
                #mag = np.sqrt(coord[0] * coord[0] + coord[1] * coord[1])
                
                tempX = (coord[0]- xmin) / (xmax-xmin)
                tempY = (coord[01]- ymin) / (ymax-ymin)
                
                tempYList.append(tempY)
                tempXList.append(tempX)
        
        normalized_Y.append(tempYList)
        normalized_X.append(tempXList)
        
    
        #calculate quadrants for starting and ending poitns
        #(1 : NW, 2 : NE, 3 : SW, 4 : SE)
    
    
        startQuadrant = 0
        endQuadrant = 0
        
        end = len(tempXList) - 1
        
        if(tempXList[0] < 0.5 and tempYList[0] > 0.5):
            startQuadrant = 1
        elif(tempXList[0] > 0.5 and tempYList[0] > 0.5):
            startQuadrant = 2
        elif(tempXList[0] < 0.5 and tempYList[0] < 0.5):
            startQuadrant = 3
        elif(tempXList[0] > 0.5 and tempYList[0] < 0.5):
            startQuadrant= 4
            
        if(tempXList[end] < 0.5 and tempYList[end] > 0.5):
            endQuadrant = 1
        elif(tempXList[end] > 0.5 and tempYList[end] > 0.5):
            endQuadrant = 2
        elif(tempXList[end] < 0.5 and tempYList[end] < 0.5):
            endQuadrant = 3
        elif(tempXList[end] > 0.5 and tempYList[end] < 0.5):
            endQuadrant= 4
                    
        quadrant.append([startQuadrant, endQuadrant])
        
        #calculate vicinity slope pairs
    
    vicinity_slope = []
    
    for i in range(len(bezierpoints)):
        tempSlope = []
        tempList = []
        for j in range(len(bezierpoints[i])):
            for coord in bezierpoints[i][j]:
                tempList.append(coord)
        
        for k in range(2, len(tempList)-2):
            horizontal = [1.0, 0.0]
            vec = [tempList[k+2][0] - tempList[k-2][0], tempList[k+2][1] - tempList[k-2][1]]
            magHorizonal = np.sqrt(horizontal[0] * horizontal[0] + horizontal[1] * horizontal[1])
            magVec = np.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
            
            cosTheta = (horizontal[0] * vec[0] + horizontal[1] * vec[1])/(magVec * magHorizonal)
            
            if(cosTheta > 1.0):
                cosTheta = 1.0
            if(cosTheta < -1.0):
                cosTheta = -1.0
            
            sinTheta = math.sin(math.acos(cosTheta))
            
            if(sinTheta > 1.0):
                sinTheta = 1.0
            if(sinTheta < -1.0):
                sinTheta = -1.0
            tempSlope.append([sinTheta, cosTheta])
            #tempSlope.append(theta)
        
        vicinity_slope.append(tempSlope)
    
     #calculate curvature
        
    curvature = []
    numCusps = []
        
    for i in range(len(bezierpoints)):
        tempList = []
        tempCurvature = []
        tempCusps = 0
        for j in range(len(bezierpoints[i])):
            for coord in bezierpoints[i][j]:
                tempList.append(coord)
        
        for k in range(2, len(tempList) - 2):
            vec1 = [tempList[k-2][0] - tempList[k][0], tempList[k-2][1] - tempList[k][1]]
            mag1 = np.sqrt(vec1[0] * vec1[0] + vec1[1] * vec1[1])
            vec2 = [tempList[k][0] - tempList[k+2][0], tempList[k][1] - tempList[k+2][1]]
            mag2 = np.sqrt(vec2[0] * vec2[0] + vec2[1] * vec2[1])
            
            cosTheta = (vec1[0] * vec2[0] + vec1[1] * vec2[1])/(mag1 * mag2)
            if(cosTheta > 1.0):
                cosTheta = 1.0
            if(cosTheta < -1.0):
                cosTheta = -1.0
                
            sinTheta = math.sin(math.acos(cosTheta))
            
            if(sinTheta > 1.0):
                sinTheta = 1.0
            if(sinTheta < -1.0):
                sinTheta = -1.0
            tempCurvature.append([sinTheta, cosTheta])
            #tempCurvature.append(theta)
            
            theta = math.acos(cosTheta)
            if(theta < (5.0 * math.pi)/180.0):
                tempCusps += 1
            
            
        numCusps.append(tempCusps)
        curvature.append(tempCurvature)
        
                  
    numStrokes = []
    for i in range(len(symbols)):
        numStrokes.append(len(indcies[i]))
        
    csvList = []
                
    for i, j in zip(symbols, range(len(symbols))):
        tempList = []
        if i == ",":
            i="comma"
        tempList.append(classDict[i])
        tempList.append(numStrokes[j])
        tempList.append(aspect_ratio[j])
        tempList.append(numCusps[j])
        tempList.append(quadrant[j][0])
        tempList.append(quadrant[j][1])
        
        for k in range(len(normalized_Y[j])):
            tempList.append(normalized_Y[j][k])
        
        for k in range(len(curvature[j])):
            tempList.append(curvature[j][k][0])
            tempList.append(curvature[j][k][1])
            #tempList.append(curvature[j][k])
        
        for k in range(len(vicinity_slope[j])):
            tempList.append(vicinity_slope[j][k][0])
            tempList.append(vicinity_slope[j][k][1])
            #tempList.append(vicinity_slope[j][k])
        
        #print tempList[0]
            
        csvList.append(tempList)
    
    for i in range(len(csvList)):
        c = csv.writer(csvFile,)
        c.writerow(csvList[i])
        
    
        #csvFile.writelines(str(csvList[i]))
csvFile.close() 
    

       
f.close()
    
   
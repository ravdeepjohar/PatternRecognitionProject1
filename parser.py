from __future__ import division
import xml.etree.ElementTree as ET
import numpy as np
from scipy.misc import comb
import matplotlib.pyplot as plt
import math
import csv


f = open('AllEM_part4_TRAIN_all.txt')
csvFile = open("features.csv", "wb")


for line in f:
    
    tree = ET.parse(line[:-1])
    #tree = ET.parse('../expressmatch/79_edwin.inkml')
    root = tree.getroot()
        
    def bernstein_poly(i, n, t):
        """
         The Bernstein polynomial of n, i as a function of t
        """
        return comb(n, i) * ( t**(n-i) ) * (1 - t)**i
        
    
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
       
    symbol_dict = {} 
    for neighbor in root.findall('{http://www.w3.org/2003/InkML}traceGroup'):
         for n in neighbor.findall('{http://www.w3.org/2003/InkML}traceGroup'):
              for n2 in n.iter('{http://www.w3.org/2003/InkML}annotation'):
                  symbol = n2.text
                  indices = []
                  for n in n.iter('{http://www.w3.org/2003/InkML}traceView'):
                      indices.append(int(n.attrib["traceDataRef"]))
                  symbol_dict[symbol]=indices
                      
    
    
    # Bezier Curve
    
    newxcor = []
    newycor = [] 
    
    totalPoints = 50
    
    points_dict = {}
    
    # create points for symbols using bezier 
    
    for sym in symbol_dict:
        itr = 1
        sums = 0
        length = len(symbol_dict[sym])    
        nTimes = int(totalPoints/length)
            
        temp = []    
        
        for x in symbol_dict[sym]:
            
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
           
            #plt.plot(xvals, yvals)
            itr = itr + 1
            sums = sums + nTimes
        
      
        points_dict[sym]=temp
        
    #calculate aspect ratio for each symbol
    
    aspect_ratio = []
        
    for i in points_dict:
        max_width = 0.0
        max_height = 0.0
        min_width = 0.0
        min_height = 0.0
        #print i
        for j in range(len(points_dict[i])):
            for coord in points_dict[i][j]:
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
        
        
        '''print max_width
        print min_width
        print max_height
        print min_height
        print '\n'''
    
        if(max_height - min_height > 0):
            aspect = (max_width - min_width)/(max_height - min_height)
        else:
            aspect = (max_width - min_width)
        aspect_ratio.append(aspect)
        
    
    #calculate normalized y coordinates
    
    normalized_Y = []
    
    for i in points_dict:
        tempYList = []
        for j in range(len(points_dict[i])):
            for coord in points_dict[i][j]:
                mag = np.sqrt(coord[0] * coord[0] + coord[1] * coord[1])
                temp = coord[1] / mag
                tempYList.append(temp)
        
        normalized_Y.append(tempYList)
        
    
    #calculate vicinity slope pairs
    
    vicinity_slope = []
    
    for i in points_dict:
        tempSlope = []
        tempList = []
        for j in range(len(points_dict[i])):
            for coord in points_dict[i][j]:
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
        
        vicinity_slope.append(tempSlope)
        
    #calculate curvature
        
    curvature = []
        
    for i in points_dict:
        tempList = []
        tempCurvature = []
        for j in range(len(points_dict[i])):
            for coord in points_dict[i][j]:
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
        
        curvature.append(tempCurvature)
        
    
    #calculate number of strokes
    
    numStrokes = []
    for i in points_dict:
        num = 0
        for j in range(len(points_dict[i])):
            num += 1
        numStrokes.append(num)
    
    
    csvList = []
                
    for i, j in zip(points_dict, range(len(numStrokes))):
        #print str(i) + " " + str(j)
        tempList = []
        tempList.append(i)
        tempList.append(numStrokes[j])
        tempList.append(aspect_ratio[j])
        
        for k in range(len(normalized_Y[j])):
            tempList.append(normalized_Y[j][k])
        
        for k in range(len(curvature[j])):
            tempList.append(curvature[j][k][0])
            tempList.append(curvature[j][k][1])
        
        for k in range(len(vicinity_slope[j])):
            tempList.append(vicinity_slope[j][k][0])
            tempList.append(vicinity_slope[j][k][1])        
        
        csvList.append(tempList)
    
    for i in range(len(csvList)):
        c = csv.writer(csvFile)
        c.writerow(csvList[i])
        
    
        #csvFile.writelines(str(csvList[i]))

csvFile.close()                  
f.close()
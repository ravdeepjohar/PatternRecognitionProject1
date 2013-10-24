from __future__ import division
import xml.etree.ElementTree as ET
import numpy as np
from scipy.misc import comb
import matplotlib.pyplot as plt


#f = open('AllEM_part4_TRAIN_all.txt')

#for line in f:
    
#tree = ET.parse(line)
tree = ET.parse('../expressmatch/98_alfonso.inkml')
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
        x.append(int(cor.split(" ")[0]))
        y.append(int(cor.split(" ")[1]))
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
       
        plt.plot(xvals, yvals)
        itr = itr + 1
        sums = sums + nTimes
    
  
    points_dict[sym]=temp
    
    
    
    
    





    





            
            
       
    
    
    
                  
             
                  
              

                        
#f.close()
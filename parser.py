import xml.etree.ElementTree as ET
import numpy as np
from scipy.misc import comb
import matplotlib.pyplot as plt


#f = open('AllEM_part4_TRAIN_all.txt')

#for line in f:
    
#tree = ET.parse(line)
tree = ET.parse('../expressmatch/65_alfonso.inkml')
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
   
    
for neighbor in root.findall('{http://www.w3.org/2003/InkML}traceGroup'):
     for n in neighbor.findall('{http://www.w3.org/2003/InkML}traceGroup'):
          for n2 in n.iter('{http://www.w3.org/2003/InkML}annotation'):
              symbol = n2.text
              indices = []
              for n in n.iter('{http://www.w3.org/2003/InkML}traceView'):
                  indices.append(int(n.attrib["traceDataRef"]))

# Remove Duplicates 

newxcor = []
newycor = []                
for xc,yc in zip(xcor,ycor): 
    newx = []
    newy = []              
    for x, y in zip(xc, yc):
        t = True
        for x1c,y1c in zip(newx,newx):
            if x==x1c and y==y1c:
                t = False
        if (t):
            newx.append(x)
            newy.append(y)
    newxcor.append(newx)
    newycor.append(newy)
    
xcor = newxcor
ycor = newycor


# Bezier Curve

newxcor = []
newycor = [] 

for x in range(len(xcor)):
    nTimes = 30
    nPoints = len(xcor[x])
    xPoints = np.array(xcor[x])
    yPoints = np.array(ycor[x])
    
    t = np.linspace(0.0, 1.0, nTimes)
    
    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    
    newxcor.append(xvals.reshape(-1,).tolist())
    newycor.append(yvals.reshape(-1,).tolist())
    
    plt.plot(xvals, yvals)
    
xcor = newxcor
ycor = newycor



cors = []
for xc,yc in zip(xcor,ycor):
    cor = []
    for x,y in zip(xc,yc):           
        cor.append([x,y])
    cors.append(cor)
    





            
            
       
    
    
    
                  
             
                  
              

                        
#f.close()
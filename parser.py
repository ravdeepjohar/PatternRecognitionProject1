import xml.etree.ElementTree as ET


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

# Benzier Curve

cors = []
for xc,yc in zip(xcor,ycor):
    cor = []
    for x,y in zip(xc,yc):           
        cor.append([x,y])
    cors.append(cor)
    





            
            
       
    
    
    
                  
             
                  
              

                        
#f.close()
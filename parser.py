import xml.etree.ElementTree as ET


#f = open('AllEM_part4_TRAIN_all.txt')

#for line in f:
    
#tree = ET.parse(line)
tree = ET.parse('../expressmatch/65_alfonso.inkml')
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
        x.append(cor.split(" ")[0]) 
        y.append(cor.split(" ")[1])
    xcor.append(x)
    ycor.append(y)
   
    
for neighbor in root.findall('{http://www.w3.org/2003/InkML}traceGroup'):
     for n in neighbor.findall('{http://www.w3.org/2003/InkML}traceGroup'):
          for n2 in n.iter('{http://www.w3.org/2003/InkML}annotation'):
              print n2.text
              for n in n.iter('{http://www.w3.org/2003/InkML}traceView'):
                  print n.attrib["traceDataRef"]
        
          print " "
       
         
#f.close()
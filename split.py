import xml.etree.ElementTree as ET
import random

f = open('AllEM_part4_TRAIN_all.txt')

symbol = {}

train = open('train.txt','a')
validate = open('validate.txt','a')
test = open('test.txt','a')
for line in f:
    
    tree = ET.parse(line.rstrip())
    root = tree.getroot()
   
    for neighbor in root.findall('{http://www.w3.org/2003/InkML}annotation'):
        if neighbor.attrib['type']=="truth":
                        
            if neighbor.text not in symbol:
                symbol[neighbor.text] = random.randint(1,3)
                
            if symbol[neighbor.text] == 1:
                train.write(line)
                symbol[neighbor.text] = 2
                
            elif symbol[neighbor.text] == 2:
                validate.write(line)
                symbol[neighbor.text] = 3
                
            elif symbol[neighbor.text] == 3:
                test.write(line)
                symbol[neighbor.text] = 1
        
        
train.close()
validate.close()
test.close()
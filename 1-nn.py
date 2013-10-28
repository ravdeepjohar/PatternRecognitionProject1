import numpy as np
import xml.etree.ElementTree as ET
import csv


if not os.path.exists('nnlg'):
    os.makedirs('nnlg')

def euclidean(a, b):
    temp= 0.0
    for i in range(len(a)):
        temp += (a[i] - b[i]) * (a[i] - b[i])
    
    return np.sqrt(temp)

train = open('trainfeatures/filelist.txt', 'rb')
trainFileList = []
data = []
X = []
Y = []
for line in train:
    trainFileList.append(line.split('\n')[0])

train.close()
'''f = open(trainFileList[0], 'rb')
print f.readline().split('\r')[0].split(',')
'''
for i in range(len(trainFileList)):
    f = open(trainFileList[i], 'rb')
    for line in f:
        temp = line.split('\r')[0].split(",")
        data.append(temp)
    f.close()
    
validate = open('validatefeatures/filelist.txt', 'rb')
validateFileList = []

for line in validate:
    validateFileList.append(line.split('\n')[0])

validate.close()

for i in range(len(validateFileList)):
    f = open(validateFileList[i], 'rb')
    for line in f:
        temp = line.split('\r')[0].split(",")
        data.append(temp)
    f.close()

for i in range(len(data)):
    Y.append(int(data[i][0]))
    temp = []
    for j in range(1, len(data[i])):
        temp.append(float(data[i][j]))
    X.append(temp)
    
    


test = open('testfeatures/filelist.txt', 'rb')
testFileList = []

for line in test:
    testFileList.append(line.split('\n')[0])
test.close()

total = 0.0
classification = 0.0

test_inkml = open('test.txt', 'rb')
testInkmlList = []

for line in test_inkml:
    testInkmlList.append(line.split('\r')[0])
test_inkml.close()

classDict = {}
indexFile = open("index.csv")

for line in indexFile:
     classDict[int((line.split(",")[1]).strip())]= line.split(",")[0]
    
    

for it in range(0, len(testFileList)):
    f = open(testFileList[it], 'rb')
    testData = []
    XX = []
    YY = [] 
    
    for line in f:
        temp = line.split('\r')[0].split(',')
        testData.append(temp)
        
    for i in range(len(testData)):
        YY.append(int(testData[i][0]))
        temp = []
        for j in range(1, len(testData)):
            temp.append(float(testData[i][j]))
        
        XX.append(temp)
    
    for i in range(len(XX)):
        nn_dist = []
        for j in range(len(X)):
            dist = euclidean(XX[i], X[j])
            nn_dist.append([Y[j], dist])
        
        min_dist = [0, 100000.0]
        for k in range(len(nn_dist)):
            if(min_dist[1] > nn_dist[k][1]):
                min_dist = nn_dist[k]
        
        #print min_dist 
        #print YY[i]
        total += 1.0
        if(min_dist[0] == YY[i]):
            classification += 1.0
    
    f.close()     
    
    tree = ET.parse(testInkmlList[it])
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
    
    s = 0
    lg = []
    edges = []
    for i in range(len(YY)):
        sym = classDict[YY[i]]
        strokes = indcies[i]
        
        if(len(strokes) > 1):
            multiStrokes = []
            for i in range(len(strokes)):
                tempLG = []
                tempLG.append('N')
                tempLG.append(' ' + str(s))
                multiStrokes.append(s)
                s = s + 1
                tempLG.append(' ' + sym)
                tempLG.append(' 1.0')
                lg.append(tempLG)
            for i in range(len(multiStrokes)):
                for j in range(i, len(multiStrokes)):
                    if i != j:
                        tempEdge = []
                        tempEdge.append('E')
                        tempEdge.append(' ' + str(multiStrokes[i]))
                        tempEdge.append(' ' + str(multiStrokes[j]))
                        tempEdge.append(' *')
                        tempEdge.append(' 1.0')
                        edges.append(tempEdge)
                        
                        tempEdge = []
                        tempEdge.append('E')
                        tempEdge.append(' ' + str(multiStrokes[j]))
                        tempEdge.append(' ' + str(multiStrokes[i]))
                        tempEdge.append(' *')
                        tempEdge.append(' 1.0')
                        edges.append(tempEdge)
        else:
            tempLG = []
            tempLG.append('N')
            tempLG.append(' ' + str(s))
            s = s + 1
            tempLG.append(' ' + sym)
            tempLG.append(' 1.0')
            lg.append(tempLG)
            
    lgFile = open('nnlg/' + str(testInkmlList[it].split('/')[1].split('.')[0]) + '.lg', 'wb')
    
    c = csv.writer(lgFile)
    
    for i in range(len(lg)):
        c.writerow(lg[i])
    
    #lgFile.write('\n')
    
    for i in range(len(edges)):
        c.writerow(edges[i])
    
    lgFile.close()
    
    


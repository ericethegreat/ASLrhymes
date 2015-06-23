
from xml.dom import minidom

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

from random import randint
import math
import copy

'''
HELPER FUNCTIONS
'''


'''
Normalizes six sets of coordinates (presumably X, Y, and Z for the left and right hands)

param: six lists
'''
def set2Coords(Xright, Xleft, Yright, Yleft, Zright, Zleft):
	(avgX, avgY, avgZ) = getCentroid(Xright + Xleft, Yright + Yleft, Zright + Zleft)
	(stdX, stdY, stdZ) = getStdDev(Xright + Xleft, Yright + Yleft, Zright + Zleft)
	adjustCoords(Xright, Yright, Zright, avgX, avgY, avgZ, stdX, stdY, stdZ)
	adjustCoords(Xleft, Yleft, Zleft, avgX, avgY, avgZ, stdX, stdY, stdZ)

'''
Normalizes three sets of coordinates (X, Y, and Z)

param: three lists
'''
def setCoords(X, Y, Z):
	(avgX, avgY, avgZ) = getCentroid(X, Y, Z)
	(stdX, stdY, stdZ) = getStdDev(X, Y, Z)
	
	adjustCoords(X, Y, Z, avgX, avgY, avgZ, stdX, stdY, stdZ)

'''
Helper function that normalizes coordinate points given their mean and standard deviation.
'''
def adjustCoords(X, Y, Z, avgX, avgY, avgZ, stdX, stdY, stdZ):
		for i in range(len(X)):
			X[i] = (X[i] - avgX) / stdX
			Y[i] = (Y[i] - avgY) / stdY
			Z[i] = (Z[i] - avgZ) / stdZ
'''
Helper function that returns a tuple of standard deviations.
'''
def getStdDev(X, Y, Z):
	return ( np.std(X), np.std(Y), np.std(Z) )

'''
Helper function that returns a tuple of averaged values.
'''
def getCentroid(X, Y, Z):
	return ( np.mean(X), np.mean(Y), np.mean(Z) )

'''
Helper function that sums the distance between pairs of X, Y, and Z coordinates.
Used when two frames (of different signs) are compared.
'''
def sumDistance(X1, Y1, Z1, X2, Y2, Z2):
	sumDist = 0
	
	for i in range(len(X1)):
		dist = calculateDistance(X1[i],Y1[i],Z1[i],X2[i],Y2[i],Z2[i])
		sumDist += dist
	return sumDist
'''
Helper function that calculates the distance between pairs of X, Y, and Z coordinates.
'''
def calculateDistance(X1, Y1, Z1, X2, Y2, Z2):
	return math.sqrt( (X2 - X1)**2 + (Y2 - Y1)**2 + (Z2 - Z1)**2 )

'''
Returns two tuples of X, Y, Z coordinates for the right and left hands
of a given sign across all frames.

param: signValue (String)
return: two tuples of three lists each
'''
def getCoords(signValue):
	xmldoc = minidom.parse('testoutput.xml')
	signslist = xmldoc.getElementsByTagName('sign')
	Xleft = []
	Yleft = []
	Zleft = []
	
	Xright = []
	Yright = []
	Zright = []

        '''
        Iterates through all signs
        '''
	for s in signslist:
                '''
                Collect X, Y, and Z coordinates if a sign name matches the name given
                '''
		if s.attributes['name'].value == signValue:
			frameslist = s.getElementsByTagName('frame')
			for f in frameslist:
				jointslist = f.getElementsByTagName('joint')
				for j in jointslist:
					if j.attributes['name'].value == "HandRight":
						Xright = Xright + [float(j.attributes['x'].value)]
						Yright = Yright + [float(j.attributes['y'].value)]
						Zright = Zright + [float(j.attributes['z'].value)]
					if j.attributes['name'].value == "HandLeft":
						Xleft = Xleft + [float(j.attributes['x'].value)]
						Yleft = Yleft + [float(j.attributes['y'].value)]
						Zleft = Zleft + [float(j.attributes['z'].value)]

	return ((Xright, Yright, Zright),(Xleft, Yleft, Zleft))

'''
Prints the distance calculation for each possible pair of signs.
'''
def calculateDistances():

        '''
        Opens blank output file.
        '''
	f = open("rhymelist.csv","w")
	nf = open("notrhymelist.csv","w")

        rhymes = getRhymeDict()
	xmldoc = minidom.parse('testoutput.xml')
	return xmldoc.getElementsByTagName('sign') 

        '''
        Iterates through each sign
        '''
	for i in range(len(signslist)):
		sign1 = signslist[i].attributes['name'].value
		((Xright1, Yright1, Zright1),(Xleft1, Yleft1, Zleft1)) = getCoords(sign1)
		
		sign1Frames = len(Xright1)
		
		(origXright1, origYright1, origZright1,origXleft1, origYleft1, origZleft1) = getDeepCopy(Xright1, Yright1, Zright1, Xleft1, Yleft1, Zleft1)
		
		set2Coords(Xright1, Xleft1, Yright1, Yleft1, Zright1, Zleft1)
		setCoords(origXright1, origYright1, origZright1)
		setCoords(origXleft1, origYleft1, origZleft1)
		
		X1 = Xright1 + Xleft1
		Y1 = Yright1 + Yleft1
		Z1 = Zright1 + Zleft1

		(Xright1copy, Yright1copy, Zright1copy,Xleft1copy, Yleft1copy, Zleft1copy) = getDeepCopy(Xright1, Yright1, Zright1, Xleft1, Yleft1, Zleft1)
		(origXright1copy, origYright1copy, origZright1copy,origXleft1copy, origYleft1copy, origZleft1copy) = getDeepCopy(origXright1, origYright1, origZright1, origXleft1, origYleft1, origZleft1)

                '''
                Iterates through the signs that appear after the current sign
                '''
		for j in range(i + 1, len(signslist)):
                        sign2 = signslist[j].attributes['name'].value
                        ((Xright2, Yright2, Zright2),(Xleft2, Yleft2, Zleft2)) = getCoords(sign2)

                        sign2Frames = len(Xright2)
                        (origXright2, origYright2, origZright2,origXleft2, origYleft2, origZleft2) = getDeepCopy(Xright2, Yright2, Zright2, Xleft2, Yleft2, Zleft2)

                        '''
                        Normalizes the coordinates of the right hand, left hand, and combination of both.
                        '''
                        setCoords(origXright2, origYright2, origZright2)
                        setCoords(origXleft2, origYleft2, origZleft2)
			set2Coords(Xright2, Xleft2, Yright2, Yleft2, Zright2, Zleft2)

                        '''
                        Between two signs, interpolates the sign of fewer frames to equal quantity as the longer sign.
                        '''
			firstSignWasModified = False;
			if ( len(Xright1) > len(Xright2) ):
                                targetSize = len(Xright1)
                                interpolateTo(Xright2, targetSize)
                                
				interpolateTo(Xleft2, targetSize)
				interpolateTo(Yright2, targetSize)
				interpolateTo(Yleft2, targetSize)
				interpolateTo(Zright2, targetSize)
				interpolateTo(Zleft2, targetSize)

				interpolateTo(origXright2, targetSize)
				interpolateTo(origXleft2, targetSize)
				interpolateTo(origYright2, targetSize)
				interpolateTo(origYleft2, targetSize)
				interpolateTo(origZright2, targetSize)
				interpolateTo(origZleft2, targetSize)
			elif ( len(Xright1) < len(Xright2) ):
				targetSize = len(Xright2)
				interpolateTo(Xright1, targetSize)
				interpolateTo(Xleft1, targetSize)
				interpolateTo(Yright1, targetSize)
				interpolateTo(Yleft1, targetSize)
				interpolateTo(Zright1, targetSize)
				interpolateTo(Zleft1, targetSize)
				
				interpolateTo(origXright1, targetSize)
				interpolateTo(origXleft1, targetSize)
				interpolateTo(origYright1, targetSize)
				interpolateTo(origYleft1, targetSize)
				interpolateTo(origZright1, targetSize)
				interpolateTo(origZleft1, targetSize)
				firstSignWasModified = True;
                                        
			X2 = Xright2 + Xleft2 
			Y2 = Yright2 + Yleft2
			Z2 = Zright2 + Zleft2
			
			'''
                        Computes and writes the total distance sum between two signs by their right hand, left hand, and both.
                        '''
			sumDistRight = sumDistance(origXright1, origYright1, origZright1, origXright2, origYright2, origZright2)
			sumDistLeft = sumDistance(origXleft1, origYleft1, origZleft1, origXleft2, origYleft2, origZleft2)	
			sumDist = sumDistance(X1, Y1, Z1, X2, Y2, Z2)

                        if sign1.upper in rhymes and sign2.upper() in rhymes[sign1.upper()]:
                                f.write(sign1 + "," + str(sign1Frames) + "," + sign2 + "," + str(sign2Frames) + "," + str(sumDistRight) + "," + str(sumDistLeft) + "," + str(sumDist) + "," + str(sumDistRight + sumDistLeft + sumDist) + "\n")
                        else:
                                nf.write(sign1 + "," + str(sign1Frames) + "," + sign2 + "," + str(sign2Frames) + "," + str(sumDistRight) + "," + str(sumDistLeft) + "," + str(sumDist) + "," + str(sumDistRight + sumDistLeft + sumDist) + "\n")

                        '''
                        Removes the interpolated points for next comparison
                        '''
                        if firstSignWasModified:
                                (origXright1, origYright1, origZright1, origXleft1, origYleft1, origZleft1) = getDeepCopy(origXright1copy, origYright1copy, origZright1copy,origXleft1copy, origYleft1copy, origZleft1copy)
                                (Xright1, Yright1, Zright1, Xleft1, Yleft1, Zleft1) = getDeepCopy(Xright1copy, Yright1copy, Zright1copy,Xleft1copy, Yleft1copy, Zleft1copy)
                                        
	f.close()
        nf.close()


'''
Expands a list to a given target size by interpolation.

param: coords (list of floating values)
       targetSize (int)
'''
def interpolateTo(coords, targetSize):
        
	origSize = len(coords)
	diff = targetSize - origSize
	
	numAvailableBins = origSize - 1
	
	if (diff >= origSize):
		standardPerBin = diff / numAvailableBins
		additionalBins = diff % numAvailableBins
	else:
		standardPerBin = 0	
		additionalBins = diff

	binsList = []
	for i in range(numAvailableBins):
		binsList = binsList + [standardPerBin]
	
	if (additionalBins != 0):
		interval = int(numAvailableBins // additionalBins)
	
		for i in range(additionalBins):
			binsList[(i + 1) * interval - 2] = binsList[(i + 1) * interval - 2] + 1
	
	tempList = []
	
	for i in range(origSize - 1):
		val1 = coords[i]
		val2 = coords[i + 1]		
		numBetween = binsList[i]
		temp = []
		if numBetween > 0:
			increment = ( (val2 - val1) / float(numBetween + 1) )
			for i in range(1, numBetween + 1):
				temp = temp + [increment * i + val1]
		tempList += [temp]
	

	for i in range(origSize - 1):
		indexToInsert = origSize - (i + 1);
		binIndex = origSize - 2 - i;

		if len(tempList[binIndex]) > 0:
			tempList[binIndex].reverse()
			for num in tempList[binIndex]:
                                coords.insert(indexToInsert, num)




'''
Returns deep copies of six lists
'''
def getDeepCopy(Xright, Yright, Zright, Xleft, Yleft, Zleft):
        return (copy.deepcopy(Xright),copy.deepcopy(Yright),copy.deepcopy(Zright),copy.deepcopy(Xleft),copy.deepcopy(Yleft),copy.deepcopy(Zleft))

def getRhymeDict():
        rhymes = {}
        
        f = open("rhymes.csv","r")
        for line in f:
                line = line.strip()
                rhymeList = line.split(",")
                for rhyme in rhymeList:
                        rhymes[rhyme] = rhymeList

        return rhymes

def getClassNumToRhymes():
        classNumToRhymes = {}
        
        f = open("rhymes.csv","r")
        i = 0
        for line in f:
                i = i + 1
                line = line.strip()
                rhymeList = line.split(",")
                classNumToRhymes[i] = rhymeList

        return classNumToRhymes


'''
Produces normalized graphs for each sign
'''
def makeGraphsFromXML():

        '''
        Opens xml used for data analysis
        '''
        
	xmldoc = minidom.parse('testoutput.xml')
	signslist = xmldoc.getElementsByTagName('sign')
        

	'''
        Iterates through all signs
        '''
	for s in signslist:
		Xleft = []
		Yleft = []
		Zleft = []
	
		Xright = []
		Yright = []
		Zright = []
		
		signname = s.attributes['name'].value
		frameslist = s.getElementsByTagName('frame')

		'''
                Iterates through all frames in a sign
                '''
		for f in frameslist:
			jointslist = f.getElementsByTagName('joint')

			'''
                        Iterates through all joints in a frame
                        '''
			for j in jointslist:

                                '''
                                Stores the X, Y, and Z coordinates of left and right hand joints
                                '''
				if j.attributes['name'].value == "HandRight":
					Xright = Xright + [float(j.attributes['x'].value)]
					Yright = Yright + [float(j.attributes['y'].value)]
					Zright = Zright + [float(j.attributes['z'].value)]
				if j.attributes['name'].value == "HandLeft":
					Xleft = Xleft + [float(j.attributes['x'].value)]
					Yleft = Yleft + [float(j.attributes['y'].value)]
					Zleft = Zleft + [float(j.attributes['z'].value)]


                #Generates graph that is not normalized
		'''
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(Xleft, Yleft, Zleft, c='r',marker='o')
		ax.scatter(Yright, Zright, Xright, c='b', marker='o')
		ax.set_xlabel('x axis')
		ax.set_ylabel('y axis')
		ax.set_zlabel('z axis')
		ax.set_title(signname)
		fig.savefig("graphs/" + signname + '.png', bbox_inches='tight')
		plt.close()
		'''

		#Normalizes coordinates
		#set2Coords(Xright, Xleft, Yright, Yleft, Zright, Zleft)

		#Generates normalized graphs
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(Xright, Yright, Zright, c='r',marker='o')
		ax.scatter(Xleft, Yleft, Zleft, c='b',marker='o')
		ax.set_xlabel('x axis')
		ax.set_ylabel('y axis')
		ax.set_zlabel('z axis')
		ax.set_title(signname + " (normalized)")
		fig.savefig("graphs/" + signname + '.png', bbox_inches='tight')
		plt.close()

'''
Produces normalized graphs for each sign
'''
def makeGraphsByClass(signsDict):

        classNumToCoords = {}
        signToClassNum = {}
        setUpClasses(classNumToCoords, signToClassNum, signsDict)

        classNumToSigns = getClassNumToRhymes()
        
        for s in signsDict:
                #skip to next sign if the sign is not in a class number
                if not s in signToClassNum:
                        continue

                classNum = signToClassNum[s]
                (Xright1, Xleft1, Yright1, Yleft1, Zright1, Zleft1) = classNumToCoords[classNum]
                
		#Normalizes coordinates

		#Generates normalized graphs
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(Xright1, Yright1, Zright1, c='r',marker='o')
		ax.scatter(Xleft1, Yleft1, Zleft1, c='b',marker='o')
		ax.set_xlabel('x axis')
		ax.set_ylabel('y axis')
		ax.set_zlabel('z axis')

		signs = str(classNumToSigns[classNum])
		ax.set_title(signs)
		fig.savefig("graphs/" + str(signs) + '.png', bbox_inches='tight')
		plt.close()

		frames = []
                for i in range(1, len(Xright1) + 1 ):
                        frames = frames + [i]

                graphAnAxis("x", signs, frames, Xright1)
                graphAnAxis("y", signs, frames, Yright1)
                graphAnAxis("z", signs, frames, Zright1)

'''
Produces normalized graphs for each sign
'''
def makeGraphs(signsDict):
        for s in signsDict:
                (Xright1, Xleft1, Yright1, Yleft1, Zright1, Zleft1) = signsDict[s]
                
		#Normalizes coordinates

		#Generates normalized graphs
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(Xright1, Yright1, Zright1, c='r',marker='o')
		ax.scatter(Xleft1, Yleft1, Zleft1, c='b',marker='o')
		ax.set_xlabel('x axis')
		ax.set_ylabel('y axis')
		ax.set_zlabel('z axis')
		ax.set_title(s)
		fig.savefig("graphs/" + s + '.png', bbox_inches='tight')
		plt.close()

		frames = []
                for i in range(1, len(Xright1) + 1 ):
                        frames = frames + [i]

                graphAnAxis("x", s, frames, Xright1)
                graphAnAxis("y", s, frames, Yright1)
                graphAnAxis("z", s, frames, Zright1)

def graphAnAxis(axisname, signname, X, Y): 
        #Generataes normalized graphs
	fig = plt.figure()
	ax = fig.add_subplot(111)
	
	ax.scatter(X, Y, c='r',marker='o')
	ax.set_xlabel('frames')
	ax.set_ylabel(axisname + 'point')
	ax.set_title(signname + " " + axisname)
	ax.set_ylim([-3,3])

        npX = np.array(X)
        npY = np.array(Y)
        z = np.polyfit(npX, npY, 18)
        f = np.poly1d(z)
        ax.plot(npX, npY, '.', npX, f(npX), '-')

	fig.savefig("graphs/" + signname + '_' + axisname + ".png", bbox_inches='tight')
	plt.close()

def setUpClasses(classNumToCoords, signToClassNum, signsList):
        
        f = open("rhymes.csv","r")
        classNum = 0
        for line in f:
                classNum = classNum + 1
                line = line.strip()
                rhymeList = line.split(",")
                Xright = []
                Xleft = []
                Yright = []
                Yleft = []
                Zright = []
                Zleft = []
                n = 0
                for aSign in rhymeList:
                        if aSign in signsList:
                                n = n + 1
                                (aXright, aXleft, aYright, aYleft, aZright, aZleft) = signsList[aSign]
                                #for first time around
                                if len(Xright) == 0:      
                                        Xright = [0] * len(aXright)
                                        Xleft = [0] * len(aXleft)
                                        Yright = [0] * len(aYright)
                                        Yleft = [0] * len(aYleft)
                                        Zright = [0] * len(aZright)
                                        Zleft = [0] * len(aZleft)
                                        signToClassNum[aSign] = classNum
                                
                                #for further interpolation
                                if len(Xright) > len(aXright):
                                        targetSize = len(Xright)
                                        interpolateTo(aXright, targetSize)
                                        interpolateTo(aXleft, targetSize)
                                        interpolateTo(aYright, targetSize)
                                        interpolateTo(aYleft, targetSize)
                                        interpolateTo(aZright, targetSize)
                                        interpolateTo(aZleft, targetSize)
                                elif len(Xright) < len(aXright):
                                        targetSize = len(aXright)
                                        interpolateTo(Xright, targetSize)
                                        interpolateTo(Xleft, targetSize)
                                        interpolateTo(Yright, targetSize)
                                        interpolateTo(Yleft, targetSize)
                                        interpolateTo(Zright, targetSize)
                                        interpolateTo(Zleft, targetSize)
                                        
                                for i in range(len(aXright)):
                                        Xright[i] += aXright[i]
                                        Xleft[i] += aXleft[i]
                                        Yright[i] += aYright[i]
                                        Yleft[i] += aYleft[i]
                                        Zright[i] += aZright[i]
                                        Zleft[i] += aZleft[i]
                                signToClassNum[aSign] = classNum
                if n == 0:
                        continue
                Xright[:] = [aValue / n for aValue in Xright]
                Xleft[:] = [aValue / n for aValue in Xleft]
                Yright[:] = [aValue / n for aValue in Yright]
                Yleft[:] = [aValue / n for aValue in Yleft]
                Zright[:] = [aValue / n for aValue in Zright]
                Zleft[:] = [aValue / n for aValue in Zleft]
                classNumToCoords[classNum] = (Xright, Xleft, Yright, Yleft, Zright, Zleft)

def adjustCoords(X1, Y1, Z1, X2, Y2, Z2):
        deltaX = X1[0] - X2[0]
        deltaY = Y1[0] - Y2[0]
        deltaZ = Z1[0] - Z2[0]
        X1[:] = [aValue - deltaX for aValue in X1]
        Y1[:] = [aValue - deltaY for aValue in Y1]
        Z1[:] = [aValue - deltaZ for aValue in Z1]

'''
Prints the distance calculation for each possible pair of signs.
'''
def calculateDistancesWithDict(signsDict):

        '''
        Opens blank output file.
        '''
	f = open("rhymelist.csv","w")
	nf = open("notrhymelist.csv","w")

        rhymes = getRhymeDict()
        classNumToRhymes = getClassNumToRhymes()
        
        signslist = signsDict.keys()
        print signslist

        classNumToCoords = {}
        signToClassNum = {}
        setUpClasses(classNumToCoords, signToClassNum, signsDict)

	for i in range(len(signslist)):
		sign1 = signslist[i]

                if not sign1 in signToClassNum:
                        (Xright1, Xleft1, Yright1, Yleft1, Zright1, Zleft1) = signsDict[sign1]
                        sign1Frames = len(Xright1)
                        (origXright1, origYright1, origZright1,origXleft1, origYleft1, origZleft1) = getDeepCopy(Xright1, Yright1, Zright1, Xleft1, Yleft1, Zleft1)
                        (Xright1copy, Yright1copy, Zright1copy,Xleft1copy, Yleft1copy, Zleft1copy) = getDeepCopy(Xright1, Yright1, Zright1, Xleft1, Yleft1, Zleft1)
                        (origXright1copy, origYright1copy, origZright1copy,origXleft1copy, origYleft1copy, origZleft1copy) = getDeepCopy(origXright1, origYright1, origZright1, origXleft1, origYleft1, origZleft1)
                        '''
                        Iterates through the signs that appear after the current sign
                        '''
                        minDist = 999999
                        secondMinDist = 99998
                        matchingClassNum = -1
                        secondMatchingClassNum = -1
                        for classNum in classNumToCoords:
                                (Xright2, Xleft2, Yright2, Yleft2, Zright2, Zleft2) = classNumToCoords[classNum]
                                sign2Frames = len(Xright2)
                                if sign2Frames == 0:
                                        continue
                                
                                (origXright2, origYright2, origZright2,origXleft2, origYleft2, origZleft2) = getDeepCopy(Xright2, Yright2, Zright2, Xleft2, Yleft2, Zleft2)
                                '''
                                Between two signs, interpolates the sign of fewer frames to equal quantity as the longer sign.
                                '''
                                firstSignWasModified = False;
                                if ( len(Xright1) > len(Xright2) ):
                                        targetSize = len(Xright1)
                                        interpolateTo(Xright2, targetSize)
                                        interpolateTo(Xleft2, targetSize)
                                        interpolateTo(Yright2, targetSize)
                                        interpolateTo(Yleft2, targetSize)
                                        interpolateTo(Zright2, targetSize)
                                        interpolateTo(Zleft2, targetSize)

                                        interpolateTo(origXright2, targetSize)
                                        interpolateTo(origXleft2, targetSize)
                                        interpolateTo(origYright2, targetSize)
                                        interpolateTo(origYleft2, targetSize)
                                        interpolateTo(origZright2, targetSize)
                                        interpolateTo(origZleft2, targetSize)
                                elif ( len(Xright1) < len(Xright2) ):

                                        targetSize = len(Xright2)
                                        interpolateTo(Xright1, targetSize)
                                        interpolateTo(Xleft1, targetSize)
                                        interpolateTo(Yright1, targetSize)
                                        interpolateTo(Yleft1, targetSize)
                                        interpolateTo(Zright1, targetSize)
                                        interpolateTo(Zleft1, targetSize)
				
                                        interpolateTo(origXright1, targetSize)
                                        interpolateTo(origXleft1, targetSize)
                                        interpolateTo(origYright1, targetSize)
                                        interpolateTo(origYleft1, targetSize)
                                        interpolateTo(origZright1, targetSize)
                                        interpolateTo(origZleft1, targetSize)
                                        firstSignWasModified = True;

                                
                                X2 = Xright2 + Xleft2 
                                Y2 = Yright2 + Yleft2
                                Z2 = Zright2 + Zleft2

                                X1 = Xright1 + Xleft1
                                Y1 = Yright1 + Yleft1
                                Z1 = Zright1 + Zleft1
			

                                print "len is " + str(len(X1))
                                print "len for X2 is "  + str(len(X2))
                                
                                '''
                                tried this approach and it confused arizona with israel and science with yellow
                                adjustCoords(origXright1, origYright1, origZright1, origXright2, origYright2, origZright2)
                                adjustCoords(origXleft1, origYleft1, origZleft1, origXleft2, origYleft2, origZleft2)
                                '''
			
                                '''
                                Computes and writes the total distance sum between two signs by their right hand, left hand, and both.
                                '''

                                
                                
                                sumDistRight = sumDistance(origXright1, origYright1, origZright1, origXright2, origYright2, origZright2)
                                sumDistLeft = sumDistance(origXleft1, origYleft1, origZleft1, origXleft2, origYleft2, origZleft2)	
                                #sumDist = sumDistance(X1, Y1, Z1, X2, Y2, Z2)

                                sign1 = sign1.upper()
                                

                                '''
                                if sign1 in rhymes and sign2 in rhymes[sign1]:
                                        print "rhyming words: " + sign1 + " " + sign2
                                        #print "RHYMES\nRHYMES\RHYMES\n " + sign1 + "," + str(sign1Frames) + "," + sign2 + "," + str(sign2Frames) + "," + str(sumDistRight) + "," + str(sumDistLeft) + "," + str(sumDist) + "," + str(sumDistRight + sumDistLeft + sumDist) + "\n"
                                        f.write(sign1 + "," + str(sign1Frames) + "," + sign2 + "," + str(sign2Frames) + "," + str(sumDistRight) + "," + str(sumDistLeft) + "," + str(sumDist) + "," + str(sumDistRight + sumDistLeft + sumDist) + "\n")
                                else:
                                        #print sign1 + "," + str(sign1Frames) + "," + sign2 + "," + str(sign2Frames) + "," + str(sumDistRight) + "," + str(sumDistLeft) + "," + str(sumDist) + "," + str(sumDistRight + sumDistLeft + sumDist) + "\n"
                                        nf.write(sign1 + "," + str(sign1Frames) + "," + sign2 + "," + str(sign2Frames) + "," + str(sumDistRight) + "," + str(sumDistLeft) + "," + str(sumDist) + "," + str(sumDistRight + sumDistLeft + sumDist) + "\n")
                                '''

                                if (sumDistLeft + sumDistRight) < minDist:
                                        minDist = sumDistLeft + sumDistRight
                                        matchingClassNum = classNum
                                elif (sumDistLeft + sumDistRight) < secondMinDist:
                                        secondMinDist = sumDistLeft + sumDistRight
                                        secondMatchingClassNum = classNum

                                '''
                                Removes the interpolated points for next comparison
                                '''
                                if firstSignWasModified:
                                        (origXright1, origYright1, origZright1, origXleft1, origYleft1, origZleft1) = getDeepCopy(origXright1copy, origYright1copy, origZright1copy,origXleft1copy, origYleft1copy, origZleft1copy)
                                        (Xright1, Yright1, Zright1, Xleft1, Yleft1, Zleft1) = getDeepCopy(Xright1copy, Yright1copy, Zright1copy,Xleft1copy, Yleft1copy, Zleft1copy)
                                        
                        f.write(sign1 + "," + str(matchingClassNum) + "," + str(minDist) + "," + str(classNumToRhymes[matchingClassNum]) + "\n")
                        f.write(sign1 + "," + str(matchingClassNum) + "," + str(secondMinDist) + "," + str(classNumToRhymes[secondMatchingClassNum]) + "\n\n")
	f.close()
        nf.close()






'''
Normalizes the signs dictionary
'''
def normalizeSignsDict(signsDict, allX, allY, allZ):

        #signsDict[signname.upper()] = (Xright, Xleft, Yright, Yleft, Zright, Zleft)

        (avgX, avgY, avgZ) = getCentroid(allX, allY, allZ)
        (stdX, stdY, stdZ) = getStdDev(allX, allY, allZ)
        
        for signname in signsDict.keys():
                (Xright, Xleft, Yright, Yleft, Zright, Zleft) = signsDict[signname.upper()]
                for i in range(len(Xright)):
                        aXright = Xright[i]
                        aXleft = Xleft[i]
                        aYright = Yright[i]
                        aYleft = Yleft[i]
                        aZright = Zright[i]
                        aZleft = Zleft[i]
                        
                        Xright[i] = (aXright - avgX) / stdX
                        Xleft[i] = (aXleft - avgX) / stdX
                        Yright[i] = (aYright - avgY) / stdY
                        Yleft[i] = (aYleft - avgY) / stdY
                        Zright[i] = (aZright - avgZ) / stdZ
                        Zright[i] = (aZleft - avgZ) / stdZ
                signsDict[signname.upper()] = (Xright, Xleft, Yright, Yleft, Zright, Zleft) 
                

'''
Returns dict of signs.
'''
def getSignsDict():

        signsDict = {}
        allX = []
        allY = []
        allZ = []

        '''
        Opens xml used for data analysis
        '''
	xmldoc = minidom.parse('testoutput.xml')
	signslist = xmldoc.getElementsByTagName('sign')

	'''
        Iterates through all signs
        '''
	for s in signslist:
		Xleft = []
		Yleft = []
		Zleft = []
	
		Xright = []
		Yright = []
		Zright = []
		
		signname = s.attributes['name'].value
		frameslist = s.getElementsByTagName('frame')

		'''
                Iterates through all frames in a sign
                '''
		for f in frameslist:
			jointslist = f.getElementsByTagName('joint')

			'''
                        Iterates through all joints in a frame
                        '''
			for j in jointslist:

                                '''
                                Stores the X, Y, and Z coordinates of left and right hand joints
                                '''
				if j.attributes['name'].value == "HandRight":
					Xright = Xright + [float(j.attributes['x'].value)]
					Yright = Yright + [float(j.attributes['y'].value)]
					Zright = Zright + [float(j.attributes['z'].value)]
				if j.attributes['name'].value == "HandLeft":
					Xleft = Xleft + [float(j.attributes['x'].value)]
					Yleft = Yleft + [float(j.attributes['y'].value)]
					Zleft = Zleft + [float(j.attributes['z'].value)]


                signname = signname.upper()
                signsDict[signname] = (Xright, Xleft, Yright, Yleft, Zright, Zleft)
                
                allX = allX + Xright + Xleft
                allY = allY + Yright + Yleft
                allZ = allZ + Zright + Zleft

        normalizeSignsDict(signsDict, allX, allY, allZ)
        return signsDict
	

'''
Generates normalzied graphs for all signs and prints distance comparison analysis.
'''
def main():
        signsDict = getSignsDict()
	#makeGraphs(signsDict)
	makeGraphsByClass(signsDict)
	#calculateDistances()
        calculateDistancesWithDict(signsDict)
	print("Done")
	
main()


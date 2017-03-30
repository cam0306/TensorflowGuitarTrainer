""""
Author: Cameron Knight
Description: uses beautiful soup to prosses htm data from
a guitar tab website then processes the tab to a sipler form of 
data.
"""


import bs4 as bs
import urllib
import numpy as np
import time
import random
from multiprocessing import Pool
import os

def CollectDataFromPage(url):
	"""
	Collects the tab from tabs ultimate format
	from given url
	url - url containing tab information
	"""
	soup = safeSoupPage(url)

	tab = ''
	for prevtab in soup.find_all('pre', class_='js-tab-content'):
		tab += prevtab.text + "\n"
	return tab

def StripSpace(tab):
	"""
	Strips the space of the tab input
	"""
	contentLines = []
	for line in tab:
		if(len(line) > 2 and line.count('-') > 3):
			contentLines.append(line.strip())
	return contentLines


def removeLabels(tab):
	"""
	removes the string labels off the front and back of each line
	"""
	newTabs = []

	for i in range(len(tab)):
		p_start = -1
		p_end = -1
		for c in range(len(tab[i])):
			if tab[i][c] == '|':
				if(p_start == -1):
					p_start = c
				p_end = c
		if(len(tab[i]) > 0):
			if(tab[i][len(tab[i])-1] == '-'):
				p_end = len(tab[i])
		spaceCount = 0
		for c in range(p_start,p_end):
			if tab[i][c] == ' ' or tab[i][c] == '\t':
				spaceCount += 1
				if(spaceCount > 4):
					p_start = p_end

		
		if(p_start != p_end):
			newTabs.append(tab[i][p_start + 1: p_end])

	return newTabs

def groupLines(tab):
	"""
	groups the tabs string into lists
	"""
	strings = 6

	if(len(tab) % 6 == 0):
		tab = [tab[i:i + strings] for i in range(0, len(tab) -1, strings)]

		return tab
	else:
		return None

def ConvertToNumbers(tab):
	"""
	Converts the tab into a numerical format
	"""
	for bar in range(len(tab)):
		for note in range(len(tab[bar])):
			for string in range(len(tab[bar][note])):
				if tab[bar][note][string].isdigit():
					tab[bar][note][string] = int(tab[bar][note][string])
				else:
					tab[bar][note][string] = -1
	return(tab)

def WriteTabToFile(tab,fileName,barSep='|',noteSep='~',stringSep=',',songSep='\n'):
	"""
	writes the tab to the file using given seporators
	"""
	with open(fileName,'a') as f:
		for bar in tab:
			for note in bar:
				for string in note:
					f.write(str(string) + stringSep)
				f.write(noteSep)
			f.write(barSep)
		f.write(songSep)


def processFileIntoTabs(fileName):
	"""
	Goes through each link in the file and attempts to write the tab from it 
	to the converted file
	"""
	numLines = num_lines = sum(1 for line in open('Data/Tabs/'+fileName,'r'))
	processedLines = 0
	onPercent = 0
	print(fileName[9] + ' Has ' + str(numLines) + " Lines!")
	for line in open('Data/Tabs/'+fileName,'r'):
		try:
			processDatafromPage(line.rstrip(),'Data/TabData/'+fileName)

		except Exception as e:
			print("Failed to process page! " + line.rstrip())
		processedLines += 1
		percent = int(100 * (processedLines/numLines))
		if( percent > onPercent):
			onPercent = percent
			print(fileName[9] + ': ' + str(percent) + "%")


	print('Done Processing: ' + fileName)


def convertLinesToNotes(tab):
	"""
	Converts each tab to notes in bars in numerical form
	"""
	bars = []

	for line in range(len(tab)):
		notes = [] 

		doubleDigetPotential = False
		if(not all(len(tab[line][0]) == len(i) for i in tab[line])):
			return None
		for noteNum in range(len(tab[line][0])):
			note = []
			if(not doubleDigetPotential):
				for string in range(6):

						t_note = (tab[line][string][noteNum])
						if(t_note.isdigit()):
							doubleDigetPotential = True
						note.append(t_note)
			else:
				for string in range(6):
					t_note = (tab[line][string][noteNum])
					if(notes[-1][string] .isdigit()):
						if(t_note.isdigit()):
							potentialDoubleNote = int(notes[-1][string]) * 10 + int(t_note)
							if(potentialDoubleNote <= 24):
								notes[-1][string] = str(potentialDoubleNote)
								t_note =('-')
						doubleDigetPotential = False
					note.append(t_note)



			if('|' in note):
				if(len(notes) > 0):
					bars.append(notes)
				notes = []
				doubleDigetPotential = False

			else:
				notes.append(note)
		bars.append(notes)
	return bars
def ValidateTab(tab):
	"""
	returns a true false value indicating weather or not a tab is valid
	"""
	#checks if tab contains data
	if(len(tab) <= 0):
		return False

	#Validates Len of the notes
	for bar in tab:
		for note in bar:
			if(len(note) != 6):
				return False



	return True
def processDatafromPage(url,fileName='Data/TabData/TabData'):
	"""
	main function which handles the collection of all data
	"""

	tab = CollectDataFromPage(url)
	tab = tab.split('\n')
	tab = removeLabels(tab)
	if(len(tab) < 6):
		return
	tab = groupLines(tab)
	if(tab == None):
		return
	tab = convertLinesToNotes(tab)
	if(tab == None):
		return
	tab = ConvertToNumbers(tab)
	if(tab == None):
		return
	if(not ValidateTab(tab)):
		return
	WriteTabToFile(tab, fileName)



def safeSoupPage(page):

	response = None
	attempts = 0
	while response == None and attempts < 5:
		try:
			response = urllib.request.urlopen(page).read()
		except Exception as e:
			time.sleep(5)
			attempts += 1


	if response == None:
		return None

	soup = bs.BeautifulSoup(response, 'lxml')

	return soup

def getAlphabetLinks(startPg):
	soup = safeSoupPage(startPg)
	if soup == None:
		return []
	links = []
	for prevtab in soup.find_all('a', class_='wb'):
		links.append(prevtab['href'])
	return list(set(links))

def gatherPageNumbers(alphaURL):
	soup = safeSoupPage(alphaURL)
	if soup == None:
		return []
	links = []
	for prevtab in soup.find_all('a'):
		if alphaURL[:-4] in prevtab['href']:
			links.append( prevtab['href'])
	links.append(alphaURL)

	return(list(set(links)))

def gatherBandPageNumbers(bandURL):
	soup = safeSoupPage(bandURL)
	if soup == None:
		return []
	links = []
	for prevtab in soup.find_all('a', class_='ys'):
		links.append( 'https://www.ultimate-guitar.com' + prevtab['href'])
	links.append(bandURL)

	return(list(set(links)))


def gatherBandsPages(pageUrl):
	soup = safeSoupPage(pageUrl)
	if soup == None:
		return []
	links = []
	for prevtab in soup.find_all('tr'):
		isTab = False
		for td in prevtab.find_all('td',attrs={'style':'color:#DDDDCC'}):
			isTab = True
		for td in prevtab.find_all('td'):
			if isTab and td.a != None and 'ultimate-guitar.com' not in td.a['href']:
				links.append('https://www.ultimate-guitar.com' + td.a['href'])
	return list(set(links))


def gatherTabs(bandUrl):
	"""
	Gets the urls of all the tabs from a band's page
	bandurl - the bands page or any number page of the band
	"""
	soup = safeSoupPage(bandUrl)
	if soup == None:
		return []
	links = []
	for prevtab in soup.find_all('tr'):
		isTab = False
		for td in prevtab.find_all('td'):
			if td.b != None:
				if td.b.string == 'Tabs' or td.b.string == 'Chords'  :
					isTab = True
		for td in prevtab.find_all('td'):
			if isTab and td.a != None and 'plus.ult' not in td.a['href'] and 'ultimate-guitar.com' in td.a['href'] and 'ukulele' not in td.a['href']:
				links.append(td.a['href'])
	return list(set(links))



def writeLinksToFile(alpha):
	"""
	Goes through each layers of links until the tabs layer is linked
	and writes the final link to a file
	alpha - Url of the highest level alphabetical sorted page
	"""
	lettersToProcess = 'abcdefghijklmnopqrstuvwxyz0'
	fileName = alpha[38] # Sets the name to the 38th iterative character which is the letter being processed
	if(fileName in lettersToProcess):
		print(fileName)
		with open("Data/Tabs/"+"tabLinks-"+fileName+".txt",'w+') as f:
			n_written = 0
			for pg in gatherPageNumbers(alpha):
				for band in gatherBandsPages(pg):
			 		for b_pg in gatherBandPageNumbers(band):
			 			for tab in gatherTabs(b_pg):
		 					f.write(tab+ "\n")
		 					n_written += 1
		print("Lines Written:", n_written,"for",fileName)

def navigateUltimateGuitar(startPg):
	"""
	Uses parallel processing to go through Tabs ultimate and 
	collects and records the links to files
	"""
	alpha = getAlphabetLinks(startPg)
	with Pool(27) as p:
		p.map(writeLinksToFile,alpha)


def GenTabs():
	"""
	Uses parallel processing to process each of the tabs into a usable format
	from file links
	"""
	fs = os.listdir('Data/Tabs')
	#processFileIntoTabs(fs[5])

	with Pool(27) as p:
		p.map(processFileIntoTabs,fs)     

def CombineOutputs():
	"""
	Conbines outputs into one file
	"""
	directory = 'Data/TabData/'
	outDir = 'Data/'

	with open(outDir + 'Tabs.txt','a') as outFile:
		for l in 'abcdefghijklmnopqrstuvwxyz0':
			with open(directory + 'tabLinks-' + l + '.txt','r') as inFile:
				p_line = ''
				for t in inFile:
					if(p_line != t):
						outFile.write(t.strip() + '\n')
					else:
						print('Duplicate Line')
					p_line = t



def main():
	GenTabs()
	CombineOutputs()

if __name__ == "__main__":
	main()

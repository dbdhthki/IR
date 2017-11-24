


import cgi, cgitb 
import urllib.request
import shutil
import re
import operator
import html
from collections import Counter
import os
import math
from nltk.stem import *
import json


'''
function to get inverted index from a file	
Parameter :
	Input:  
	Output: returns inverted index ditionary
'''
def GetInvertedIndex():
	with open('invertedFile.txt') as data_file:
		data = json.load(data_file)
	return data

'''
function to get freqency of unique word in alphabetical order	
Parameter :
	Input: list of words 
	Output: returns sorted words in alphabetical order with its frequecies
'''
def CountWords(wordlist):
	#Sorts tokens alphabatically
	sortedTokens =  sorted(wordlist)

	#dictionary of words and its frequency
	wordWithFreq = Counter(sortedTokens)
	return wordWithFreq


def CleanQueryContent(queryContent):
	SPEC_CHAR_REGEX =re.compile(r"[^A-Za-z0-9-]+") #regular expression to remove all characters that are not alphabets and numbers except hyhens
	NUMBER_REGEX =re.compile(r"[0-9]")#regular expression to remove numbers
	result =NUMBER_REGEX.sub(' ',queryContent)
	result =SPEC_CHAR_REGEX.sub(' ',result)

	#converts to lower case	
	result=result.lower()
	return result

'''
function to remove stop words from a file along with stemming
Parameter :
	Input: content from which stop words are to be removed and stemmed and stopwords array
	Output: list of words without stopwords and words that are stemmed
'''
def StopWordsRemovalAndStemming(htmlContent):

	# stores stop words in an array
	fileObj=open('english.stopwords.txt','r')
	fileContent=fileObj.read()
	fileObj.close()# to close a file
	stopwords=fileContent.split()

	#creates object of porter stemmer
	stemmer = PorterStemmer()

	words = htmlContent.split()

	result=[]
	for word in words:
		if word not in stopwords:
			result.append(stemmer.stem(word))

	return result

'''
function to get user entered query
Parameter :
	Input: 
	Output: returns tokenized and stemmed query
'''
def GetQuery(enteredQuery):
	query=CleanQueryContent(enteredQuery)
	query=StopWordsRemovalAndStemming(query)	
	query=CountWords(query)
	return query

def ValidateUserQuery(query,indexFile):
	cnt=0
	for qterm,f in query.items():
		if qterm in indexFile:
			cnt=1
			break
		else:
			cnt=0
	if cnt==0:
		print('No results found!')
		return False
	else:
		return True


'''
function to get user entered query
Parameter :
	Input: tokenized and stemmed query
	Output: returns tokenized and stemmed query
'''
def GetDocWithQueryTerm(query,indexFile):
	docWithQueryTerm=set()
	for qterm,f in query.items():
		#print(qterm,f)
		for wordInIndex,docFreqDetail  in indexFile.items():
			if qterm==wordInIndex:
				for k,v in docFreqDetail['doclist'].items():
					docWithQueryTerm.add(k)
	return docWithQueryTerm

'''
function to get all distinct documents in a collection
Parameter :
	Input: inverted index file
	Output: returns list of distinct documents
'''
def GetAllDistinctDocs(indexFile):
	distinctDoc=set()
	for wordInIndex,docFreqDetail in indexFile.items():
		for k,v in docFreqDetail['doclist'].items():
			distinctDoc.add(k)
	return distinctDoc

'''
function to all term and frequency of document which contains query
Parameter :
	Input: 
	Output: returns list which contains term and its frequency for all documents
'''
def GetAllTermAndFreqForDocsWithQuery(docWithQueryTerm,indexFile):
	docDetails={}
	for d in docWithQueryTerm:	
		termFreq={}
		for wordInIndex,docFreqDetail in indexFile.items():
			for k,tf in docFreqDetail['doclist'].items():
				if(k==d):
					termFreq[wordInIndex]=tf
		docDetails[d]=termFreq

	return docDetails

'''
function to get retrive results on the basis of entered query
Parameter :
	Input: total number of documents,query,inverted index and list containing term and document frequency
	Output: returns ranked list of retrieved documents
'''
def GetRankedResults(N,query,indexFile,docDetails):
	R=[]#holds unique document containing query
	weightOfD=dict()#holds term weight for each term in each document
	weightOfQ=dict()#holds term weight fot each term in query
	dotProdQD=dict()#holds dot product of term in document and query
	lenOfDoc=dict()#holds length of each document
	finalScore=dict()#holds cosine similarity value for each document

	m=max(query.items(), key=operator.itemgetter(1))[1]#holds maximum term frequency of a term in query to normalize tf

	#print(m)

	#retrieval algorithm using inverted index
	for qterm,qf in query.items():
		df=indexFile[qterm]['df']#gets document frequency for each term
		#print(qterm,df)
		idf=math.log((N/df),2)#calculates idf 
		weightOfQ[qterm]=(qf/m)*idf	#calculates tf-idf for each term in query

		#list of document containing query terms
		L=indexFile[qterm]['doclist']
		
		#for each document and count of query in L
		for D,C in L.items():
			#add document to R if it is not added already
			if D not in R:
				R.append(D)
				dotProdQD[D]=0.0
				weightOfD[D]=[]
			
			n=max(docDetails[D].items(), key=operator.itemgetter(1))[1]#get maximum frequency of term in document to normalize tf

			tf=C/n#normalizing term frequency		
			dotProdQD[D]+=weightOfQ[qterm]*idf*tf #dot product of query and document
		
			weightOfD[D].append({qterm:idf*tf})# holds weight for each term in each document

	#finds length of query vector Q
	lenOfQ=0.0
	for k,v in weightOfQ.items():
		lenOfQ+=v*v
	lenOfQ=math.sqrt(lenOfQ)

	#print(weightOfD)

	#finds length of each document
	for d in R:
		sumoFfreq=0.0
		for termFreq in weightOfD[d]:
			for term,freq in termFreq.items():
				sumoFfreq+=freq*freq

		lenOfDoc[d]=math.sqrt(sumoFfreq)

	#calculates cosine similarity of each document with query
	for d in R:
		try:
			finalScore[d]=dotProdQD[d]/(lenOfDoc[d]*lenOfQ)
		except ZeroDivisionError:
			finalScore[d]=1

	#print(finalScore)
	return finalScore

def GetResults(enteredQuery):
	
	query=GetQuery(enteredQuery)

	indexFile=GetInvertedIndex()
	#pprint(indexFile)

	if(ValidateUserQuery(query,indexFile)):
		#get documents containing query terms
		docWithQueryTerm=set()
		docWithQueryTerm=GetDocWithQueryTerm(query,indexFile)
		#print(docWithQueryTerm)

		#total number of documents in collection
		distinctDoc=GetAllDistinctDocs(indexFile)
		N=len(distinctDoc)
		#print(N)

		#gets all term and its frequency for all documents that contain query term
		#needed to tf normalization to get maximum frequency of term of each document
		docDetails={}
		docDetails=GetAllTermAndFreqForDocsWithQuery(docWithQueryTerm,indexFile)
		#print(docDetails)

		#gets ranked documents
		res=GetRankedResults(N,query,indexFile,docDetails)
		
		#rank result in descending order of cosineSim
		res=sorted(res.items(), key=lambda x: (-x[1], x[0]))
		for l in res:
			s=l[0]
			s=s.split()
			print(s[1])
		


GetResults("daya budhathoki")



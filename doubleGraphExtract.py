from xml.dom.minidom import parse
import xml.dom.minidom
import operator
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from itertools import takewhile, tee, izip, chain, product
from collections import Counter
import os
import math
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer

import networkx
import string
import sys

#added for solving the headache ascii encode/decode problem
reload(sys)  
sys.setdefaultencoding('utf8')

def writeMapToFile(listFile, fileName):
	theFile = open(fileName, 'w')
	for item in listFile:
		theFile.write("%s:" % str(item))
		for inner in listFile[item]:
			theFile.write("%s " % str(inner))
		theFile.seek(-1, os.SEEK_CUR)
		theFile.write("\n")
	theFile.close()
	return

def writeListToFile(listFile, fileName):
	theFile = open(fileName, 'w')
	for item in listFile:
		theFile.write("%s: " % str(item[0]))
		for inner in item[1]:
			theFile.write("%s, " % str(inner))
		theFile.seek(-2, os.SEEK_CUR)
		theFile.write("\n")
	theFile.close()
	return

def write2DList(listFile, fileName):
	tf = open(fileName, "w")
	for outerList in listFile:
		for ele in outerList:
			tf.write("%s, " % str(ele))

		tf.seek(-2, os.SEEK_CUR)
		tf.write("\n")

	tf.close()
	return

def write2DListVertical(listFile, fileName):
	tf = open(fileName, 'w')
	for outerList in listFile:
		for ele in outerList:
			tf.write("%s\n" % str(ele))
		tf.write("\n\n")
	tf.close()
	return

def writeFreqToFile(listFile, fileName):
	theFile = open(fileName, "w")
	for item in listFile:
		theFile.write("%s: " % str(item[0]))
		theFile.write("%s\n" % str(item[1]))
	return

def XmlParsing(targetFile, targetTag):
	try:
		DOMTree = xml.dom.minidom.parse(targetFile)
	except xml.parsers.expat.ExpatError, e:
		print "The file causing the error is: ", fileName
		print "The detailed error is: %s" %e
	else:
		collection = DOMTree.documentElement

		resultList = collection.getElementsByTagName(targetTag)
		return resultList

	return "ERROR"

def tagNPFilter(sentence):
	tokens = nltk.word_tokenize(sentence)
	tagged = nltk.pos_tag(tokens)
	# NPgrammar = r"""NP:{<DT>?<JJ|NN|NNS>*<NN|NNS>}
	# ND:{<DT>?<NN|NNS><IN><DT>?<JJ|NN|NNS>*}"""
	NPgrammar = "NP:{<DT>?<JJ|NN|NNS>*<NN|NNS>}"
	#Problem: "a powerful computer with strong support from university" 
	#1, nested; 2, 'computer' is the keywords? or 'computer with support' is the keywords?
	cp = nltk.RegexpParser(NPgrammar)
	resultTree = cp.parse(tagged)   #result is of type nltk.tree.Tree
	result = ""
	stemmer = SnowballStemmer("english")
	for node in resultTree:
		if (type(node) == nltk.tree.Tree):
			#result += ''.join(item[0] for item in node.leaves()) #connect every words
			#result += stemmer.stem(node.leaves()[len(node.leaves()) - 1][0]) #use just the last NN

			if node[0][1] == 'DT':
				node.remove(node[0])  #remove the determiners
			currNounPhrase = ''.join(stemmer.stem(item[0]) for item in node.leaves())
			result += currNounPhrase

			multiplyTimes = len(node.leaves())
			for i in range(multiplyTimes - 1):
				result += " "
				result += currNounPhrase

		else:
			result += stemmer.stem(node[0])
		result += " "
	return result

def generateNPTextList(text):

	stop_words = set(nltk.corpus.stopwords.words('english'))

	result = []
	originalResult = []
	termLengthMap = {}
	tokens = nltk.word_tokenize(text)
	tagged = nltk.pos_tag(tokens)
	# NPGrammar = "NP:{<DT>?<JJ|NN|NNS>*<NN|NNS>}"
	NPGrammar = "NP:{<JJ|NN|NNS>*<NN|NNS>}"
	cp = nltk.RegexpParser(NPGrammar)
	resultTree = cp.parse(tagged)
	stemmer = SnowballStemmer("english")
	for node in resultTree:
		if (type(node) == nltk.tree.Tree):
			# if node[0][1] == 'DT':
			# 	node.remove(node[0])
			if node.leaves()[-1][0].lower() not in stop_words or len(node.leaves()) >= 2:
				# currNNs = [stemmer.stem(item[0]) for item in node.leaves()]
				currNNs = [item[0] for item in node.leaves()]
				currNPs = []

				for index, NN in enumerate(currNNs):
					# currFormedPhrase = ''.join(item for item in currNNs[index:])
					currFormedPhrase = ' '.join(item for item in currNNs[index:])
					currNPs.append(currFormedPhrase)
					originalResult.append("".join(currNNs))
					if not currFormedPhrase in termLengthMap:
						termLengthMap[currFormedPhrase] = len(currNNs) - index - 1
					# for replicateTimes in range(len(currNNs) - index - 1):
					# 	currNPs.append(''.join(item for item in currNNs[index:]))

				result.append(currNPs)
		
		#Remove the following else clause to ignore all single terms
		else:
			if node[0].lower() not in stop_words:
				# originalResult.append(stemmer.stem(node[0]))
				originalResult.append(node[0])

	return result, termLengthMap, originalResult

def getRippleScores(wordRanks, numOfWordsWanted, graph):
	'''the ripple smooths in 3 folds: 0.2, 0.6, 0.8'''
	# word_ranks = {word_rank[0]: word_rank[1]
	# 	for word_rank in sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)[:n_keywords]}
	# keywords = set(word_ranks.keys())
	result = {}
	keywordList = []
	ripple = [0.36, 0.6]
	for index in range(numOfWordsWanted):
		if len(wordRanks) > 0:
			currKeyword = sorted(wordRanks.iteritems(), key=lambda x: x[1], reverse=True)[0][0]
			currKeywordRank = sorted(wordRanks.iteritems(), key=lambda x: x[1], reverse=True)[0][1]
			keywordList.append(currKeyword)
			result[currKeyword] = currKeywordRank

			currOldValue = wordRanks[currKeyword]
			wordRanks[currKeyword] = (0.2 * currOldValue)
			openSet = [currKeyword]
			closeSet = []
			for iterIndex in range(2):
				currRipple = ripple[iterIndex]
				temp = []
				for ele in openSet:
					affectedEdges = graph.edges(ele)
					for affectedEdge in affectedEdges:
						node = affectedEdge[1]
						if node not in closeSet:
							oldValue = wordRanks[node]
							wordRanks[node] = (oldValue * currRipple)
							temp.append(node)

				closeSet += openSet
				openSet = list(temp)

	return result

def keyWordFilter(article, keywords, filterTags):
	# the idea is to use citation, reference, keyword list to find software engineering related articles
	for filterTag in filterTags:
		tagContents = article.getElementsByTagName(filterTag)

		for tagContent in tagContents:
			for keyword in keywords:
				keywordDash = '-'.join(keyword.split(' '))
				if ((keyword in tagContent.childNodes[0].data.lower()) 
					or (keywordDash in tagContent.childNodes[0].data.lower())):
					return True
	
	return False

def extract_candidate_words(text, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):
	stop_words = set(nltk.corpus.stopwords.words('english'))
	tagged_words = nltk.pos_tag(nltk.word_tokenize(text))
	candidates = [word.lower() for word, tag in tagged_words
		if tag in good_tags and word.lower() not in stop_words and len(word) > 1]
	return candidates

def getKeyphraseByTextRankFromNP(textList, termLengthMap, n_keywords=0.95, n_windowSize=3):
	graph = networkx.Graph()
	graph.add_nodes_from(set([NN for NNList in textList for NN in NNList]))
	listLength = len(textList)
	for i in range(0, n_windowSize - 1):
		for textIndex, texts in enumerate(textList):
			if textIndex + i + 1 < listLength:
				neighboringTexts = textList[textIndex + i + 1]
				# If one of the words is a single term, link only it with its neighbor's full NP
				if len(texts) == 1 or len(neighboringTexts) == 1:
					# graph.add_edge(texts[0], neighboringTexts[0], weight = 1)
					pass
				else:
					temptexts = list(texts)
					tempneighboringtexts = list(neighboringTexts)
					temptexts.pop()
					tempneighboringtexts.pop()
					# graph.add_edges_from([(texta, textb) for texta in temptexts for textb in tempneighboringtexts])
					edges = [(texta, textb) for texta in temptexts for textb in tempneighboringtexts]
					for edge in edges:
						if graph.has_edge(edge[0], edge[1]):
							graph[edge[0]][edge[1]]['weight'] += 1
						else:
							graph.add_edge(edge[0], edge[1], weight = 1)

	ranks = networkx.pagerank(graph)

	if 0 < n_keywords < 1:
		n_keywords = int(round(listLength * n_keywords))

	word_ranks = {word_rank[0]: word_rank[1]
		for word_rank in sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)[:n_keywords]}
	keywords = set(word_ranks.keys())


	return word_ranks

def getKeyphraseByTextRank(text, n_keywords=0.4, n_windowSize=4, n_cooccurSize=3):
	words = [word.lower()
		for word in nltk.word_tokenize(text)
		if len(word) > 1]
	
	candidates = extract_candidate_words(text)
	# print candidates
	graph = networkx.Graph()
	graph.add_nodes_from(set(candidates))
	
	for i in range(0, n_windowSize-1):
		def pairwise(iterable):
			a, b = tee(iterable)
			next(b, None)
			for j in range(0, i):
				next(b, None)
			return izip(a, b)
		for w1, w2 in pairwise(candidates):
			if w2:
				graph.add_edge(*sorted([w1, w2]))

	ranks = networkx.pagerank(graph)
	if 0 < n_keywords < 1:
		n_keywords = int(round(len(candidates) * n_keywords))


	MMRKeywordList = []
	word_ranks = {}
	for index in range(n_keywords):
		if len(sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)) > 0:
			currKeyword = sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)[0][0]
			currKeywordRank = sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)[0][1]
			MMRKeywordList.append(currKeyword)
			word_ranks[currKeyword] = currKeywordRank
			graph.remove_node(currKeyword)
			ranks = networkx.pagerank(graph)


	# word_ranks = getRippleScores(ranks, n_keywords, graph)
	# keywords = set(word_ranks.keys())


	# word_ranks = {word_rank[0]: word_rank[1]
	# 	for word_rank in sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)[:n_keywords]}
	# keywords = set(word_ranks.keys())


	keywords = set(MMRKeywordList)

	keyphrases = {}
	j = 0
	for i, word in enumerate(words):
		if i<j:
			continue
		if word in keywords:
			kp_words = list(takewhile(lambda x: x in keywords, words[i:i+n_cooccurSize]))
			avg_pagerank = sum(word_ranks[w] for w in kp_words) / float(len(kp_words))
			keyphrases[' '.join(kp_words)] = avg_pagerank
			if len(kp_words) > 1:
				for kpWord in kp_words:
					keyphrases[kpWord] = word_ranks[kpWord]

			j = i + len(kp_words)

	results = [(ele[0], ele[1]) for ele in sorted(keyphrases.iteritems(), key=lambda x: x[1], reverse=True)]
	return duplicateHigherRankingTerms(results)

def duplicateHigherRankingTerms(rawList): # This function actually is stemming the word in each phrase
										  # Nothing to do with the duplciation of higher ranking terms
	rawList = removeDuplicates(rawList)
	if len(rawList) < 1:
		return ""
	baseFreq = float(rawList[-1][1]) # Unused var
	result = []

	phraseScoreMap = {}
	targetTagSet = ['NN', 'NNS', 'NNP', 'NNPS']
	stemmer = SnowballStemmer("english")
	for ele in rawList:
		tempSet = removeDuplicates(ele[0].split())
		# Only consider the noun phrases
		if nltk.pos_tag(nltk.word_tokenize(tempSet[-1]))[0][1] not in targetTagSet:
			pass
		else:
			newPhrase = ''.join(stemmer.stem(wordEle) for wordEle in tempSet)
			result.append((newPhrase, ele[1]))

	phraseScoreMap = {item[0]:item[1] for item in result}

	return phraseScoreMap

def removeDuplicates(seq):
	seen = set()
	seen_add = seen.add
	return [x for x in seq if not (x in seen or seen_add(x))]

def getInfluentialPhraseScoreSeries(phraseScoreMapSeries):
	upperBound = 0.5
	lowerBound = 0.05
	yearCover = len(phraseScoreMapSeries)
	startingTemp = phraseScoreMapSeries[0]
	result = {key:[0 for n in range(yearCover)] for key in startingTemp}
	for index in range(yearCover):
		phraseScoreMap = phraseScoreMapSeries[index]
		for phrase in phraseScoreMap:
			if not result.has_key(phrase):
				result[phrase] = [0 for n in range(yearCover)]
			result[phrase][index] = phraseScoreMap[phrase]

	finalResult = {}
	for phrase in result:
		checkTimeSeries = result[phrase]
		zeroYears = np.asarray(checkTimeSeries)
		# if (zeroYears == 0).sum() >= (lowerBound * yearCover) and (zeroYears == 0).sum() <= (upperBound * yearCover):
		if (zeroYears == 0).sum() <= (upperBound * yearCover):
			finalResult[phrase] = checkTimeSeries
	return finalResult

def formRawSentences(phraseList):
	# print phraseList
	sentenceList = []
	# for ele in phraseList:
	# 	multiplyTimes = len(ele)
	# 	originalLength = len(sentenceList)
	# 	sentenceList = sentenceList * multiplyTimes
	# 	for newIndex in range(multiplyTimes):
	# 		newWord = ele[newIndex]
	# 		for oldIndex in range(originalLength):
	# 			sentenceList[oldIndex + newIndex * originalLength] += (newWord + " ")

	# sentenceList = [ele.strip() for ele in sentenceList]

	# print sentenceList

	tempSentenceList = list(product(*phraseList))
	sentenceList = [list(ele) for ele in tempSentenceList]
	# print sentenceList
	return sentenceList

def retrieveAuthors(currArticleNode):
	authorIdMap = {}
	currAuthorIDs = []

	authorList = currArticleNode.getElementsByTagName("au")
	# if authorList == "ERROR":
	# 	continue

	for author in authorList:
		try:
			firstName = author.getElementsByTagName("first_name").item(0).childNodes[0].data
			lastName = author.getElementsByTagName("last_name").item(0).childNodes[0].data
			id = author.getElementsByTagName("author_profile_id").item(0).childNodes[0].data
		#print "first name is: " + str(firstName) + " last name is: " + str(lastName) + " " + id
		except IndexError, e:
			print "Xml file author tag parsing error: %s" %e
			#continue
		except AttributeError, e1:
			print "No selected attribute detected: %s" %e1
			#continue
		else:
			fullName = firstName.split(".")[0] + " " + lastName

			currAuthorIDs.append(id)
			authorIdMap[id] = fullName

			# if not authorIdMap.has_key(id):
			# 	authorIdMap[id] = fullName
			# else:
			# 	if len(authorIdMap[id]) < len(fullName):
			# 		authorIdMap[id] = fullName
	
	return authorIdMap, currAuthorIDs

def doubleGraphRankingNPMatrixUpdate(authorTokenTupleList, windowSize = 3):
	"""
	Input data format: [([author1, author2], [set(ABC, BC, C), set(DE, E)]), ([author3, author4], [set(GH, H), set(IJK, JK, K)]), ...]
	ABC format: "software engineering project"; keep the original form, and use space for separation
	Note: no need for words to be SET, can be any iterable data types
	"""
	authors2D = [ele[0] for ele in authorTokenTupleList]
	authorsList = [inner for ele in authors2D for inner in ele]
	authors = Counter(authorsList)
	words3D = [ele[1] for ele in authorTokenTupleList]
	words2D = [s for ele in words3D for s in ele]
	wordsList = [inner for ele in words2D for inner in ele]
	words = Counter(wordsList)

	finalizedAuthors = [ele[0] for ele in sorted(authors.items(), key = operator.itemgetter(1), reverse = True)]
	finalizedWords = [ele[0] for ele in sorted(words.items(), key = operator.itemgetter(1), reverse = True)]

	authorsMatrix = np.identity(len(authors)) * 0
	wordsMatrix = np.identity(len(words)) * 0
	authorWordMatrix = np.zeros((len(authors), len(words)))

	for paper in authorTokenTupleList:
		currAuthors = paper[0]
		# paper[1]: [set(ABC, BC, C), set(DE, E)]
		currWords = [inner for ele in paper[1] for inner in ele]
		currWordsOriginal = paper[1]
		# currWords = paper[1]

		for author in currAuthors:
			authorIndex = finalizedAuthors.index(author)
			for word in currWords:
				wordIndex = finalizedWords.index(word)
				authorWordMatrix[authorIndex][wordIndex] += 1

			coauthors = list(currAuthors)
			coauthors.remove(author)

			for coauthor in coauthors:
				coauthorIndex = finalizedAuthors.index(coauthor)
				authorsMatrix[authorIndex][coauthorIndex] += 1
				# Note: no need for [coauthor][author] += 1, as it'll go through it again

		# original impl for single terms only
		# for index, word in enumerate(currWords):
		# 	wordIndex = finalizedWords.index(word)

		# 	for runningNeighbor in range(windowSize - 1):
		# 		if index + runningNeighbor + 1 < len(currWords):
		# 			neighboringWord = currWords[index + runningNeighbor + 1]

		# 			neighboringIndex = finalizedWords.index(neighboringWord)
		# 			wordsMatrix[wordIndex][neighboringIndex] += 1
		# 			wordsMatrix[neighboringIndex][wordIndex] += 1

		for index, wordSet in enumerate(currWordsOriginal):
			for word in wordSet:
				wordIndex = finalizedWords.index(word)

				for runningNeighbor in range(windowSize - 1):
					if index + runningNeighbor + 1 < len(currWordsOriginal):
						neighboringWordSet = currWordsOriginal[index + runningNeighbor + 1]

						for neighboringWord in neighboringWordSet:
							neighboringIndex = finalizedWords.index(neighboringWord)
							wordsMatrix[wordIndex][neighboringIndex] += 1
							wordsMatrix[neighboringIndex][wordIndex] += 1


	# wf-iaf construction
	# wf: # of this word by author / # of words by author
	# iaf: log(# of authors / # of authors who write this word)

	authorWordMatrix = authorWordMatrix.astype(float)
	authorWFMatrix = authorWordMatrix / authorWordMatrix.sum(axis=1)[:, None]
	numOfAuthors = authorWordMatrix.shape[0]
	numOfWords = authorWordMatrix.shape[1]
	authorIAFMatrix = np.zeros(authorWordMatrix.shape)
	for colIndex in range(numOfWords):
		column = authorWordMatrix[:, colIndex]
		iafValue = 0
		if np.count_nonzero(column) != 0:
			iafValue = math.log((float(numOfAuthors) / np.count_nonzero(column)), 2)
		authorIAFMatrix[:, colIndex] += iafValue

	wfIafMatrix = authorWFMatrix * authorIAFMatrix

	finalAuthorScores = np.ones(numOfAuthors)
	finalWordScores = np.ones(numOfWords)
	alpha = 0.5
	beta = 1 - alpha # beta is the coefficient for author-word edge weight

	for iterIndex in range(1000):
		finalAuthorScores = alpha * authorsMatrix.dot(finalAuthorScores) + beta * wfIafMatrix.dot(finalWordScores)
		finalAuthorScores = normalize(finalAuthorScores)
		finalWordScores = alpha * wordsMatrix.dot(finalWordScores) + beta * wfIafMatrix.T.dot(finalAuthorScores)
		finalWordScores = normalize(finalWordScores)


	authorScoreMap = {}
	wordScoreMap = {}
	for index in range(len(finalizedAuthors)):
		authorScoreMap[finalizedAuthors[index]] = finalAuthorScores[index]

	for index in range(len(finalizedWords)):
		wordScoreMap[finalizedWords[index]] = finalWordScores[index]

	print "Finished HITS processing!"

	return authorScoreMap, wordScoreMap

def doubleGraphRankingMatrixUpdate(authorTokenTupleList, windowSize = 3):
	"""
	Input data format: [([author1, author2], [token1, token2]), ([author3, author4], [token3, token4]), ...]
	"""
	authors2D = [ele[0] for ele in authorTokenTupleList]
	authorsList = [inner for ele in authors2D for inner in ele]
	authors = Counter(authorsList)
	words2D = [ele[1] for ele in authorTokenTupleList]
	wordsList = [inner for ele in words2D for inner in ele]
	words = Counter(wordsList)

	finalizedAuthors = [ele[0] for ele in sorted(authors.items(), key = operator.itemgetter(1), reverse = True)]
	finalizedWords = [ele[0] for ele in sorted(words.items(), key = operator.itemgetter(1), reverse = True)]

	authorsMatrix = np.identity(len(authors)) * 0
	wordsMatrix = np.identity(len(words)) * 0
	authorWordMatrix = np.zeros((len(authors), len(words)))

	for paper in authorTokenTupleList:
		currAuthors = paper[0]
		currWords = paper[1]

		for author in currAuthors:
			authorIndex = finalizedAuthors.index(author)
			for word in currWords:
				wordIndex = finalizedWords.index(word)
				authorWordMatrix[authorIndex][wordIndex] += 1

			coauthors = list(currAuthors)
			coauthors.remove(author)

			for coauthor in coauthors:
				coauthorIndex = finalizedAuthors.index(coauthor)
				authorsMatrix[authorIndex][coauthorIndex] += 1
				# Note: no need for [coauthor][author] += 1, as it'll go through it again

		for index, word in enumerate(currWords):
			wordIndex = finalizedWords.index(word)

			for runningNeighbor in range(windowSize - 1):
				if index + runningNeighbor + 1 < len(currWords):
					neighboringWord = currWords[index + runningNeighbor + 1]

					neighboringIndex = finalizedWords.index(neighboringWord)
					wordsMatrix[wordIndex][neighboringIndex] += 1
					wordsMatrix[neighboringIndex][wordIndex] += 1


	# wf-iaf construction
	# wf: # of this word by author / # of words by author
	# iaf: log(# of authors / # of authors who write this word)

	authorWordMatrix = authorWordMatrix.astype(float)
	authorWFMatrix = authorWordMatrix / authorWordMatrix.sum(axis=1)[:, None]
	numOfAuthors = authorWordMatrix.shape[0]
	numOfWords = authorWordMatrix.shape[1]
	authorIAFMatrix = np.zeros(authorWordMatrix.shape)
	for colIndex in range(numOfWords):
		column = authorWordMatrix[:, colIndex]
		iafValue = 0
		if np.count_nonzero(column) != 0:
			iafValue = math.log((float(numOfAuthors) / np.count_nonzero(column)), 2)
		authorIAFMatrix[:, colIndex] += iafValue

	wfIafMatrix = authorWFMatrix * authorIAFMatrix

	finalAuthorScores = np.ones(numOfAuthors)
	finalWordScores = np.ones(numOfWords)
	alpha = 0.5
	beta = 1 - alpha # beta is the coefficient for author-word edge weight

	for iterIndex in range(5000):
		finalAuthorScores = alpha * authorsMatrix.dot(finalAuthorScores) + beta * wfIafMatrix.dot(finalWordScores)
		finalAuthorScores = normalize(finalAuthorScores)
		finalWordScores = alpha * wordsMatrix.dot(finalWordScores) + beta * wfIafMatrix.T.dot(finalAuthorScores)
		finalWordScores = normalize(finalWordScores)


	authorScoreMap = {}
	wordScoreMap = {}
	for index in range(len(finalizedAuthors)):
		authorScoreMap[finalizedAuthors[index]] = finalAuthorScores[index]

	for index in range(len(finalizedWords)):
		wordScoreMap[finalizedWords[index]] = finalWordScores[index]

	print "Finished HITS processing!"

	return authorScoreMap, wordScoreMap

def doubleGraphRankingUpdate(authorTokenTupleList, windowSize = 3):
	"""
	Input data format: [([author1, author2], [token1, token2]), ([author3, author4], [token3, token4]), ...]
	"""
	currAuthorWordsMap = {}
	currWordAuthorsMap = {}
	currCoauthorMap = {}
	currCowordMap = {}
	for paper in authorTokenTupleList:
		authors = paper[0]
		words = paper[1]
		for author in authors:
			if author in currAuthorWordsMap.keys():
				currAuthorWordsMap[author] += words
			else:
				currAuthorWordsMap[author] = words

			coauthors = list(authors)
			coauthors.remove(author)

			if author in currCoauthorMap.keys():
				currCoauthorMap[author] += coauthors
			else:
				currCoauthorMap[author] = coauthors

		for index, word in enumerate(words):
			if word in currWordAuthorsMap.keys():
				currWordAuthorsMap[word] += authors
			else:
				currWordAuthorsMap[word] = authors

			startingIndex = index - (windowSize - 1) if (index - (windowSize - 1)) >= 0 else 0
			# endingIndex = index + (windowSize - 1) if (index + (windowSize - 1)) < len(words) else (len(words) - 1)
			endingIndex = startingIndex + windowSize * 2 - 2
			cowords = words[startingIndex : endingIndex + 1]
			cowords.remove(word)

			if word in currCowordMap.keys():
				currCowordMap[word] += cowords
			else:
				currCowordMap[word] = cowords

	"""
	form the wf-iaf relationship
	wf: # of this word by author / # of words by author
	iaf: log(# of authors / # of authors who write this word)
	"""
	finalAuthorScore = {author: 1 for author in currAuthorWordsMap.keys()}
	finalWordScore = {word: 1 for word in currWordAuthorsMap.keys()}

	numOfAuthors = len(finalAuthorScore)

	wfIafMap = {}

	alpha = 0.5
	beta = 1 - alpha # beta is the coefficient for author-word graph edge

	for author in finalAuthorScore:
		hisWords = currAuthorWordsMap[author]
		hisVocab = Counter(currAuthorWordsMap[author])
		for token in hisVocab:
			numOfIdentifiedAuthors = len(Counter(currWordAuthorsMap[token]))
			wf = float(hisVocab[token]) / len(hisWords)
			iaf = math.log((numOfAuthors / float(numOfIdentifiedAuthors)), 2)
			wfIafMap[(author, token)] = wf * iaf

	# This part is to ensure each author-word pair has been iterated through
	# Since for now only the above for loop will result in missing keys in wf-iaf map
	for word in finalWordScore:
		itsAuthors = Counter(currWordAuthorsMap[word])
		for author in itsAuthors:
			hisWords = currAuthorWordsMap[author]
			hisVocab = Counter(hisWords)

			numOfIdentifiedAuthors = len(itsAuthors)
			wf = float(hisVocab[word]) / len(hisWords)
			iaf = math.log((numOfAuthors / float(numOfIdentifiedAuthors)), 2)
			wfIafMap[(author, word)] = wf * iaf

	for iterIndex in range(2):
		# author score update
		currAuthorScore = dict(finalAuthorScore)
		norm = 0.0
		for author in finalAuthorScore:
			hisVocab = Counter(currAuthorWordsMap[author])
			hisCoauthors = Counter(currCoauthorMap[author])
			hisAuthorScore = 0.0
			for coauthor in hisCoauthors:
				hisAuthorScore += (hisCoauthors[coauthor] * finalAuthorScore[coauthor])
			hisWordScore = 0.0
			for token in hisVocab:
				# print token
				hisWordScore += (wfIafMap[(author, token)] * finalWordScore[token])
			hisFinalScore = alpha * hisAuthorScore + beta * hisWordScore
			norm += hisFinalScore ** 2
			currAuthorScore[author] = hisFinalScore

		norm = math.sqrt(norm)
		finalAuthorScore = {author: (score / norm) for author, score in currAuthorScore.items()}

		# word score update
		norm = 0.0
		currWordScore = dict(finalWordScore)
		for word in finalWordScore:
			itsAuthors = Counter(currWordAuthorsMap[word])
			itsCoword = Counter(currCowordMap[word])
			itsWordScore = 0.0
			for coword in itsCoword:
				itsWordScore += (itsCoword[coword] * finalWordScore[coword])
			itsAuthorScore = 0.0
			for author in itsAuthors:
				print word
				if (author, word) in wfIafMap:
					itsAuthorScore += (wfIafMap[(author, word)] * finalAuthorScore[author])
				else:
					print "Not found!"
			itsFinalScore = alpha * itsWordScore + beta * itsAuthorScore
			norm += itsFinalScore ** 2
			currWordScore[word] = itsFinalScore

		norm = math.sqrt(norm)
		finalWordScore = {word: (score / norm) for word, score in currWordScore.items()}


	print "Finished current year HITS processing!"
	return finalAuthorScore, finalWordScore

def normalize(vector):
	vector = np.asarray(vector)
	norm = np.linalg.norm(vector)
	if norm == 0:
		return vector
	return vector / norm

if (__name__ == '__main__'):
	fileList = os.listdir('.')
	targetList = []
	for fileName in fileList:
		if fileName.endswith('.xml'):
			targetList.append(fileName)

	scoreList = {}
	authorIdMap = {}
	count = 0
	flag9000_1 = False
	flag9000_2 = False

	predefinedNumberOfVIPAuthors = 300
	predefinedNumberOfVIPPhrases = 300
	commonPhrasesRecognitionCriteria = int(0.25 * predefinedNumberOfVIPAuthors)
	keywords = ["software engineering", "software and its engineering"]
	# keywords = ["database", "storage"]
	# keywords = ["www", "search engine", "world wide web", "information retrieval", "internet"]
	filterList = ["kw", "ref_text", "cited_by_text", "concept_desc", "subtitle"]


	vocabOccurrenceMapSeries = [{} for n in range(70)]
	vocabTextRankMapSeries = [{} for n in range(70)]
	rawTextSeries = [[] for n in range(70)]
	oriTextSeries = [[] for n in range(70)]
	baseYear = 1950

	count = 0

	"""
	author-words mapping var
	"""

	authorsRawTextsListSeries = [[] for n in range(70)]

	authorIDMapFull = {}


	"""
	End of author-words mapping var
	"""


	smallestYear = 1960
	for target in targetList:

		articleList = XmlParsing(target, "article_rec")
		if articleList == "ERROR":
			continue

		# This part is to check whether all files have been iterated through
		count += 1

		print 'Currently processing: %d/9719' % count

		for article in articleList:

			if keyWordFilter(article, keywords, filterList):
				# Retrieve author information

				currAuthorIDMap, authorList = retrieveAuthors(article)

				for authorID in currAuthorIDMap.keys():
					if authorID not in authorIDMapFull.keys():
						authorIDMapFull[authorID] = currAuthorIDMap[authorID]

				# Finished retrieving author information

				timeStamp = int(article.getElementsByTagName("article_publication_date").item(0).childNodes[0].data.split("-")[2])
				if timeStamp < smallestYear:
					smallestYear = timeStamp

				offset = timeStamp - baseYear
				
				vocabTextRankMap = vocabTextRankMapSeries[offset]
				currRawTextList = rawTextSeries[offset]
				currOriTextList = oriTextSeries[offset]

				currAuthorsWordsTupleList = authorsRawTextsListSeries[offset]


				# currEmbRawtextList = embeddingRawTextListInYears[offset]

				abstract = article.getElementsByTagName("par")
				# abstract = article.getElementsByTagName("ft_body")
				
				if len(abstract) > 0 and len(abstract.item(0).childNodes) > 0:
					abstract = abstract.item(0).childNodes[0].data
					abstract = re.sub(r'<.*?>', "", abstract)
					abstract = re.sub(r'\"', "", abstract)

					abstract = str(abstract.encode('utf-8')).translate(None, string.punctuation)
					abstract = ''.join([i for i in abstract if not i.isdigit()])

					"""
					logic: obtain NN and JJ tokens first, and directly form the map
					"""
					qualifiedTokens = extract_candidate_words(abstract)

					currRawTextList.append(" ".join(qualifiedTokens))

					# currTuple = (authorList, qualifiedTokens)
					# currAuthorsWordsTupleList.append(currTuple)

					currNPList, termLength, originalMap = generateNPTextList(abstract)
					currTupleNP = (authorList, currNPList)
					currAuthorsWordsTupleList.append(currTupleNP)

	authorsRawTextsListSeries = [ele for ele in authorsRawTextsListSeries if bool(ele)]
	vocabHITSMapSeries = []
	authorsHITSSeries = []
	for currAuthorRawTextsList in authorsRawTextsListSeries:
		currAuthorScore, currWordScore = doubleGraphRankingNPMatrixUpdate(currAuthorRawTextsList)
		topAuthorIDs = [ele[0] for ele in sorted(currAuthorScore.items(), key = operator.itemgetter(1), reverse = True)[:10]]
		topAuthorNames = [authorIDMapFull[ele] for ele in topAuthorIDs]
		vocabHITSMapSeries.append(currWordScore)
		authorsHITSSeries.append(topAuthorNames)

	finalVocabTextrankMapSeriesMap = getInfluentialPhraseScoreSeries(vocabHITSMapSeries)

	"""
	TF-IDF component
	"""
	rawTextSeries = [ele for ele in rawTextSeries if bool(ele)]
	vocabTFIDFMapSeries = [{} for n in range(len(rawTextSeries))]
	for SeriesIndex in range(len(rawTextSeries)):
		sklearn_tfidf = TfidfVectorizer(norm="l2", min_df = 0, use_idf=True, smooth_idf = False)
		scores = sklearn_tfidf.fit_transform(rawTextSeries[SeriesIndex]).toarray()
		rawTextList = rawTextSeries[SeriesIndex]
		# scores = tfidfScores[SeriesIndex].toarray()
		# scores = np.add(tfidfScores[SeriesIndex].toarray())

		# sumScores = scores[0]
		sumScores = scores.mean(axis = 0)
		vocabTFIDFMap = vocabTFIDFMapSeries[SeriesIndex]
		vocabMap = sklearn_tfidf.vocabulary_
		for vocab in vocabMap:
			vocabIndex = vocabMap[vocab]
			vocabTFIDFMap[vocab] = sumScores[vocabIndex]
	finalVocabTFIDFMapSeriesMap = getInfluentialPhraseScoreSeries(vocabTFIDFMapSeries)

	"""
	End of TF-IDF
	"""

	writeMapToFile(finalVocabTextrankMapSeriesMap, 'process/doublegraph-NPs.txt')
	write2DListVertical(authorsHITSSeries, 'process/top-authors-NPs.txt')
	writeMapToFile(finalVocabTFIDFMapSeriesMap, 'process/doublegraph-vocab.txt')

	print 'End processing!'

	print 'First year: '
	print smallestYear

	pass
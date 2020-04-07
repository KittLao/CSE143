import math
import sys, getopt, os

START_WORD = "<START>"
STOP_WORD = "<STOP>"
UNKOWN_WORD = "<UNK>"

def readFile(filename):
	file = open(filename)
	data = [START_WORD + " " + sentence[:len(sentence) - 1] + " " + STOP_WORD for sentence in file]
	file.close()
	return data

def tokenize(sentences):
	return [sentence.split(" ") for sentence in sentences]

def createTokenBank(token_sents):
	token_bank = {}
	token_count = 0
	for token_sent in token_sents:
		for token in token_sent:
			if token not in token_bank:
				token_bank[token] = 1
			else:
				token_bank[token] += 1
			token_count += 1;
	return (token_bank, token_count)

def replaceWithUNK1(token_sents, token_bank, unk_threshold):
	for i in range(0, len(token_sents)):
		for j in range(0, len(token_sents[i])):
			if token_bank[token_sents[i][j]] < unk_threshold:
				token_sents[i][j] = UNKOWN_WORD
	return token_sents

def replaceWithUnk2(token_sents, train_data, unknown_word):
	for i in range(0, len(token_sents)):
		for j in range(0, len(token_sents[i])):
			if token_sents[i][j] not in train_data:
				token_sents[i][j] = unknown_word
	return token_sents

def createNgrams(token_sents_unk, n):
	n_grams_sents = []
	for token_sent_unk in token_sents_unk:
		n_grams_sent = []
		if len(token_sent_unk) < n:
			continue
		for i in range(0, len(token_sent_unk) - n + 1):
			n_grams = tuple(token_sent_unk[j] for j in range(i, i + n))
			n_grams_sent.append(n_grams)
		n_grams_sents.append(n_grams_sent)
	return n_grams_sents

"""
Inputs:
  token_bank: dictionary to hold the frequency of a certain token.
  cond_token_bank: dictionary to hold the frequency of the previous
  n - 1 words.
  n_grams: the data being calculated on.
  M: number of words in the testing data.
  N: number of words in the training data.
  n: gram.
"""
def computeLogProbability(token_bank, cond_token_bank, n_grams, N, n):
	log_lik = []
	for n_gram in n_grams:
		log_lik_s = []
		for token in n_gram:
			"""
			If unigram and encountered stop or start word,
			probability is 1. Of if gram isn't present.
			"""
			if (token[0] == STOP_WORD and n == 1):
				log_lik_s.append(0)
				continue
			if token not in token_bank:
				log_lik_s.append(float("inf"))
				continue
			"""
			If it's unigram, denominator should be number of
			words in train data. Otherwise, its the frequency
			of the previous n-1 words.
			"""
			if n > 1:
				N = cond_token_bank[token[:n - 1]]
			log_pr_token = math.log(token_bank[token] / N, 2)
			log_lik_s.append(log_pr_token)
		log_lik.append(log_lik_s)
	return log_lik

"""
If perplexity is 0, then it is infinity.
"""
def computePerplexity(log_liks, M):
	log_lik = sum([sum(log_lik_s) for log_lik_s in log_liks])
	perplexity = math.pow(2, -1 * log_lik / M)
	return float("inf") if perplexity == 0 else perplexity

def test(train_data, testing_data, word_count_test, word_count_train, n):
	result = []
	log_liks = []
	for i in range(n):
		log_lik = computeLogProbability(train_data[i + 1], train_data[i], testing_data[i], word_count_train, i + 1)
		perplexity = computePerplexity(log_lik, word_count_test)
		result.append(perplexity)
		log_liks.append(log_lik)
	return (result, log_liks)

"""
unigram: remove start word.
bigram: no change.
trigram: add prob of first 2 words in each sentence.

Produce a list of tuples, where the tuples contains log likelihood of
the sentences. Apply the smoothing parameters to each log likelihood
of the words. Make sure to convert the log likelihood to probabilities
first before multiplying by the parameters and adding them up. If the
log likelihood is inf, don't take the log of it. Just make  it 0.
"""
def smoothing(log_liks, lams):
	log_liks[0] = [log_lik_s[1:] for log_lik_s in log_liks[0]]
	log_liks[2] = [[log_lik_s2[0]] + log_lik_s3 for log_lik_s2, log_lik_s3 in zip(log_liks[1], log_liks[2])]
	dotProd = lambda x, y: sum([x[i] * y[i] for i in range(len(x))])
	exponentiate = lambda x: 0 if x == float("inf") else math.pow(2, x)
	take_log_2 = lambda x: float("inf") if x == 0 else math.log(x, 2)
	smoothed_log_liks = []
	list_of_tuple_of_sent = [tuple(log_liks[j][i] for j in range(3)) for i in range(len(log_liks[0]))]
	for tuple_of_sent in list_of_tuple_of_sent:
		smoothed_sent = [take_log_2(dotProd(list(map(exponentiate, pr_words)), lams)) for pr_words in zip(*tuple_of_sent)]
		smoothed_log_liks.append(smoothed_sent)
	return smoothed_log_liks

def applySmoothing(log_liks, lams, M):
	smoothed_log_lik = smoothing(log_liks, lams)
	p = computePerplexity(smoothed_log_lik, M)
	return p

def displayResults(result, type, smoothed_result, lams):
	print(type + ":")
	for i in range(len(result)):
		print(i + 1, "-gram: ", result[i])
	print("Smoothed with l1 =", lams[0], "l2 =", lams[1], "l3=", lams[2], ":", smoothed_result)

def main():
	"""
	These are the default parameters.
	"""
	n = 3
	unk_threshold = 3
	token_banks = {0 : {}}
	lams = [0.1, 0.3, 0.6]
	train_data_percent = 100

	argv = sys.argv[1::]
	opts, args = getopt.getopt(argv, "", ["unk_threshold=", "train_data_percent=", "l1=", "l2=", "l3="])
	for opt, arg in opts:
		if opt == "--unk_threshold":
			unk_threshold = int(arg)
		if opt == "--train_data_percent":
			train_data_percent = int(arg)
		if opt == "--l1":
			lams[0] = float(arg)
		if opt == "--l2":
			lams[1] = float(arg)
		if opt == "--l3":
			lams[2] = float(arg)
	"""
	Getting training, testing, and development result all requires reading data and
	preprocessing it. Preprocessing involves tokenizing the sentences and replacing
	some words with UNK's. More preprocessing will need to be done for each n-gram.
	"""
	sentences = readFile("A1-Data/1b_benchmark.train.tokens")
	instances = len(sentences)
	sentences = sentences[:int(train_data_percent / 100 * instances)]
	token_sents = tokenize(sentences)
	token_bank, token_count = createTokenBank(token_sents)
	token_sents_unk = replaceWithUNK1(token_sents, token_bank, unk_threshold)

	"""
	Keep track of training data acnd number of words in training data
	for test and dev.
	"""
	train_data, __ = createTokenBank(token_sents_unk) # training data for word frequency
	train_data_words = token_count - instances

	print("Words in training data with <STOP>: ", train_data_words)
	print("Unique words including <STOP> and <UNK>", len(train_data) - 1)

	"""
	Training result:
	Using the tokenized sentences that contains UNK's, generate n-grams for it
	and store it. Create a token bank from the n-gram, which are used as part
	of the training data for the test and dev.
	"""
	n_grams = []
	for i in range(n):
		n_gram = createNgrams(token_sents_unk, i + 1)
		token_banks[i + 1], __ = createTokenBank(n_gram)
		n_grams.append(n_gram)
	train_result, train_log_liks = test(token_banks, n_grams, train_data_words, train_data_words, n)
	smoothed_train_result = applySmoothing(train_log_liks, lams, train_data_words)
	displayResults(train_result, "Training", smoothed_train_result, lams)

	"""
	Testing result
	"""
	sentences = readFile("A1-Data/1b_benchmark.test.tokens")
	instances = len(sentences)
	token_sents = tokenize(sentences)
	token_bank, token_count = createTokenBank(token_sents)
	token_sents_unk = replaceWithUnk2(token_sents, train_data, UNKOWN_WORD)

	n_grams = [createNgrams(token_sents_unk, i + 1) for i in range(n)]
	test_result, test_log_liks = test(token_banks, n_grams, token_count - instances, train_data_words, n)
	smoothed_test_result = applySmoothing(test_log_liks, lams, token_count - instances)
	displayResults(test_result, "Testing", smoothed_test_result, lams)

	"""
	Dev result
	"""
	sentences = readFile("A1-Data/1b_benchmark.dev.tokens")
	instances = len(sentences)
	token_sents = tokenize(sentences)
	token_bank, token_count = createTokenBank(token_sents)
	token_sents_unk = replaceWithUnk2(token_sents, train_data, UNKOWN_WORD)

	n_grams = [createNgrams(token_sents_unk, i + 1) for i in range(n)]
	dev_result, dev_log_liks = test(token_banks, n_grams, token_count - instances, train_data_words, n)
	smoothed_dev_result = applySmoothing(dev_log_liks, lams, token_count - instances)
	displayResults(dev_result, "Development", smoothed_dev_result, lams)

main()

















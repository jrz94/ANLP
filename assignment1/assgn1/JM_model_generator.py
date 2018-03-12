import re, string
import sys, math, operator
import numpy as np
import mixem
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
from decimal import Decimal

alphabet = "qwertyuiopasdfghjklzxcvbnm0. #"
all_trigram = [''.join(i) for i in itertools.product(alphabet, repeat = 3)]

def strip_non_ascii(string):
    ''' 
    strip_non_ascii takes a string need to be modified and 
    returns that string without non ASCII characters.
    '''
    stripped = (c for c in string if 0 < ord(c) < 127 and c is not "\n")
    return ''.join(stripped)


def file_len(fname):
    '''
    file_len takes a file and returns the number of lines in that file.
    '''
    for i, l in enumerate(fname):
        pass
    return i + 1


def split(a, n):
    '''
    split takes a list and the number of chunks need to be splited and 
    returns the splited list. 
    '''
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def sort_and_chunk(dictionary, n):
    '''
    sort_and_chunk takes a dictionary and the number of chunks and returns
    a list of tuple sorted by n.
    '''
    return list(split(sorted(dictionary.items(), key=operator.itemgetter(1)), n))


def preprocess_line(line):
    '''
    preprocess_line takes a line in the file and returns a new line without
    characters with accents and umlauts and the punctuation marks excpt period.
    It also lowercases all capital letters and converts digits to 0.
    '''
    line = "##" + re.sub("\d", "0",strip_non_ascii(line.lower()).translate(str.maketrans('', '', string.punctuation.replace(".", "")))) + "#"
    return line


def n_gram_count(n, file, length):
    '''
    n_gram_count takes the n of n-gram, an input file and the length we use to 
    train lambda and returns two dictionaries containing the counts of the n-grams, 
    one for calculating the likelihood and one for train the lambdas.
    '''
    counts = [defaultdict(int), defaultdict(int)]
    held_out_index = int(length*0.7)
    index_flag = 0
    line_count=0
    for line in file:
        line = preprocess_line(line)
        index_flag = 1 if line_count > held_out_index else 0
        for j in range(len(line)-(n-1)):
            gram = line[j:j+n]
            counts[index_flag][gram] += 1
        line_count += 1
    return counts


def generate_count_dict():
    '''
    generate_count_dict takes the training file from users and returns trigram,
    bigram, unigram dictionaries.
    '''
    if len(sys.argv) != 2:
        print("Usage: ", sys.argv[0], "<training_file>")
        sys.exit(1)
    infile = sys.argv[1] 
    with open(infile) as f:
        result_1 = []
        result_2 = []
        length = file_len(f)
        for i in range(1,4):
            f.seek(0) #reset the file read
            dicts =n_gram_count(i,f, length)
            result_1.append(dicts[0])
            result_2.append(dicts[1])
        return result_1, result_2


def get_likelihood(trigram, bigram, unigram, count):
    '''
    get_likelihood takes the number of trigram, bigram, unigram 
    and the size of corpus and returns to a tuple of likelihoods
    for each gram.
    '''
    return tuple((trigram/bigram, bigram/unigram, unigram/count))



def train_lambda(data):
    '''
    train_lambda takes training data and returns the lambdas which will used 
    in the EM algorithm.
    We use a implementation of the Expectation-Maximization (EM) algorithm
    called "mixem" to tune the parameters lambda in the interpolation.
    https://pypi.python.org/pypi/mixem
    '''
    weights, distributions, ll = mixem.em(np.sort(np.array(data)), [mixem.distribution.NormalDistribution(0,1),mixem.distribution.NormalDistribution(0.3,5),mixem.distribution.NormalDistribution(1,9)])
    return weights


def J_M_interpolation(dicts):
    '''
    J_M_interpolation takes a dictionary(containing the counts of trigram,
    bigram, unigram and also corresponding lambdas)and returns to the probability
    after interpolation smoothing.
    '''
    uni_counts, bi_counts, tri_counts = dicts[0]
    uni_counts_lamda, bi_counts_lamda, tri_counts_lamda = dicts[1]
    probabilities = []
    lambdas = []
    v1 = sum(uni_counts.values())
    v2 = sum(uni_counts_lamda.values())

    #bucketing -> Train the lambda
    for chunk in sort_and_chunk(tri_counts_lamda, 10):
        data = []
        #Create the input vector for train the lambdas
        for trigram in chunk:
            key=trigram[0]
            likelihood = get_likelihood(tri_counts_lamda[key],bi_counts_lamda[key[:-1]], uni_counts_lamda[key[:-2]], v2)
            data.append(likelihood[0])
            data.append(likelihood[1])
            data.append(likelihood[2])
        lambdas.append(train_lambda(np.array(data)))

    index = 0
    #Interpolate!
    for chunk in sort_and_chunk(tri_counts, 10):
        for trigram in chunk:
            key=trigram[0]
            likelihood = get_likelihood(tri_counts[key],bi_counts[key[:-1]], uni_counts[key[:-2]], v1)
            probabilities.append([key,lambdas[index][0]*likelihood[0] + lambdas[index][1]*likelihood[1] + lambdas[index][2]*likelihood[2] ])

    #calculate probability for all unseen trigram
    for trigram in all_trigram:
        if trigram not in tri_counts.keys():
            if bi_counts_lamda[trigram[:-1]] == 0:
                probability = [trigram, 0.05*uni_counts[trigram[:-2]]/v1]
            else:
                probability = [trigram, 0.05*uni_counts[trigram[:-2]]/v1 + 0.07*bi_counts_lamda[trigram[:-1]]/uni_counts[trigram[:-2]]]
            probabilities.append(probability)

    print (lambdas)
    return probabilities



def generate_JM_model(model_name):
    '''
    generate_JM_model takes a name string and returns a model named by that
    string and generated by our method(interpolation).
    '''
    probabilities = J_M_interpolation(generate_count_dict())
    with open(model_name, "w") as f:
        for couple in probabilities:
            f.write(couple[0]+"\t"+ ('%.2E' % Decimal(couple[1]) +"\n"))


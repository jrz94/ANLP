import re, string
import sys, math, operator
from collections import defaultdict
from decimal import Decimal
import numpy as np
import mixem
import matplotlib.pyplot as plt
import itertools

alphabet = "qwertyuiopasdfghjklzxcvbnm0. #"
all_trigram = [''.join(i) for i in itertools.product(alphabet, repeat = 3)]

#Remove non-ASCII characters
def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127 and c is not "\n")
    return ''.join(stripped)

#Count the number of lines in a file
def file_len(fname):
    for i, l in enumerate(fname):
        pass
    return i + 1

#split a list in n chunks
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

#Converts a dictionary into a list of tuple sorted by the second element and return the list splitted in n chunks
def sort_and_chunk(dictionary, n):
    return list(split(sorted(dictionary.items(), key=operator.itemgetter(1)), n))

#Lowecase all character, remove non ascii characters, then punctiations (except full stop), converts all digits to 0 and add the initials and final marks
def preprocess_line(line):
    line = "##" + re.sub("\d", "0",strip_non_ascii(line.lower()).translate(str.maketrans('', '', string.punctuation.replace(".", "")))) + "#"
    return line

#Returns two dictionaries containing the counts of the n-grams, one for calculating the likelihood and one for train the lambdas
def n_gram_count(n, file, length):
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

#Load the training file and return 3-2-1 counts dictionaries
def generate_count_dict():
    if len(sys.argv) != 2:
        print("Usage: ", sys.argv[0], "<training_file>")
        sys.exit(1)
    infile = sys.argv[1] #get input argument: the training file

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
    return tuple((trigram/bigram, bigram/unigram, unigram/count))



#Apply EM Algorithm to calculate the optimal lambdas
def train_lambda(data):
    # plt.scatter(np.array(range(data.shape[0])), data)
    # plt.show();
    weights, distributions, ll = mixem.em(np.sort(np.array(data)), [mixem.distribution.NormalDistribution(0,1),mixem.distribution.NormalDistribution(0.3,5),mixem.distribution.NormalDistribution(1,9)])
    return weights


def J_M_interpolation(dicts):
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
        #break
    return probabilities



def generate_JM_model(model_name):
    probabilities = J_M_interpolation(generate_count_dict())
    with open(model_name, "w") as f:
        for couple in probabilities:
            f.write(couple[0]+"\t"+ ('%.2E' % Decimal(couple[1]) +"\n"))

def generate_MLE_model():
    tri_counts, bi_counts, s_counts = generate_count_dict()
    with open("my_model_MLE.en", "w") as f:
        for key in sorted(tri_counts.keys()):
            #Write key ,tab> likelihood in the file
            f.write(key+"\t"+ ('%.2E' % Decimal( tri_counts[key] / bi_counts[key[:-1]]) +"\n"))
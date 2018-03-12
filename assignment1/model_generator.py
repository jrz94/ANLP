#Here are some libraries you're likely to use. You might want/need others as well.
import re, string
import sys, math
from collections import defaultdict
from decimal import Decimal
import numpy as np
import mixem

#Remove non-ASCII characters
def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127 and c is not "\n")
    return ''.join(stripped)

#Lowecase all character, remove non ascii characters, then punctiations (except full stop), converts all digits to 0 and add the initials and final marks
def preprocess_line(line):
    line = "##" + re.sub("\d", "0",strip_non_ascii(line.lower()).translate(str.maketrans('', '', string.punctuation.replace(".", "")))) + "#"
    return line

#Returns a dictionary containing the counts of the n-grams
def n_gram_count(n, file):
    counts = defaultdict(int)
    for line in file:
        line = preprocess_line(line)
        for j in range(len(line)-(n-1)):
            gram = line[j:j+n]
            counts[gram] += 1
    return counts

#Load the training file and return 3-2-1 counts dictionaries
def generate_count_dict():
    if len(sys.argv) != 2:
        print("Usage: ", sys.argv[0], "<training_file>")
        sys.exit(1)
    infile = sys.argv[1] #get input argument: the training file

    with open(infile) as f:
        result = []
        for i in range(1,4):
            result.append(n_gram_count(i,f))
            f.seek(0) #reset the file read
        return result

#Apply EM Algorithm to calculate the optimal lambdas
def train_lambda(data):
    weights, distributions, ll = mixem.em(np.array(data), [mixem.distribution.NormalDistribution(0,1),mixem.distribution.NormalDistribution(3,4),mixem.distribution.NormalDistribution(7,8)])
    return weights

# #Returns a list of (trigram, probability)

def J_M_interpolation(tri_counts, bi_counts, uni_counts):
    probabilities = []
    for key in tri_counts.keys():
        data = []
        data.append(tri_counts[key]/bi_counts[key[:-1]])
        data.append(bi_counts[key[:-1]]/uni_counts[key[:-2]])
        data.append(uni_counts[key[:-2]]/sum(uni_counts.values()))
        lambdas = train_lambda(np.array(data))
        if math.isnan(lambdas[0]):
            print ("NaN!")
            lambdas=[0.8,0.15,0.5]
        #interpolation
        #TODO: To calculate the probability we should nt use the same data we used to train the model!!!
        probabilities.append([key,lambdas[0]*data[0] + lambdas[1]*data[1] + lambdas[2]*data[2] ])
        print (lambdas, lambdas[0]*data[0] + lambdas[1]*data[1] + lambdas[2]*data[2], data[0])
        #break
    return probabilities


def generate_JM_model():
    uni_counts, bi_counts, tri_counts = generate_count_dict()
    probabilities = J_M_interpolation(tri_counts, bi_counts, uni_counts)
    with open("my_JM_model.en", "w") as f:
        for couple in probabilities:
            f.write(couple[0]+"\t"+ ('%.2E' % Decimal(couple[1]) +"\n"))

def generate_MLE_model():
    tri_counts, bi_counts, s_counts = generate_count_dict()
    with open("my_model_MLE.en", "w") as f:
        for key in sorted(tri_counts.keys()):
            #Write key ,tab> likelihood in the file
            f.write(key+"\t"+ ('%.2E' % Decimal( tri_counts[key] / bi_counts[key[:-1]]) +"\n"))


#Here are some functions that I though I needed but turned

# def concatenate(x,y):
#     return np.concatenate((x,y),axis=1)
#
# #Return the numpy array of dictionary values
# def convert_dict(dictionary):
#     return np.array(list(dictionary.values()))
#
# #Given an n-gram and an n-1_gram dictionary, return a (N,2) numpy array of N (n-gran count, n-1 ram count) elements
# def get_n_1_gram_array(n_counts, n_1_counts):
#     n_1_list = []
#     for n_gram in n_counts.keys():
#         n_1_list.appent(n_1_counts[n_gram[:-1]])
#     return concatenate(convert_dict(n_counts), np.array(n_1_list).T)
#
# def get_n_1_gram_array(n_gram, n_1_counts):
#     return n_1_counts[n_gram[:-1]]
#     return concatenate(convert_dict(n_counts), np.array(n_1_list).T)
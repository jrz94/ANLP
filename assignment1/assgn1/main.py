import re
import numpy as np
import math
from collections import defaultdict
from numpy.random import random_sample
from JM_model_generator import  generate_JM_model,preprocess_line
from decimal import Decimal
from math import log

model_dict = defaultdict(float)

def generate_from_LM(length, model):
    '''
    generate_from_LM takes the length of a sequence and the language model and 
    returns a sequence generated from given language model.
    '''
    with open(model) as f:
        for line in f:
            l_line =  re.split('\t|\n',line)
            model_dict[l_line[0]] = float(l_line[1])
    outcomes = np.array(list(model_dict.keys()))
    probs = np.array(list(model_dict.values()))
    bins = np.cumsum(probs)
    l = list(outcomes[np.digitize(random_sample(int(length/3)), bins)])
    sequence = ""
    for word in l:
        sequence = sequence + word
    return sequence


def doc_perplexity(document,model,length_gram):
    '''
    doc_perplexity takes a test document, a language model and the length of the n-gram 
    and returns the perplexity of this document based on the given language model. 
    Note: If the document has mutiple lines, then it will return the average perplexity
    of all lines.
    '''
    entropy=0
    perplexity_list = []
    entropy_list = []
    with open(model) as mod:
        for line in mod:
            l_line =  (re.split('\t|\n',line))
            model_dict[l_line[0]] = float(l_line[1])
    with open(document) as doc:
        for line in doc:
            line = preprocess_line(line)
            corpus=0
            for j in range(len(line)-(length_gram-1)):
                trigram = line[j:j+length_gram]
                entropy = entropy + log(model_dict[trigram],2)
                corpus += 1
            entropy = (-1/(corpus)) * entropy
            entropy_list.append(entropy)
            perplexity_list.append(np.power(2,entropy))
    return np.mean(perplexity_list)
	
#generate a LM from a training file, and named "model_name"
generate_JM_model("my_JM_model.en")
#use this model to compute perplexity of a testfile
print(doc_perplexity("test","my_JM_model.en",3))
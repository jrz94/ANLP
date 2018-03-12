#Here are some libraries you're likely to use. You might want/need others as well.
import re
import numpy as np
import math
from collections import defaultdict
from numpy.random import random_sample
from model_generator import generate_MLE_model, generate_JM_model,preprocess_line
from decimal import Decimal
from math import log

model_dict = defaultdict(float)

#Returns a list of trigrams according to the model
def generate_from_LM(length, model):
    with open(model) as f:
        for line in f:
            l_line =  re.split('\t|\n',line)
            model_dict[l_line[0]] = float(l_line[1]) #or maybe we shall keep it in the scientific notation?

    outcomes = np.array(list(model_dict.keys()))
    probs = np.array(list(model_dict.values()))
    bins = np.cumsum(probs)
    return list(outcomes[np.digitize(random_sample(int(length/3)), bins)])

#compute the entropy, then return to a average perplexity if there are multi-lines in test file
def doc_perplexity(document,model,length_gram):
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
                #print(trigram)
                #print(model_dict[trigram])
                entropy = entropy + log(model_dict[trigram],2)
                corpus += 1
                #the entropy of the letter sequence
            entropy = (-1/(corpus)) * entropy
            entropy_list.append(entropy)
            perplexity_list.append(np.power(2,entropy))
    print(np.mean(entropy_list))
    return np.mean(perplexity_list)

#generate_JM_model()

print(doc_perplexity("test_temp","model-test.en",3))
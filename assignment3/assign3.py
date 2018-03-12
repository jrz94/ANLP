from __future__ import division
from math import log,sqrt
import operator
from nltk.stem import *
from nltk.stem.porter import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

STEMMER = PorterStemmer()

# helper function to get the count of a word (string)
def w_count(word):
  return o_counts[word2wid[word]]

def tw_stemmer(word):
  '''Stems the word using Porter stemmer, unless it is a 
  username (starts with @).  If so, returns the word unchanged.

  :type word: str
  :param word: the word to be stemmed
  :rtype: str
  :return: the stemmed word

  '''
  if word[0] == '@': #don't stem these
    return word
  else:
    return STEMMER.stem(word)



def PMI(c_xy, c_x, c_y, N):
  '''Compute the pointwise mutual information using cooccurrence counts.

  :type c_xy: int 
  :type c_x: int 
  :type c_y: int 
  :type N: int
  :param c_xy: coocurrence count of x and y
  :param c_x: occurrence count of x
  :param c_y: occurrence count of y
  :param N: total observation count
  :rtype: float
  :return: the pmi value

  '''
  return log(N*c_xy/(c_x*c_y), 2)


def add2_PMI(c_xy, c_x, c_y, N):
  '''Compute the add-2 smoothing pointwise mutual information using cooccurrence counts.

  :type c_xy: int 
  :type c_x: int 
  :type c_y: int 
  :type N: int
  :param c_xy: coocurrence count of x and y
  :param c_x: occurrence count of x
  :param c_y: occurrence count of y
  :param N: total observation count
  :rtype: float
  :return: the add-2 pmi value

  '''
  return log(((N+2)*N)*(c_xy+2)/((c_x+2)*(c_y+2)), 2)# you need to fix this

def tTest(c_xy, c_x, c_y, N):
  '''Compute the t-test using cooccurrence counts.

  :type c_xy: int 
  :type c_x: int 
  :type c_y: int 
  :type N: int
  :param c_xy: coocurrence count of x and y
  :param c_x: occurrence count of x
  :param c_y: occurrence count of y
  :param N: total observation count
  :rtype: float
  :return: the t-test value

  '''
    return (c_xy - c_x*c_y/int(N))/np.sqrt(c_x*c_y)

#Do a simple error check using value computed by hand
if(PMI(2,4,3,12) != 1): # these numbers are from our y,z example
    print("Warning: PMI is incorrectly defined")
else:
    print("PMI check passed")

def cos_sim(v0,v1):
  '''Compute the cosine similarity between two sparse vectors.

  :type v0: dict
  :type v1: dict
  :param v0: first sparse vector
  :param v1: second sparse vector
  :rtype: float
  :return: cosine between v0 and v1
  '''
  # We recommend that you store the sparse vectors as dictionaries
  # with keys giving the indices of the non-zero entries, and values
  # giving the values at those dimensions.
  mul = 0
  mul0 = 0
  mul1 = 0
  for i in v0.keys():
      if i in v1.keys():
          mul = mul + v0[i] * v1[i]
  for i in v0.keys():
      mul0 = mul0 + v0[i]*v0[i]
  for i in v1.keys():
      mul1 = mul1 + v1[i]*v1[i]
  if mul == 0:
      return 0
  else:
      return mul/np.sqrt(mul0*mul1)
    
def jac_sim(v0,v1):
  '''Compute the jaccard similarity between two sparse vectors.

  :type v0: dict
  :type v1: dict
  :param v0: first sparse vector
  :param v1: second sparse vector
  :rtype: float
  :return: jaccard similarity between v0 and v1
  '''
    min_sum = 0
    max_sum = 0
    for i in v0.keys():
        if i in v1.keys():
            if v0[i] < v1[i]:
                min_sum = min_sum + v0[i]
                max_sum = max_sum + v1[i]
            else:
                min_sum = min_sum + v1[i]
                max_sum = max_sum + v0[i]
    if min_sum == 0:
        return 0
    else:
        return min_sum/max_sum
    
def dice_sim(v0,v1):
  '''Compute the dice similarity between two sparse vectors.

  :type v0: dict
  :type v1: dict
  :param v0: first sparse vector
  :param v1: second sparse vector
  :rtype: float
  :return: dice similarity between v0 and v1
  '''
    min_sum = 0
    sum01 = 0
    for i in v0.keys():
        if i in v1.keys():
            if v0[i] < v1[i]:
                min_sum = min_sum + v0[i]
                sum01 = sum01 + v0[i] + v1[i]
            else:
                min_sum = min_sum + v1[i]
                sum01 = sum01 + v0[i] + v1[i]
    if min_sum == 0:
        return 0
    else:
        return 2*min_sum/sum01



def create_ppmi_vectors(wids, o_counts, co_counts, tot_count):
    '''Creates context vectors for the words in wids, using PPMI.
    These should be sparse vectors.

    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''
    vectors = {}
    for target in wids:
        target_count = o_counts[target]
        vectors[target] = {}
        for wid0 in wid2word.keys():
            wid0_count = o_counts[wid0]
            if(wid0 in co_counts[target].keys()): # Check if the words actually co-occur
                cc = co_counts[target][wid0]; # Extract the co-occurrence counts for the target and positive word pair.
                target_wid0_pmi = PMI(cc,target_count,wid0_count,tot_count)# Compute PMI and append to the list
                if(target_wid0_pmi > 0):
                    vectors[target][wid0] = target_wid0_pmi
    return vectors

def create_add2PMI_vectors(wids, o_counts, co_counts, tot_count):
    '''Creates context vectors for the words in wids, using add-2 PMI.
    These should be sparse vectors.

    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''
    vectors = {}
    for target in wids:
        target_count = o_counts[target]
        vectors[target] = {}
        for wid0 in wid2word.keys():
            wid0_count = o_counts[wid0]
            if(wid0 in co_counts[target].keys()): # Check if the words actually co-occur
                cc = co_counts[target][wid0] # Extract the co-occurrence counts for the target and positive word pair.
                target_wid0_add2 = add2_PMI(cc,target_count,wid0_count,tot_count); # Compute PMI and append to the list
                if(target_wid0_add2 != 0):
                    vectors[target][wid0] = target_wid0_add2
    return vectors

def create_PMI_vectors(wids, o_counts, co_counts, tot_count):
    '''Creates context vectors for the words in wids, using PMI.
    These should be sparse vectors.

    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''
    vectors = {}
    for target in wids:
        target_count = o_counts[target]
        vectors[target] = {}
        for wid0 in wid2word.keys():
            wid0_count = o_counts[wid0]
            if(wid0 in co_counts[target].keys()): # Check if the words actually co-occur
                cc = co_counts[target][wid0] # Extract the co-occurrence counts for the target and positive word pair.
                target_wid0_add2 = PMI(cc,target_count,wid0_count,tot_count); # Compute PMI and append to the list
                if(target_wid0_add2 != 0):
                    vectors[target][wid0] = target_wid0_add2
    return vectors

def create_tTest_vectors(wids, o_counts, co_counts, tot_count):
    '''Creates context vectors for the words in wids, using t-test.
    These should be sparse vectors.

    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''
    vectors = {}
    for target in wids:
        target_count = o_counts[target]
        vectors[target] = {}
        for wid0 in wid2word.keys():
            wid0_count = o_counts[wid0]
            if(wid0 in co_counts[target].keys()): # Check if the words actually co-occur
                cc = co_counts[target][wid0] # Extract the co-occurrence counts for the target and positive word pair.
                target_wid0_ttest = tTest(cc,target_count,wid0_count,tot_count); # Compute PMI and append to the list
                if(target_wid0_ttest != 0):
                    vectors[target][wid0] = target_wid0_ttest
    return vectors

def create_counts_vectors(wids, o_counts, co_counts, tot_count):
    '''Creates context vectors for the words in wids, using co-counts.
    These should be sparse vectors.

    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''
    vectors = {}
    for target in wids:
        target_count = o_counts[target]
        vectors[target] = {}
        for wid0 in wid2word.keys():
            wid0_count = o_counts[wid0]
            if(wid0 in co_counts[target].keys()):
                cc = co_counts[target][wid0]
                vectors[target][wid0] = cc
    return vectors

def read_counts(filename, wids):
  '''Reads the counts from file. It returns counts for all words, but to
  save memory it only returns cooccurrence counts for the words
  whose ids are listed in wids.

  :type filename: string
  :type wids: list
  :param filename: where to read info from
  :param wids: a list of word ids
  :returns: occurence counts, cooccurence counts, and tot number of observations
  '''
  o_counts = {} # Occurence counts
  co_counts = {} # Cooccurence counts
  fp = open(filename)
  N = float(next(fp))
  for line in fp:
    line = line.strip().split("\t")
    wid0 = int(line[0])
    o_counts[wid0] = int(line[1])
    if(wid0 in wids):
        co_counts[wid0] = dict([int(y) for y in x.split(" ")] for x in line[2:])
  return (o_counts, co_counts, N)

def print_sorted_pairs(similarities, o_counts, first=0, last=100):
  '''Sorts the pairs of words by their similarity scores and prints
  out the sorted list from index first to last, along with the
  counts of each word in each pair.

  :type similarities: dict 
  :type o_counts: dict
  :type first: int
  :type last: int
  :param similarities: the word id pairs (keys) with similarity scores (values)
  :param o_counts: the counts of each word id
  :param first: index to start printing from
  :param last: index to stop printing
  :return: none
  '''
  if first < 0: last = len(similarities)
  for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse = True)[first:last]:
    word_pair = (wid2word[pair[0]], wid2word[pair[1]])
    print("{:.6f}\t{:30}\t{}\t{}".format(similarities[pair],str(word_pair),
                                         o_counts[pair[0]],o_counts[pair[1]]))

def freq_v_sim(sims):
  xs = []
  ys = []
  for pair in sims.items():
    ys.append(pair[1])
    c0 = o_counts[pair[0][0]]
    c1 = o_counts[pair[0][1]]
    xs.append(min(c0,c1))
  plt.clf() # clear previous plots (if any)
  plt.xscale('log') #set x axis to log scale. Must do *before* creating plot
  plt.plot(xs, ys, 'k.') # create the scatter plot
  plt.xlabel('Min Freq')
  plt.ylabel('Similarity')
  print("Freq vs Similarity Spearman correlation = {:.2f}".format(spearmanr(xs,ys)[0]))
  plt.show() #display the set of plots

def make_pairs(items):
  '''Takes a list of items and creates a list of the unique pairs
  with each pair sorted, so that if (a, b) is a pair, (b, a) is not
  also included. Self-pairs (a, a) are also not included.

  :type items: list
  :param items: the list to pair up
  :return: list of pairs

  '''
  return [(x, y) for x in items for y in items if x < y]



#test_words = ["cat", "dog", "mouse", "computer","@justinbieber"]
test_words = ['gelena', 'ciley','today','work','tommorow']
    
#test_words = ['gelena', 'ciley','#changmin', '#yunho','today','work','tommorow','yesterday','morning','milk','beer',
#             'dog','cat','monitor','computer','apple','banana','samsung','sony', 'microsoft','tooth','brush','taylorswift']
stemmed_words = [tw_stemmer(w) for w in test_words]
all_wids = set([word2wid[x] for x in stemmed_words]) #stemming might create duplicates; remove them
# you could choose to just select some pairs and add them by hand instead
# but here we automatically create all pairs 
wid_pairs = make_pairs(all_wids)


#read in the count information
(o_counts, co_counts, N) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/counts", all_wids)


#make the word vectors
#
#test_vect = create_tTest_vectors(all_wids, o_counts, co_counts, N)
#d_sims = {(wid0,wid1): dice_sim(vectors_t[wid0],vectors_t[wid1]) for (wid0,wid1) in wid_pairs}
#print_sorted_pairs(d_sims, o_counts)


#pmi
print('PMI:')
vectors_2 = create_PMI_vectors(all_wids, o_counts, co_counts, N)
c_sims = {(wid0,wid1): cos_sim(vectors_2[wid0],vectors_2[wid1]) for (wid0,wid1) in wid_pairs}
j_sims = {(wid0,wid1): jac_sim(vectors_2[wid0],vectors_2[wid1]) for (wid0,wid1) in wid_pairs}
d_sims = {(wid0,wid1): dice_sim(vectors_2[wid0],vectors_2[wid1]) for (wid0,wid1) in wid_pairs}
print("Sort by cosine similarity")
print_sorted_pairs(c_sims, o_counts)
print("Sort by jaccard similarity")
print_sorted_pairs(j_sims, o_counts)
print("Sort by dice similarity")
print_sorted_pairs(d_sims, o_counts)
#print('cosine similarity:')
#freq_v_sim(c_sims)
#print('jaccard similarity:')
#freq_v_sim(j_sims)
#print('dice similarity: ')
#freq_v_sim(d_sims)

#ppmi
print('PPMI:')
vectors = create_ppmi_vectors(all_wids, o_counts, co_counts, N)
c_sims = {(wid0,wid1): cos_sim(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
j_sims = {(wid0,wid1): jac_sim(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
d_sims = {(wid0,wid1): dice_sim(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
print("Sort by cosine similarity")
print_sorted_pairs(c_sims, o_counts)
print("Sort by jaccard similarity")
print_sorted_pairs(j_sims, o_counts)
print("Sort by dice similarity")
print_sorted_pairs(d_sims, o_counts)
#print('cosine similarity:')
#freq_v_sim(c_sims)
#print('jaccard similarity:')
#freq_v_sim(j_sims)
#print('dice similarity: ')
#freq_v_sim(d_sims)

#add2 PPMI
print('add2_PPMI:')
vectors_2 = create_add2PMI_vectors(all_wids, o_counts, co_counts, N)
c_sims = {(wid0,wid1): cos_sim(vectors_2[wid0],vectors_2[wid1]) for (wid0,wid1) in wid_pairs}
j_sims = {(wid0,wid1): jac_sim(vectors_2[wid0],vectors_2[wid1]) for (wid0,wid1) in wid_pairs}
d_sims = {(wid0,wid1): dice_sim(vectors_2[wid0],vectors_2[wid1]) for (wid0,wid1) in wid_pairs}
print("Sort by cosine similarity")
print_sorted_pairs(c_sims, o_counts)
print("Sort by jaccard similarity")
print_sorted_pairs(j_sims, o_counts)
print("Sort by dice similarity")
print_sorted_pairs(d_sims, o_counts)
#print('cosine similarity:')
#freq_v_sim(c_sims)
#print('jaccard similarity:')
#freq_v_sim(j_sims)
#print('dice similarity: ')
#freq_v_sim(d_sims)

#ttest
print('TTest:')
vectors_t = create_tTest_vectors(all_wids, o_counts, co_counts, N)
c_sims = {(wid0,wid1): cos_sim(vectors_t[wid0],vectors_t[wid1]) for (wid0,wid1) in wid_pairs}
j_sims = {(wid0,wid1): jac_sim(vectors_t[wid0],vectors_t[wid1]) for (wid0,wid1) in wid_pairs}
d_sims = {(wid0,wid1): dice_sim(vectors_t[wid0],vectors_t[wid1]) for (wid0,wid1) in wid_pairs}
print("Sort by cosine similarity")
print_sorted_pairs(c_sims, o_counts)
print("Sort by jaccard similarity")
print_sorted_pairs(j_sims, o_counts)
print("Sort by dice similarity")
print_sorted_pairs(d_sims, o_counts)
#print('cosine similarity:')
#freq_v_sim(c_sims)
#print('jaccard similarity:')
#freq_v_sim(j_sims)
#print('dice similarity: ')
#freq_v_sim(d_sims)

#co-counts
print('Co_counts:')
vectors_c = create_counts_vectors(all_wids, o_counts, co_counts, N)
c_sims = {(wid0,wid1): cos_sim(vectors_t[wid0],vectors_c[wid1]) for (wid0,wid1) in wid_pairs}
j_sims = {(wid0,wid1): jac_sim(vectors_t[wid0],vectors_c[wid1]) for (wid0,wid1) in wid_pairs}
d_sims = {(wid0,wid1): dice_sim(vectors_t[wid0],vectors_c[wid1]) for (wid0,wid1) in wid_pairs}
print("Sort by cosine similarity")
print_sorted_pairs(c_sims, o_counts)
print("Sort by jaccard similarity")
print_sorted_pairs(j_sims, o_counts)
print("Sort by dice similarity")
print_sorted_pairs(d_sims, o_counts)
#print('cosine similarity:')
#freq_v_sim(c_sims)
#print('jaccard similarity:')
#freq_v_sim(j_sims)
#print('dice similarity: ')
#freq_v_sim(d_sims)
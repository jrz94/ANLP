#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:34:33 2017

@author: s1719048
Automatically pick target words and also co-occurence words in a specific range of frequency.
"""

from __future__ import division
from math import log,sqrt
import operator
from nltk.stem import *
from nltk.stem.porter import *
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import spearmanr

def find_counts(filename,wid):
  o_counts = 0 # Occurence counts
  fp = open(filename)
  N = float(next(fp))
  for line in fp:
    line = line.strip().split("\t")
    wid0 = int(line[0])
    o_counts = int(line[1])
    if(wid0 == wid):
        print('co_words counts: ', o_counts)
  

def read_counts(filename, min_num, max_num, min_co, max_co):
  fp = open(filename)
  N = float(next(fp))
  for line in fp:
    line = line.strip().split("\t")
    wid0 = int(line[0])
    o_counts = int(line[1])
    if(o_counts < max_num and o_counts > min_num):
        print('The target words: ',wid2word[wid0])
        print('target word counts: ', o_counts)
        co_counts = dict([int(y) for y in x.split(" ")] for x in line[2:])
        for i in co_counts.keys():
            if (co_counts[i] < max_co and co_counts[i] > min_co):
                print('co_words is : ', wid2word[i])
                print('co counts is: ', co_counts[i])
                find_counts(filename,i)
        print('\n')        

target_min_counts = 100
traget_max_counts = 150
co_occur_min_counts = 50
co_occur_max_counts = 70
read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/counts",target_min_counts ,traget_max_counts,co_occur_min_counts ,co_occur_max_counts )

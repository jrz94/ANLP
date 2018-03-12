import sys,re
import nltk
from collections import defaultdict
import cfg_fix
from cfg_fix import parse_grammar, CFG
from pprint import pprint
# The printing and tracing functionality is in a separate file in order
#  to make this file easier to read
from cky_print import CKY_pprint, CKY_log, Cell__str__, Cell_str, Cell_log

class CKY:
    """An implementation of the Cocke-Kasami-Younger (bottom-up) CFG recogniser.

    Goes beyond strict CKY's insistance on Chomsky Normal Form.
    It allows arbitrary unary productions, not just NT->T
    ones, that is X -> Y with either Y -> A B or Y -> Z .
    It also allows mixed binary productions, that is NT -> NT T or -> T NT"""

    def __init__(self,grammar):
        '''Create an extended CKY processor for a particular grammar

        Grammar is an NLTK CFG
        consisting of unary and binary rules (no empty rules,
        no more than two symbols on the right-hand side

        (We use "symbol" throughout this code to refer to _either_ a string or
        an nltk.grammar.Nonterminal, that is, the two thinegs we find in
        nltk.grammar.Production)

        :type grammar: nltk.grammar.CFG, as fixed by cfg_fix
        :param grammar: A context-free grammar
        :return: none'''

        self.verbose=False
        assert(isinstance(grammar,CFG))
        self.grammar=grammar
        # split and index the grammar
        self.buildIndices(grammar.productions())

    def buildIndices(self,productions):
        '''
        Postcondition: Initialise two dictionaries from the grammar2 rules. Their keys are children, values are parents. Reasons for using dictionary of list: children may have several parents in the grammar, and CKY is a bottom-up algorithm, so we need to know the parents when given children.

        How: Starting from the first grammar rule, if the right hand has one child, then 
        append the lhs(left hand side) to the value of unary dictionary with the key rhs(right hand side). If there are two children, then append the lhs to the value of binary dictionary with the key rhs.

        :type productions: list(nltk.grammar.Production)
        :param productions: the list of grammar rules(nltk.grammar.Production)

        '''
        self.unary=defaultdict(list)
        self.binary=defaultdict(list)
        for production in productions:
            rhs=production.rhs()
            lhs=production.lhs()
            assert(len(rhs)>0 and len(rhs)<=2)
            if len(rhs)==1:
                self.unary[rhs[0]].append(lhs)
            else:
                self.binary[rhs].append(lhs)

    def recognise(self,tokens,verbose=False):
        '''
        Postcondition: Initialise a matrix from the sentence, then run the CKY algorithm over it and fill it. The matrix is a list, each item is a row(list) of that matrix. Each cell holds the related terminal or non-terminal.

        How: Starting from the upper-left cell from left to right, only append cells to the upper-right part of the matrix. Then fill in each cell using the methods "unaryFill" and "binaryScan".

        :type tokens: list(string)
        :param tokens: the token list of the sentence 
        :type verbose: bool
        :param verbose: show debugging output if True, defaults to False
        :rtype: bool or int
        :return: If the input is not recognised, return False. If recognised, return the number of successful analyses.
        '''
        self.verbose=verbose
        self.words = tokens
        self.n = len(self.words)+1
        self.matrix = []
        # We index by row, then column
        #  So Y below is 1,2 and Z is 0,3
        #    1   2   3  ...
        # 0  .   .   Z
        # 1      Y   .
        # 2          .
        # ...
        for r in range(self.n-1):
             # rows
             row=[]
             for c in range(self.n):
                 # columns
                 if c>r:
                     # This is one we care about, add a cell
                     row.append(Cell(r,c,self))
                 else:
                     # just a filler
                     row.append(None)
             self.matrix.append(row)
        self.unaryFill()
        self.binaryScan()
        # Replace the line below for Q6
        return self.grammar.start() in self.matrix[0][self.n-1].labels()

    def unaryFill(self):
        '''
        Postcondition: Fill the diagonal of the matrix, containing terminal or non-terminal, the non-terminal is the parent of that terminal directly or indirectly.
        How: Starting from the first word(token) in the sentence, filling the cells from the upper-left diagonal. Firstly, it adds the terminal, then call the "unaryUpdate" to find all direct or indirect parents and fill in the cell. 

        '''
        for r in range(self.n-1):
            cell=self.matrix[r][r+1]
            word=self.words[r]
            cell.addLabel(word)
            cell.unaryUpdate(word)

    def binaryScan(self):
        '''(The heart of the implementation.)

Postcondition: the matrix has been filled with all constituents that
can be built from the input words and grammar.

How: Starting with constituents of length 2 (because length 1 has been
done already), proceed across the upper-right diagonals from left to
right and in increasing order of constituent length. Call maybeBuild
for each possible choice of (start, mid, end) positions to try to
build something at those positions.

        '''
        for span in range(2, self.n):
            for start in range(self.n-span):
                end = start + span
                for mid in range(start+1, end):
                    self.maybeBuild(start, mid, end)

    def maybeBuild(self, start, mid, end):
        '''
        Postcondition: When given two positions: (start, mid) and (mid, end), try to fill in the position(start, end).

        How: Starting from the first label of cell[start][mid] and cell[mid][end], try to find their children, if exists, add this child as a label of the new cell[start][end], then call "unaryUpdate" to find all direct and indirect labels.

        :type start: int
        :param start: the row number of one of two parents and the child.
        :type mid: int
        :param mid: the column number of one parent, also the row number of another parent in the same binary rule.
        :type end: int
        :param end: the colume number of one of two parents and the child.
        '''
        self.log("%s--%s--%s:",start, mid, end)
        cell=self.matrix[start][end]
        for s1 in self.matrix[start][mid].labels():
            for s2 in self.matrix[mid][end].labels():
                if (s1,s2) in self.binary:
                    for s in self.binary[(s1,s2)]:
                        self.log("%s -> %s %s", s, s1, s2, indent=1)
                        cell.addLabel(s)
                        cell.unaryUpdate(s,1)

# helper methods from cky_print
CKY.pprint=CKY_pprint
CKY.log=CKY_log

class Cell:
    '''A cell in a CKY matrix'''
    def __init__(self,row,column,matrix):
        self._row=row
        self._column=column
        self.matrix=matrix
        self._labels=[]

    def addLabel(self,label):
        self._labels.append(label)

    def labels(self):
        return self._labels

    def unaryUpdate(self,symbol,depth=0,recursive=False):
        '''
        Postcondition: Update a cell in the CKY matrix, find the parents in a recursive way, which means given a child, find its parent and add it as a label. Then treat this parent as a child and find its parent. 

        How: If the symbol(child) is the key in unary dictionary, then treat its value(parent) as a new key and find the corresponding value(parent), add this value to the label. Then call unaryUpdate recursively to find all related parents.
        
        :type symbol: a string (for terminals) or an nltk.grammar.Nonterminal
        :param symbol: a terminal or non-terminal
        '''
        if not recursive:
            self.log(str(symbol),indent=depth)
        if symbol in self.matrix.unary:
            for parent in self.matrix.unary[symbol]:
                self.matrix.log("%s -> %s",parent,symbol,indent=depth+1)
                self.addLabel(parent)
                self.unaryUpdate(parent,depth+1,True)

# helper methods from cky_print
Cell.__str__=Cell__str__
Cell.str=Cell_str
Cell.log=Cell_log

class Label:
    '''A label for a substring in a CKY chart Cell

    Includes a terminal or non-terminal symbol, possibly other
    information.  Add more to this docstring when you start using this
    class'''
    def __init__(self,symbol,
                 # Fill in here, if more needed
                 ):
        '''Create a label from a symbol and ...
        :type symbol: a string (for terminals) or an nltk.grammar.Nonterminal
        :param symbol: a terminal or non-terminal
        '''
        self._symbol=symbol
        # augment as appropriate, with comments

    def __str__(self):
        return str(self._symbol)

    def __eq__(self,other):
        '''How to test for equality -- other must be a label,
        and symbols have to be equal'''
        assert isinstance(other,Label)
        return self._symbol==other._symbol

    def symbol(self):
        return self._symbol
    # Add more methods as required, with docstring and comments

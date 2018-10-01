
import nltk;
import re;
import numpy as np
from nltk.corpus import stopwords;
stop = stopwords.words('english');
import os
from collections import Iterable


NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']

class Utils:
    

    DATA_PATH = 'data/lexpredict-contraxsuite-samples/agreements/';
    DATA_DIRS = []
    COLUMNS = ['data','name'];
    BLACK_LISTED_FILES = ['001.txt', '002.txt','003.txt']
    
    def format_title(document):
        punctuation = '[^A-Za-z0-9]+'
        document = Utils.clean_document(document, True, True, punctuation)     
        docs = list(map(lambda t : str(t).lower(), document.split(" ")))
        return Utils.join_list(docs, "_")
    
    @staticmethod
    def clean_document(document, merge_acronym = False, rm_stop_words = False, punctuation = '[^A-Za-z .-]+'):
        """Cleans document by removing unnecessary punctuation. It also removes
        any extra periods and merges acronyms to prevent the tokenizer from
        splitting a false sentence
        """
        # Remove all characters outside of Alpha Numeric
        # and some punctuation
        document = re.sub(punctuation, ' ', document)
        document = document.replace('-', '')
        document = document.replace('...', '')
        document = document.replace('Mr.', 'Mr').replace('Mrs.', 'Mrs')

        # Remove Ancronymns M.I.T. -> MIT
        # to help with sentence tokenizing
        if(merge_acronym): document = Utils.merge_acronyms(document)

        # Remove extra whitespace
        document = ' '.join(document.split())
        
        if(rm_stop_words): document = Utils.remove_stop_words(document)
            
        return document
    
    @staticmethod
    def split_document(document, sep = ' '):
        return document.split(sep)
        
    @staticmethod
    def remove_stop_words(document):
        """Returns document without stop words"""
        document = ' '.join([i for i in document.split() if i not in stop])
        return document

    @staticmethod
    def similarity_score(t, s):
        """Returns a similarity score for a given sentence.
        similarity score = the total number of tokens in a sentence that exits
                            within the title / total words in title
        """
        t = Utils.remove_stop_words(t.lower())
        s = Utils.remove_stop_words(s.lower())
        t_tokens, s_tokens = t.split(), s.split()
        similar = [w for w in s_tokens if w in t_tokens]
        score = (len(similar) * 0.1 ) / len(t_tokens)
        return score

    @staticmethod
    def merge_acronyms(s):
        """Merges all acronyms in a given sentence. For example M.I.T -> MIT"""
        r = re.compile(r'(?:(?<=\.|\s)[A-Z]\.)+')
        acronyms = r.findall(s)
        for a in acronyms:
            s = s.replace(a, a.replace('.',''))
        return s

    @staticmethod
    def rank_sentences(doc, doc_matrix, feature_names, top_n=3, title= None):
        """Returns top_n sentences. Theses sentences are then used as summary
        of document.
        input
        ------------
        doc : a document as type str
        doc_matrix : a dense tf-idf matrix calculated with Scikits TfidfTransformer
        feature_names : a list of all features, the index is used to look up
                        tf-idf scores in the doc_matrix
        top_n : number of sentences to return
        """
        sents = nltk.sent_tokenize(doc)
        #print (sents)
        sentences = [nltk.word_tokenize(sent) for sent in sents]
        sentences = [[w for w in sent if nltk.pos_tag([w])[0][1] in NOUNS]
                      for sent in sentences]
        #print(sentences)
        tfidf_sent = [[doc_matrix[feature_names.index(w.lower())]
                       for w in sent if w.lower() in feature_names]
                     for sent in sentences]

        # Calculate Sentence Values
        doc_val = sum(doc_matrix)
        sent_values = [sum(sent) / doc_val for sent in tfidf_sent]

        # Apply Similariy Score Weightings
        similarity_scores = [Utils.similarity_score(title, sent) for sent in sents]
        scored_sents = np.array(sent_values) + np.array(similarity_scores)

        # Apply Position Weights
        ranked_sents = [sent*(i/len(sent_values))
                        for i, sent in enumerate(sent_values)]

        ranked_sents = [pair for pair in zip(range(len(sent_values)), sent_values)]
        ranked_sents = sorted(ranked_sents, key=lambda x: x[1] *-1)

        return ranked_sents[:top_n]
    
    @staticmethod
    def clean_data(dataArray):            
        return [Utils.remove_stop_words(Utils.clean_document(document)) for document in dataArray]
    
    @staticmethod
    def read_file_data(filepath):
        text = ''
        with open(filepath) as f:   
            content = f.read();
            text = content;
        return text;

    @staticmethod
    def flatten(items):
        """Yield items from any nested iterable; see REF."""
        for x in items:
            #print (x)
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                yield from Utils.flatten(x)
            elif(x == None):
                pass
            else:
                yield x

    @staticmethod
    def join_list(items, sep = " "):
        if(sep == None): sep = " "
        return sep.join(items)
    @staticmethod
    def int_to_roman(input, lower = False):  
        if type(input) != type(1):
            raise TypeError ("expected integer, got %s" % type(input))
        if not 0 < input < 4000:
            raise ValueError("Argument must be between 1 and 3999"   )
        ints = (1000, 900,  500, 400, 100,  90, 50,  40, 10,  9,   5,  4,   1)
        nums = ('M',  'CM', 'D', 'CD','C', 'XC','L','XL','X','IX','V','IV','I')
        nums = tuple(map(lambda x : x.lower(), nums)) if(lower) else nums
        result = ""
        for i in range(len(ints)):
            count = int(input / ints[i])
            result += nums[i] * count
            input -= ints[i] * count
        return result
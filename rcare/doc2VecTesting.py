import sys
import logging;
import sqlite3;
import pandas as pd
import os
sys.path.append('/home/paperspace/dev/en-nlp/')
sys.path.append('/home/paperspace/dev/en-nlp/rcare/nlp')
sys.path.append('/home/paperspace/dev/en-nlp/rcare')
import warnings


#print({'path':sys.path})
errors = []
try:
    # Trying to find module in the parent package
    from rcare.readutils import DataFrameUtils as Utils   
    
    from rcare.nlp.customize_nlp import rcare
    import spacy;    
    from rcare.singleton import SingletonDecorator        
    nlp = rcare(spacy.load('en_core_web_sm')).nlp 
    
    from rcare.nlp.doc2vec_rcare import Doc2VecWrapper, SentenceDocument, QueryDocument
    from rcare.es_helper import ESHelper
    ES_CONFIG = {
        "host": [
            {
                "host": "localhost",                  
                "port":"9200"
            }
        ],
        "log": "error"
    };
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    
    def doc2VecTesting(args):    
        logging.info("Building Training Data .. Start")  
        #esHelper = ESHelper(ES_CONFIG)
        #print(esHelper.info())   
        text = args[1];
        name = args[2];
        sectionId = args[3];
        index = args[4];        
        query = QueryDocument(name, sectionId, text)
        sec_D2Vec = Doc2VecWrapper(nlp = nlp, esconfig = ES_CONFIG, configName = "section")
        sen_D2Vec = Doc2VecWrapper(nlp = nlp, esconfig = ES_CONFIG, configName = "sentence")
        reload = True
        sen_Sims, sentDocs = sen_D2Vec.similarity(query, reload = reload)
        sec_Sims, secDocs = sec_D2Vec.similarity(query, reload = reload)
        
        sentDf = pd.DataFrame( sentDocs,columns = sentDocs[0]._fields)
        sentDf.drop(['sectionSimilarity'], axis=1,inplace  = True)
        secDf = pd.DataFrame( secDocs,columns = secDocs[0]._fields)
        sentDf = sentDf.merge(secDf[["name", "sectionId", "sectionSimilarity"]], how = "left", on =["name", "sectionId"])
        #print(text);
        
        #pint(sims);

        sen_D2Vec.save_to_es(index = index, doc_type ="agreement", df = sentDf)

        print(index, query)
        
        print(True)
        logging.info("Building Training data complete") 

        return               
            
    #print(sys.argv)
    doc2VecTesting(sys.argv)
    
except ImportError as err:
    print('Relative import failed {}'.format(err))
except Exception as err:   
    print('Exception {}'.format(err))    


    




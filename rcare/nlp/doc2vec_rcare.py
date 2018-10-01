
import os
import logging
import pandas as pd
import multiprocessing
import re

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_non_alphanum , strip_multiple_whitespaces

from collections import namedtuple

from rcare.readutils import DataFrameUtils as Utils
from rcare.nlp.sections_rcare import get_sections
from rcare.es_helper import ESHelper, ESAlias

MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

QueryDocument  = namedtuple('QueryDocument',"name sectionId text")
SentenceDocument = namedtuple('SentenceDocument', 'words tags names index name sectionId sentId sentText startChar \
                                                    endChar sectionText sentSimilarity query target predict sectionSimilarity \
                                                    rank docCount')
DocCategorize  = namedtuple('DocCategorize',"best moderate worst")
ModelConfig = namedtuple('ModelConfig',"model index docType name")
SimDocument =  namedtuple('SimDocument',"sim, index, document")

Document = namedtuple('Document', 'id name filePath text sectionCount sections sentCount sents ents rank')
Section = namedtuple('Section', 'id sectionId text sentCount index rank')
Sentence = namedtuple('Sentence', 'id sectionId sentId text startChar endChar sectionText rank')
Entity = namedtuple('Entity', 'sectionId text startChar endChar label')

defaultConfig = "sentence"
defaultQuery = body = {
              "sort": [
               {
                  "name.keyword": {
                    "order": "asc"
                  }
                }
              ],
              "from": 0,
              "size": 10000,
              "query": {
                "match_all": {}
              }
          }
class Doc2VecWrapper(object):
    
    MODELS = ["section", "sentence"]
    MODEL_CONFIGS = {
        "section": ModelConfig("./doc2vec.sec.pickle",  "credit.sec", "agreement", "section"),
        "sentence": ModelConfig("./doc2vec.pickle",  "credit", "agreement", "sentence"),
    }
        
    
    @staticmethod
    def config(name = defaultConfig)-> ModelConfig : 
        return Doc2VecWrapper.MODEL_CONFIGS[name];
    
        
    
    def __init__(self, nlp, esconfig = None, configName = defaultConfig):  
        self.config = None
        self.checkConfig(configName)
        self.SIM_MODEL = None;
        self.taggedDocs = []
        self.nlp = nlp
        self.esconfig = esconfig
        #logging.warn("Loading using config {}".format(self.config))    
                       
    def load(self):
        logging.warn("Loading using config {}".format(self.config))    
        self.load_model()
        self.load_training_data()
        
        if(self.SIM_MODEL.docvecs.count != len(self.taggedDocs)):
            raise Exception ("Model doc count {} different from training doc count {}.".\
                             format(self.SIM_MODEL.docvecs.count, len(self.taggedDocs)))
        
    def load_model(self):
        self.SIM_MODEL = Doc2Vec.load(os.path.join(MODULE_PATH, self.config.model))
        #self.SIM_MODEL_SEC = Doc2Vec.load(os.path.join(MODULE_PATH, "./doc2vec.sec.pickle"))
        
    def meta_data(self, metaIndex = "demo.meta", body = defaultQuery):
        
        esHelper = ESHelper(self.esconfig)
        gen = esHelper.search_nt (metaIndex, body = body)
        return gen
        
    def load_training_data(self)    :
        body = {
              "sort": [
                {
                  "index": {
                    "order": "asc"
                  }
                },
                {
                  "sectionId": {
                    "order": "asc"
                  }
                },
                {
                  "sentId": {
                    "order": "asc"
                  }
                }
              ],
              "from": 0,
              "size": 10000,
              "query": {
                "match_all": {}
              }
            }
        esHelper = ESHelper(self.esconfig)
        self.taggedDocs = list(esHelper.search_nt (self.config.index, body = body, name = "SentenceDocument"))
    
    def checkConfig(self, configName):
        if(configName == None and self.config == None):
            self.config = Doc2VecWrapper.config(defaultConfig);            
        else:
            if(configName != None and self.config == None): 
                self.config = Doc2VecWrapper.config(configName);                
            else: # self.config is loaded.. use the default do nothing                  
                pass
            
        logging.warn("Config being used {}".format(self.config))    
            
    def prep_training_data(self, config = None):
        self.checkConfig(config)
        if(self.config.name == defaultConfig):
            return self.prep_training_data_sent()
        else:
            return self.prep_training_data_section()
        
    def tokenize(self, text):
        text =  remove_stopwords(text);
        text = strip_punctuation(text)
        text = strip_non_alphanum(text)
        text = text.lower();
        text = re.sub(r"([\.\",\(\)!\?;:])", " \\1 ", text)
        text = re.sub(r'[0-9]+', '', text) 
        text = strip_multiple_whitespaces(text)
        tokens = text.split(" ")
        
        tokens = list(filter(lambda token: (str(token).strip() not in [""]), tokens ))
        return tokens;
        
    def prep_data_for_section(self, section, contract):
        taggedDocs = []      
        
        index = 0
        #contract.sents.append(new section)        
        tokens = self.tokenize(str(section.text)) 
        name = section.name
        if(len(tokens) > 0):
            #label = "{}_{}".format("TRAIN_SEC", index)
            label = index
            tags = [label]
            names = [name]        
            taggedDocs.append(
                SentenceDocument(tokens, tags, names, label, name, -1 ,0,
                                               section.text, 0, len(section.text), section.text,0.0,  
                                               section, 0, 0 , 0, 0, 0 )
                                )         
            index = index + 1   
            
        for sent in contract.sents:
            sent_np = Sentence(*[sent[key] for key in list(Sentence._fields)])
            tokens = []
            name = contract.name
            text = sent_np.text                
            tokens = self.tokenize(str(text)) 
            if (index % 1000) == 0:
                logging.warn("Progress doc count: {}".format(index))
            if(len(tokens) > 0):
                #label = "{}_{}".format("TRAIN", index)
                label = index
                tags = [label]
                names = [name]        
                taggedDocs.append(
                    SentenceDocument(tokens, tags, names, label, name, sent_np.sectionId ,sent_np.sentId,
                                                   text, sent_np.startChar, sent_np.endChar, sent_np.sectionText,0.0,  
                                                   section, 0, 0 , 0, sent_np.rank, 0  )
                                    )         
                index = index + 1
        
        #tokens = self.__tokens__(section.text, spacy_obj = False) 
        
            
        return taggedDocs
        
    
    def start(self, contractFile, masterSectionId = None): # given a master file name.
        #pass;
        body = {
            "sort": [
               {
                  "name.keyword": {
                    "order": "asc"
                  }
                }
              ],
              "from": 0,
              "size": 10000,
              "query": {
                "bool": {
                    "must":{
                        "term" : {"name.keyword": contractFile}
                    }
                }
              }
        }
        master_query = {
            "sort": [
               {
                  "name.keyword": {
                    "order": "asc"
                  }
                }
              ],
              "from": 0,
              "size": 10000,
              "query": {
                "bool": {
                    "must":{
                        "term" : {"name.keyword": "demo_master_visa_credit_card_agreement.txt"}
                    }
                }
              }
        }
        masterDocs= []
        for doc in self.meta_data(body = master_query):
            masterDocs.append(doc)
        es_index_alias = [];

        
        if(len(masterDocs)>0):
            masterDoc = masterDocs[0]
            startSectionId = masterSectionId if(masterSectionId) else 0;
            endSectionId = masterSectionId + 3 if(masterSectionId) else masterDoc.sectionCount;
            
            for contract in self.meta_data(body = body):
                for sec in masterDoc.sections[startSectionId: endSectionId]:
                    mSection = Section(*[sec[key] for key in list(Section._fields)])
                    #section                
                    secQuery = QueryDocument(masterDoc.name, mSection.sectionId, mSection.text)
                    es_index = self.train_model(mSection, contract , secQuery)
                    es_alias = ESAlias(es_index, "{}_{}".format(masterDoc.name, mSection.sectionId), "add")
                    es_index_alias.append(es_alias)
                    logging.warn("es_alias# {}".format(es_alias));
                    
            self.update_alias(es_index_alias)                        
        else:
            logging.error("masterdoc empty..");
            
        return self.taggedDocs, self.SIM_MODEL        
                
                
    def update_alias(self, alias):
        esHelper = ESHelper(self.esconfig)
        esHelper.alias(alias)
        
    def train_model(self, section, contract, secQuery):
       
        self.taggedDocs = self.prep_data_for_section(secQuery, contract)
        
        #model = self.train(modelId = 1, vector_size = 100, epochs = 300, window = 8)#model parameters
        self.train_john()
        simDoc, tDocs, sims = self.similarity_john(secQuery)
        
        #save data temp
        es_index = "{}_{}_{}".format(contract.name, secQuery.sectionId, 1)        
        self.save_training_data(index = es_index, doc_type = "agreement") #index, doc_type,
        
        startIndex, endIndex = self.find_target_sent_block(self.taggedDocs, sims)
        
        #startIndex, endIndex = self.find_target_sent_block_naveen(self.taggedDocs, sims)
        logging.warn("targetIndex# {} - {}".format(startIndex, endIndex))
        
        self.editTaggedDocs(startIndex, endIndex, contract)        
        
        logging.warn("taggedDocs length# {}".format(len(self.taggedDocs)))        
                             
        self.train_john()
        simDoc, tDocs, sims = self.similarity_john(secQuery)
        
        #logging.warn("sims (run 2) # {}".format(sims))      
        
        es_index = "{}_{}_{}".format(contract.name, secQuery.sectionId, 2)                
        self.save_training_data(index = es_index, doc_type = "agreement") #index, doc_type,
               
        return es_index
        # call prep data and train the model.
        #train model set params
        #find similarity # save to db.
        #find target sent block.
        #edit taggedDocs -- add the new block and remove the other sentences..
        #rebuild model
            #train model set params
            #find similarity 
                
    def editTaggedDocs(self, startIndex, endIndex, contract) :
        tokens = []
        newTaggedDoc = None;
        sectionId = None
        sectionText = [];
        sentText = [];
        taggedDocs = []
        taggedDocs.append(self.taggedDocs[0])
        
        for doc in self.taggedDocs[startIndex: endIndex]:
            tokens = tokens + doc.words
            newTaggedDoc = doc;
            sentText.append( doc.sentText);
            if(not doc.sectionId == sectionId):
                sectionId = doc.sectionId
                sectionText.append(doc.sectionText)
            
        
        newTaggedDoc = newTaggedDoc._replace(words = tokens) 
        newTaggedDoc = newTaggedDoc._replace(sectionText = "||".join(sectionText)) 
        newTaggedDoc = newTaggedDoc._replace(sentText = " ".join(sentText)) 
        newTaggedDoc = newTaggedDoc._replace(startChar = startIndex) 
        newTaggedDoc = newTaggedDoc._replace(endChar = endIndex) 
        #newTaggedDoc = newTaggedDoc._replace(endChar = 0) 
        newTaggedDoc = newTaggedDoc._replace(sectionId = -2) 
        newTaggedDoc = newTaggedDoc._replace(sentId = 0) 
        #label = "{}_{}".format("TEST_NEW", "0")
        label = 1
        
        #index = len(self.taggedDocs)
        newTaggedDoc = newTaggedDoc._replace(tags = [label]) 
        newTaggedDoc = newTaggedDoc._replace(index = label) 
        
        logging.warn("DOC: {}".format(newTaggedDoc))
        taggedDocs .append(newTaggedDoc);
        #self.taggedDocs.append(newTaggedDoc)
        
        
        index = 2
        for doc in self.taggedDocs[1:]:
            if(not (doc.index >= startIndex and doc.index <= endIndex)):
                doc = doc._replace(index = 0) 
                #label = "{}_{}".format("TEST", index)
                label = index
                doc = doc._replace(tags = [label]) 
                doc = doc._replace(index = label) 
                taggedDocs.append(doc)
                #logging.warn("INDEX: {} - DOC: {}".format(index, doc))
                index = index + 1;
        logging.warn("OLD: {} vs NEW: {}".format(len(self.taggedDocs), len(taggedDocs)))
        self.taggedDocs = taggedDocs;
        
        
    def find_target_sent_block_naveen(self, taggedDocs, sims):
        recCount = 15
        df = pd.DataFrame( taggedDocs,columns = taggedDocs[0]._fields).sort_values(["sentSimilarity"], ascending = [0])
        dfg = df[1:].head(recCount).groupby(["sectionId"])
        sf = dfg['sectionId'].count().sort_values(ascending = False)
        df_sent = pd.DataFrame({'sectionId':sf.index, 'sentCount':sf.values})
        sf = dfg["sentSimilarity"].mean().sort_values(ascending = False)
        df_sim = pd.DataFrame({'sectionId':sf.index, 'avg':sf.values})
        df_new = pd.merge(df_sent, df_sim, on="sectionId")
        df_new = df_new.query("sentCount == {}".format( df_new["sentCount"].max())).sort_values(["avg"], ascending = False)
        res = (df[1:].head(recCount).query("sectionId == {}".format(df_new["sectionId"].values[0]))[["sentSimilarity","index"]]).describe()
        return int(res.loc["min"]["index"]), int(res.loc["max"]["index"])

    def find_target_sent_block(self, taggedDocs, sims):# return startIndex & endIndex
        results = sims[:20];
        logging.warn("find_target_sent_block {}".format(results))    
        size = len(taggedDocs) - 1
        resultsArray = [0] * size          
        for i in range(0,size):
            tmpItem = [item for item in results if item[0] == i]
            if len(tmpItem) == 1:
                resultsArray[i] = 1
        logging.warn("{}".format(resultsArray))    
        longestSub = 0
        tmpSub = 0
        startIndex = 0
        endIndex = 0
        tmpStart = 0
        tmpEnd = 0
        skipLim = 1
        skip = 0
        skipTrig = False
        newSub = False
        for i in range(0,len(resultsArray)-1):
            if (resultsArray[i]==1):
                if (newSub==False):
                    newSub = True
                    tmpStart = i
                if (newSub==True):
                    tmpSub = tmpSub + 1
            if (resultsArray[i]==0):
                if (newSub==False):
                    continue
                if (skipTrig==False):
                    tmpSub = tmpSub + 1
                    skipTrig = True
                elif (skipTrig == True):
                    skipTrig = False
                    if (tmpSub > longestSub):
                        startIndex = tmpStart
                        endIndex = i
                        longestSub = tmpSub
                        tmpSub = 0
                    tmpSub = 0
                    newSub = False
            
        
        return (startIndex, endIndex);
    
    def prep_training_data_section(self ):      
        self.taggedDocs , tokens = [], []   
        index = 0
        logging.warn("prep_training_data_section start ")    
        for doc in self.meta_data():              
            logging.warn("Processing - {} => {} ".format(doc.name,  doc.sectionCount))   
            name = doc.name
            for sec in doc.sections:
                sec_np = Section(*[sec[key] for key in list(Section._fields)])
                tokens = []                                       
                text = sec_np.text                
                tokens = self.__tokens__(text, spacy_obj = False) 
                if(len(tokens) > 0):
                    #label = "{}_{}".format("TRAIN_SEC", index)
                    label = index
                    tags = [label]
                    names = [name]        
                    self.taggedDocs.append(SentenceDocument(tokens, tags, names, label, name, sec_np.sectionId ,0,
                                                       text, 0, len(text), text,0.0,  
                                                       QueryDocument("", "", ""), 0, 0 , 0  , sec_np.rank   , 0 
                                                      ))         
                    index = index + 1

        logging.warn("prep_training_data_section done ")           
        return self.taggedDocs  
    
    def prep_training_data_sent(self ):      
        self.taggedDocs , tokens = [], []   
        index = 0
        logging.warn("prep_training_data_sent start ")    
        for doc in self.meta_data():          
            logging.warn("Processing - {} => {} ".format(doc.name,  doc.sentCount))   
            name = doc.name
            for sent in doc.sents:
                sent_np = Sentence(*[sent[key] for key in list(Sentence._fields)])
                tokens = []                                       
                text = sent_np.text                
                tokens = self.__tokens__(text, spacy_obj = False) 
                if (index % 1000) == 0:
                    logging.warn("Progress doc count: {}".format(index))
                if(len(tokens) > 0):
                    #label = "{}_{}".format("TRAIN_SENT", index)
                    label = index
                    tags = [label]
                    names = [name]        
                    self.taggedDocs.append(
                        SentenceDocument(tokens, tags, names, label, name, sent_np.sectionId ,sent_np.sentId,
                                                       text, sent_np.startChar, sent_np.endChar, sent_np.sectionText,0.0,  
                                                       QueryDocument("", "", ""), 0, 0 , 0, sent_np.rank , 0 )
                                        )         
                    index = index + 1

        logging.warn("prep_training_data_sent done ")           
        return self.taggedDocs    
        
    def save_training_data(self, index, doc_type, esconfig = None , taggedDocs = None):
        def get_id(item):
            return "{}_{}_{}".format(item['name'] ,item['sectionId'], item['sentId'])
        
        self.esconfig = esconfig if(esconfig != None) else self.esconfig
        self.taggedDocs = taggedDocs if(taggedDocs != None) else self.taggedDocs
        if(self.taggedDocs == None and len(self.taggedDocs)<=0 ):
            raise Exception ("Nothing to save.");
            
        self.save_to_es(index, doc_type, pd.DataFrame( self.taggedDocs,columns = self.taggedDocs[0]._fields))        
    
    def save_to_es(self, index, doc_type, df):
        def get_id(item):
            return "{}_{}_{}".format(item['name'] ,item['sectionId'], item['sentId'])
        
        esHelper = ESHelper(self.esconfig)
        esHelper.bulk_stream_collection( df.to_dict(orient='records'), 
                                index = index, doc_type = doc_type,  get_id = get_id)
        
        logging.warn("Data saved {}/{} @ {}".format(index,doc_type, self.esconfig))        
        
    def save_data(self, configName = None):
        self.checkConfig(configName)
        self.save_training_data(self.config.index, self.config.docType, self.esconfig, self.taggedDocs)        
        
    def train_john(self):
        
        if(self.taggedDocs == None or len(self.taggedDocs) <=0):
            raise Exception ("Prep training data - prep_training_data");
            
        logging.warn('Model Training Start.')
        cores = multiprocessing.cpu_count()
        epochs = 300
        max_epochs = 300   
        vec_size = 75
        alpha = 0.025 
        self.SIM_MODEL = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=0,
                    dbow_words=1, window=5, workers=cores) 
        self.SIM_MODEL.build_vocab(self.taggedDocs)
        for epoch in range(max_epochs):
            self.SIM_MODEL.train(self.taggedDocs,
                        total_examples=self.SIM_MODEL.corpus_count,
                        epochs=self.SIM_MODEL.iter)
            # decrease the learning rate
            self.SIM_MODEL.alpha -= 0.0002
            # fix the learning rate, no decay
            self.SIM_MODEL.min_alpha = self.SIM_MODEL.alpha 

        
        logging.warn('Model Training Complete.')
 
        
    def train(self, modelId = 0, vector_size = 300, epochs = 100, window = 15):        
        if(self.taggedDocs == None or len(self.taggedDocs) <=0):
            raise Exception ("Prep training data - prep_training_data");
            
        cores = multiprocessing.cpu_count()
        #assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"            
        
     
        min_count = 2
        sampling_threshold = 1e-5
        negative_size = 5
        
     
        
        simple_models = [
            # PV-DBOW plain
            Doc2Vec(dm=0, vector_size=vector_size, dbow_words=1, window=window, negative=negative_size, hs=0, min_count=min_count,\
                    sample=sampling_threshold, epochs=epochs, workers=cores),
            # PV-DM w/ default averaging; a higher starting alpha may improve CBOW/PV-DM modes
            # Doc2Vec(dm=1, vector_size=vector_size, window=window, negative=5, hs=0, min_count=2, sample=0, 
            # epochs=epochs, workers=cores, alpha=0.05, comment='alpha=0.05'),
            Doc2Vec(dm=1, dm_mean=1, vector_size=vector_size, window=window, negative=negative_size, hs=0, min_count=min_count, 
                    epochs=epochs, workers=cores, alpha=0.0005),
            # PV-DM w/ concatenation - big, slow, experimental mode
            # window=5 (both sides) approximates paper's apparent 10-word total window size
            Doc2Vec(dm=1, dm_concat=1, vector_size=vector_size, window=window, negative=5, hs=0, min_count=min_count, \
                    sample=sampling_threshold, epochs=epochs, workers=cores),
        ]
        
        model = simple_models[modelId]
        model.build_vocab(self.taggedDocs)
        #model.train(alldocs, total_examples=model.corpus_count, epochs=model.epochs)
        #(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1, pretrained_emb=pretrained_emb, iter=train_epoch)
                

        
        model.train(self.taggedDocs, total_examples=model.corpus_count, epochs=model.epochs)
            
        self.SIM_MODEL = model;
        logging.warn('Model Training Complete.')
        return self.SIM_MODEL
        
    def save_model(self):
        doc2vec_pkl_filename = os.path.join(MODULE_PATH, self.config.model)
        self.SIM_MODEL.save(doc2vec_pkl_filename)
        
    def save(self):
        self.save_model();
        self.save_data()
        self.load_model()
    
    def __tokens__(self, text, spacy_obj = False):
        text = text if (spacy_obj) else self.nlp(text) 
        tokens = []
        for w in text: 
            if(w.is_stop):
                continue
            if(not w.is_ascii):
                continue
            elif (w.is_digit):
                #tokens.append("-DIGIT-")  
                continue;
            elif (w.like_num):
                #tokens.append("-DIGIT-")  
                continue;                    
            elif (w._.is_cardinal):
                #tokens.append("-CARDINAL-") 
                continue;
            elif (w.is_punct):
                #tokens.append(w.text)     
                continue;
            elif (w.is_stop):
                #tokens.append("-STOP-")     
                tokens.append(w.lower_)
                continue;
            elif (w.like_url):
                #tokens.append("-URL-")                  
                continue;                    
            elif (w.ent_type_ == 'ORG') or (w.ent_type_ == 'GPE'):
                tokens.append("-{}-".format(w.ent_type_))
                continue;
            elif(w.lemma_ == '-PRON-'):
                 tokens.append(w.lower_)
            elif (w.text.strip() not in ('\n', '$','\n\n' , '\n\n\n', '\n\n\n\n','' ) ) :                 
                tokens.append(w.lemma_)  
                
        return tokens;
    
    def similarity_john(self, query):
        model = self.SIM_MODEL;
        sims = model.docvecs.most_similar([model.docvecs[0]], topn=len(self.taggedDocs))
        logging.warn("sims:{}".format(len(sims)))
        simDocs = self.__updateTaggedDocs__(sims, query)
        return simDocs, self.taggedDocs, sims;
        
    def similarity(self, query, reload = True, threshold = 0.5, configName = None):
        self.checkConfig(configName)
        if(reload == True):            
            self.load()
            
        model = self.SIM_MODEL;
        logging.warn("Model:{} - {}".format(model, len(self.taggedDocs)))
        #tokens = self.__tokens__(query.text, spacy_obj = False)
        tokens = self.tokenize(query.text)
        inferred_vector = model.infer_vector(tokens)    
        logging.warn("query:{}".format(tokens))
        #logging.warn("inferred_vector:{}".format(inferred_vector))
        sims = model.docvecs.most_similar([inferred_vector], topn=len(self.taggedDocs))  # get *all* similar documents model.docvecs.count,
        logging.warn("sims:{}".format(len(sims)))
        simDocs = self.__updateTaggedDocs__(sims, query)
        return simDocs, self.taggedDocs, sims;
    
    def getTaggedDocs(self):
        return self.taggedDocs;
    
    def getModel(self):
        return self.model;
    
    def __updateTaggedDocs__(self, sims, query, threshold = DocCategorize(0.7, 0.5, 0 ) ): 
        simDocs = []
        rank = 0
        docCount = len(sims)
        logging.warn("__updateTaggedDocs__ sims:{}".format(len(sims)))
        for index, sim in sims:
            #logging.warn("{}, {}".format(index, sim))
            #print(self.taggedDocs[index])
            try:
                if(self.config.name == defaultConfig):                 
                    self.taggedDocs[index] = self.taggedDocs[index]._replace(sentSimilarity = sim)
                else:
                    self.taggedDocs[index] = self.taggedDocs[index]._replace(sectionSimilarity = sim)
                    
                self.taggedDocs[index] = self.taggedDocs[index]._replace(docCount = docCount)     
                self.taggedDocs[index] = self.taggedDocs[index]._replace(rank = rank)     
                self.taggedDocs[index] = self.taggedDocs[index]._replace(query = query)            
                '''
                self.taggedDocs[index] = self.taggedDocs[index]._replace(target = 1 if(sim >= threshold.best) else (2 if(sim < threshold.best and sim > threshold.moderate) else 0) )
                '''
                simDocs.append(
                    SimDocument(sim, index, self.taggedDocs[index])
                )
            except Exception as err:   
                logging.error('Exception {} ({} {})'.format(err, index, sim))  
                
            rank = rank + 1
            
        return simDocs
            #self.taggedDocs[index] = self.taggedDocs[index]._replace(predict = 1 if(sim > threshold) else 0)            
            
     
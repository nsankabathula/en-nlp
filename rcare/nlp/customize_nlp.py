from rcare.nlp.cardinal_recognizer import CardinalRecognizer
from rcare.singleton import SingletonDecorator
import logging;
import spacy;


class __CustomizeNLP:
    
    DEFAULT_MODEL = 'en_core_web_sm'
    
    
    def __init__(self, nlp = None):   
        if(nlp == None): 
            logging.warn("Loading spacy default model: {}".format(self.DEFAULT_MODEL));
            self.nlp = spacy.load(self.DEFAULT_MODEL);
        else:
            self.nlp = nlp;
                       
            
        if(CardinalRecognizer.name not in self.nlp.pipe_names):
            cardinalComponent = CardinalRecognizer(self.nlp, CardinalRecognizer.DEFAULT_CARDINAL_PATTERN , CardinalRecognizer.label)
            self.nlp.add_pipe(cardinalComponent, last = True)
        else:
            logging.warn("NLP is already customized to include pipe: {}".format (CardinalRecognizer.name))

        logging.info(self.nlp.pipe_names)
        
        
        
    def default_doc(self):        
        return self.nlp(" ")
        


class rcare: pass
rcare = SingletonDecorator(__CustomizeNLP)          
from rcare.utils import Utils;


class Feature:
    def __init__(self):
        self.regulation = []
        self.amounts = []
        self.citations = []
        self.conditions = []      
        self.constraints = []
        self.dates = []
        self.definitions = []
        self.durations = []        
        self.organizations = []        
        self.stfd_persons = []
        self.nltk_persons = []
        self.nltk_geopolitical = []
        self.nltk_companies = []
        self.nltk_parties = []
        self.nltk_re_companies = []
        self.address = []
        self.titles = []
        
    def __str__(self):  
        return str(self.__dict__);
    
    def to_dict(self):
        return self.__dict__;
    
   
       
    
    def __merge__(self, fDict, rDict, feature, isTuple = True, delimiter ="^"):
                
        
        #(list(filter(None,list(x)))
        def __merge_tuple__(lst1, lst2, delimiter ="^"):
            return lst1 + list(filter(lambda f: f!='', [ list(map(lambda x: Utils.clean_document(x), list(l2)))    if (type(l2) == tuple) else l2 for l2 in lst2]));
    
        def __merge_simple__(lst1, lst2):
            return lst1 + lst2 if(lst2 != None) else [];      
        
        def __flatten__(lst1, lst2):
            return lst1 + list(filter(lambda f: f!= None, Utils.flatten(lst2)))
            
        def __flatten_string__(lst1, lst2):
            return ' '.join(__flatten__( lst1, lst2));
    
        try:
            if(isTuple):
                return __merge_tuple__(fDict[feature], rDict[feature], delimiter);
            else:
                return __merge_simple__(fDict[feature], rDict[feature]);
            
        except Exception as err:    
            print ("Probelm with " + feature + ", Error: " + str(err) + ". fDict: ", 
                   str (fDict[feature]) + ". rDict: ", str(rDict[feature])  );
            return fDict[feature];

    def merge(self, rec, delimiter ="^"):
        fDict = self.__dict__
        rDict = rec.__dict__
        self.conditions = self.__merge__(fDict, rDict, "conditions", isTuple = True, delimiter = delimiter)
        self.constraints = self.__merge__(fDict, rDict, "constraints", isTuple = True, delimiter = delimiter)
        self.citations = self.__merge__(fDict, rDict, "citations", isTuple = False, delimiter = delimiter)
        self.definitions = self.__merge__(fDict, rDict, "definitions", isTuple = True, delimiter = delimiter)
        self.regulation = self.__merge__(fDict, rDict, "regulation", isTuple = True, delimiter = delimiter)
        self.amounts = self.__merge__(fDict, rDict, "amounts", isTuple = False, delimiter = delimiter)
        self.dates = self.__merge__(fDict, rDict, "dates", isTuple = False, delimiter = delimiter)
        self.organizations = self.__merge__(fDict, rDict, "organizations", isTuple = False, delimiter = delimiter)        
        self.stfd_persons = self.__merge__(fDict, rDict, "stfd_persons", isTuple = False, delimiter = delimiter)
        self.nltk_persons = self.__merge__(fDict, rDict, "nltk_persons", isTuple = False, delimiter = delimiter)
        self.nltk_geopolitical = self.__merge__(fDict, rDict, "nltk_geopolitical", isTuple = False, delimiter = delimiter)
        self.nltk_companies = self.__merge__(fDict, rDict, "nltk_companies", isTuple = True, delimiter = delimiter)
        self.nltk_parties = self.__merge__(fDict, rDict, "nltk_parties", isTuple = False, delimiter = delimiter)
        self.nltk_re_companies = self.__merge__(fDict, rDict, "nltk_re_companies", isTuple = False, delimiter = delimiter)
        self.address = self.__merge__(fDict, rDict, "address", isTuple = False, delimiter = delimiter)
        self.titles = self.__merge__(fDict, rDict, "titles", isTuple = False, delimiter = delimiter)

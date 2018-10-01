from lexnlp.nlp.en.segments.paragraphs import get_paragraphs 
from lexnlp.nlp.en.segments.sentences import get_sentence_list

from lexnlp.extract.en.regulations import get_regulations
from lexnlp.extract.en.addresses.addresses import get_addresses
from lexnlp.extract.en.amounts import get_amounts;
from lexnlp.extract.en.citations  import get_citations;
from lexnlp.extract.en.conditions   import get_conditions;
from lexnlp.extract.en.constraints    import get_constraints;
from lexnlp.extract.en.courts    import get_courts;
from lexnlp.extract.en.dates import get_dates;
from lexnlp.extract.en.definitions import get_definitions;
from lexnlp.extract.en.durations import get_durations
from lexnlp.extract.en.geoentities import get_geoentities;
from lexnlp.extract.en.money import get_money;
from lexnlp.extract.en.pii import get_ssns, get_us_phones, get_pii;
from lexnlp.extract.en.ratios import get_ratios;
from lexnlp.extract.en.regulations import get_regulations;
from lexnlp.extract.en.urls import get_urls;
from lexnlp.nlp.en.segments.titles  import get_titles, build_title_features;
from lexnlp.nlp.en.segments.sections import get_sections;
from lexnlp.nlp.en.segments.pages import get_pages;

from lexnlp import is_stanford_enabled,enable_stanford
enable_stanford()

from lexnlp.extract.en.entities.stanford_ner import get_organizations as stfd_get_organizations, get_persons as stfd_get_persons;

from lexnlp.extract.en.entities.nltk_maxent import get_persons as ntlk_get_persons, get_geopolitical as nltk_get_geopolitical, get_companies as nltk_get_companies
from lexnlp.extract.en.entities.nltk_re import get_parties_as as nltk_re_get_parties_as, get_companies as nltk_re_get_companies

import logging

# riskcare feature model
from rcare.models import Feature

def get_features(text):
    g_features = extract_features_gen(text)
    f = combine_features(g_features);
    return f;

def get_sections(text):    
    return get_sections(text);

def extract_features(sentence):
    if(sentence != None):
        #print (sentence)
        f = Feature();
        #f.regulation = list(get_regulations(sentence, return_source=True));
        #f.amounts = list(get_amounts(sentence, return_sources=True));
        #f.citations = list(get_citations(sentence, return_source=True));
        f.conditions = list(get_conditions(sentence));
        f.constraints = list(get_constraints(sentence));
        #f.dates = list(get_dates(sentence, return_source=True));
        f.definitions = list(get_definitions(sentence, return_sources=True));
        #f.duration = list(get_durations(sentence, return_sources=True));
        #f.organizations = list(get_organizations(sentence)); # causing problem        
        #f.stfd_persons = list(stfd_get_persons(sentence));
        #f.nltk_persons = list(ntlk_get_persons(sentence));
        #f.nltk_geopolitical = list(nltk_get_geopolitical(sentence));
        f.nltk_companies = list(nltk_get_companies(sentence)),
        f.nltk_parties = list(nltk_re_get_parties_as(sentence));
        f.nltk_re_companies = list(nltk_re_get_companies(sentence));
        #f.address = list(get_addresses(sentence));
        #f.titles = list(get_titles(sentence)); #ValueError: Number of features of the model must match the input. Model n_features is 368 and input n_features is 312 

    return f;
    
def extract_features_gen(text):
    str_sentences = (str(sentence) for sentence in get_sentence_list(text))
    g_features = ( extract_features(sentence) for sentence in str_sentences)    
    return g_features;


def combine_features(g_features, count = None):
    i = 0;   
    l_features = Feature();
    flag = True
    for l in g_features:
        try:
            logging.debug("Processing row: %d", i);                      
            if (count == None or i < count): 
                l_features.merge(l);        
                i = i + 1;
            else:
                break;     
        except Exception as err: 
            logging.exception(str(err));
    
    return l_features
               

def print_sample(gen, count = 10):
    i = 0;
    
   
    for l in gen:
        if (i > count): break;
                
        print(l)
        i = i + 1;
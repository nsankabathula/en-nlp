import logging
import string
import unicodedata
import pandas as pd
import numpy as np
import os
from typing import Generator
from sklearn.preprocessing import StandardScaler 

from sklearn.externals import joblib
from rcare.nlp.customize_nlp import rcare
r_nlp = rcare()
from rcare.readutils import DataFrameUtils as Utils
from functools import reduce


# Setup module path

MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

# Load segmenters
DT_SECTION_SEGMENTER_MODEL = joblib.load(os.path.join(MODULE_PATH, "./section_segmenter_dt.pickle"))
MLP_SECTION_SEGMENTER_MODEL = joblib.load(os.path.join(MODULE_PATH, "./section_segmenter_mlp.pickle"))

default_doc = r_nlp.nlp(" ");
DEFAULT_SENT_COUNT = 3


DEBUG_COLUMNS =  ['debug','name',  'line_id', 'target', 'probability']#['debug','name', 'target', 'line_id', "text"]

def split_spacy_doc_g(s_doc, punct_sep = ":"):
    last_punct_index = 0    
    
    for token in s_doc:
        #print (token, token.i)
        if(token.is_punct and token.text == punct_sep and last_punct_index != token.i):
            yield s_doc[last_punct_index:token.i]
            #print (spans)         
            last_punct_index = token.i+1 
        else:
            pass

    if(last_punct_index < len(s_doc)):
        yield s_doc[last_punct_index:]

def split_spacy_doc(s_doc, punct_sep = ":"):
    return list(split_spacy_doc_g(s_doc, punct_sep))


def build_token_features(token, prefix = "", suffix = ""):
    token_feature = \
    {
        
        "{0}_token_is_ucase_{1}".format(prefix,suffix): token.is_upper,     
        "{0}_token_is_lcase_{1}".format(prefix,suffix): token.is_lower,     
        "{0}_token_is_tcase_{1}".format(prefix,suffix): token.is_title,     
        "{0}_token_is_stop_{1}".format(prefix,suffix): token.is_stop,     
        "{0}_token_is_punct_{1}".format(prefix,suffix): token.is_punct,     
        "{0}_token_is_cardinal_{1}".format(prefix,suffix): token._.is_cardinal,     
        "{0}_token_is_digit_{1}".format(prefix,suffix): token.is_digit,     
        "{0}_token_is_alpha_{1}".format(prefix,suffix): token.is_alpha,    
        "{0}_token_is_space_{1}".format(prefix,suffix): token.is_space,          
        "{0}_token_is_left_punct_{1}".format(prefix,suffix): token.is_left_punct,  
        "{0}_token_is_right_punct_{1}".format(prefix,suffix): token.is_right_punct,  
        "{0}_token_like_num_{1}".format(prefix,suffix): token.like_num,  
        "{0}_token_like_email_{1}".format(prefix,suffix): token.like_email,  
        #"{0}_token_tag_{1}".format(prefix,suffix): token.tag,  
        #"{0}_token_idx_{1}".format(prefix,suffix): token.idx,  
        #"{0}_token_is_oov_{1}".format(prefix,suffix): token.is_oov,  
        #"{0}_token_{1}".format(prefix,suffix): token.text,
        
        
    }
    return token_feature

def enrich_line_features(s_doc,  prefix = "", suffix = "",tokens_window_pre = 5, tokens_window_post = -2,):    
    sents = list(s_doc.sents);
    
    sents = sents + list(map(lambda i: default_doc, range(len(sents),DEFAULT_SENT_COUNT)))
    
    e_line_features = line_features(s_doc, prefix = prefix, suffix = suffix, \
                                    tokens_window_pre = tokens_window_pre, tokens_window_post = tokens_window_post)
    
    for idx, sent in enumerate(sents):
        if(idx < DEFAULT_SENT_COUNT):
            e_line_features.update(line_features(s_doc = sent, prefix = "sent", suffix = idx, tokens_window_pre = 0, tokens_window_post= 0))
        else:
            break;
    
    return e_line_features
    
def line_features(s_doc,  prefix = "", suffix = "", tokens_window_pre = 5, tokens_window_post = -2 ):    
    feature = \
    {
        #"{0}_all_ucase_{1}".format(prefix,suffix): s_doc.text.upper() == s_doc.text,
        #"{0}_all_lcase_{1}".format(prefix,suffix): s_doc.text.lower() == s_doc.text, # if tcount == lower_tcount
        
        "{0}_tcount_{1}".format(prefix,suffix): len(s_doc),                
        "{0}_lcase_tcount_{1}".format(prefix,suffix): int(sum([1 for token in s_doc if token.is_lower])),
        "{0}_ucase_tcount_{1}".format(prefix,suffix): int(sum([1 for token in s_doc if token.is_upper])),
        "{0}_tcase_tcount_{1}".format(prefix,suffix): int(sum([1 for token in s_doc if token.is_title])),
        
        "{0}_alpha_tcount_{1}".format(prefix,suffix): int(sum([1 for token in s_doc if token.is_alpha])),
        "{0}_space_tcount_{1}".format(prefix,suffix): int(sum([1 for token in s_doc if token.is_space])),
        "{0}_punct_tcount_{1}".format(prefix,suffix): int(sum([1 for token in s_doc if token.is_punct])),        
        "{0}_stop_tcount_{1}".format(prefix,suffix): int(sum([1 for token in s_doc if token.is_stop])),     
                
        "{0}_digit_tcount_{1}".format(prefix,suffix): int(sum([1 for token in s_doc if token.is_digit])),
        "{0}_number_tcount_{1}".format(prefix,suffix): int(sum([1 for token in s_doc if token.like_num])),        
        
        "{0}_cardinal_count_{1}".format(prefix,suffix): int(sum([1 for token in s_doc if token._.is_cardinal])/2),        
        
        "{0}_has_cardinal_{1}".format(prefix,suffix): s_doc._.has_cardinal,
     }  
        
            
    feature ["{0}_is_empty_{1}".format(prefix,suffix)] = \
    feature ["{0}_tcount_{1}".format(prefix,suffix)] == feature ["{0}_space_tcount_{1}".format(prefix,suffix)]
    
    for i in range(tokens_window_post, tokens_window_pre):
        try:
            token = s_doc[i]
        #print (i, s_lines[8][i:i+1])        
        except IndexError:
            token = default_doc[0]

        feature.update (build_token_features(token, "{0}_{1}".format(prefix,suffix), i if i >= 0 else "n_"+ str(i * -1) ))    

    
    return feature;

def build_section_break_features_spacy(lines, line_id, line_window_pre, line_window_post, characters=string.printable,
                                 include_doc = None, debug = False, nlp = r_nlp.nlp, using_spacy_lines = True):
    def array_to_dict(arr, prefix= "", suffix = ""):
        return {"{0}_{1}_{2}".format(prefix,i,suffix): arr[i] for i in range(0, len(arr))}
    """
    Build a feature vector for a given line ID with given parameters.

    :param lines:
    :param line_id:
    :param line_window_pre:
    :param line_window_post:
    :param characters:
    :param include_doc:
    :return:
    """
    # Feature vector
    feature_vector = {}

    # Check start offset
    if line_id < line_window_pre:
        line_window_pre = line_id

    # Check final offset
    if (line_id + line_window_post) >= len(lines):
        line_window_post = len(lines) - line_window_post - 1

    # Iterate through window
    for i in range(-line_window_pre, line_window_post + 1):
        try:
            if(using_spacy_lines):
                line = lines[line_id + i]
            else: 
                line = nlp(lines[line_id + i] if len(lines[line_id + i].strip()) > 0 else " ")
        except IndexError:
            continue

        if(i == 0):
            feature_vector.update(enrich_line_features(line, "line", i))
            #feature_vector.update(array_to_dict(line.vector, "vector", i))
            if debug:
                feature_vector["debug"] = line.text     
        else:
            feature_vector.update(line_features(line, "line", i if i >= 0 else "n_"+ str(i * -1) )) 
            
                            
                     
    # Add doc if requested
    if include_doc: 
        feature_vector.update(include_doc)
   
                             

    return feature_vector
  
    
def build_model(lines, window_pre = 3, window_post = 3, score_threshold=0.5, include_doc = None, \
                debug = False, nlp = r_nlp.nlp, using_spacy_lines = True):
    #doc_distribution = build_document_line_distribution(text)
       
    def populate_start_sent(row):
        trueCase = ((row.line_n_1_token_is_punct_n_1 == 1 or row.line_is_empty_n_1 == 1 or \
                     (row.line_is_empty_n_1 == -1 and row.line_is_empty_n_2 == -1 and row.line_is_empty_n_3 == -1)) \
                    and \
                        ( row.line_0_token_is_ucase_0 == 1 or row.line_0_token_is_tcase_0 == 1 \
                        or row.line_0_token_is_cardinal_0 == 1 ) )
        return pd.Series([1 if (trueCase) else 0])

    def populate_end_sent(row):
        trueCase = ((row.line_0_token_is_punct_n_1 == 1 or row.line_is_empty_0 == 1 ) \
                    and \
                        ( row.line_1_token_is_ucase_0 == 1 or row.line_1_token_is_tcase_0 == 1 \
                        or row.line_1_token_is_cardinal_0 == 1 
                         or (row.line_is_empty_1 == -1 and row.line_is_empty_2 == -1 and row.line_is_empty_3 == -1)
                        ) )
        return pd.Series([1 if (trueCase) else 0])

 
    if(using_spacy_lines):           
        s_lines = lines;
    else:
        s_lines = list(map(lambda line : nlp(line.strip()) if(len(line.strip()) > 0 ) else default_doc , lines))
        '''
        for line in lines:
            s_lines.append(nlp(line.strip()) if(len(line.strip()) > 0 ) else default_doc)        
        '''
    using_spacy_lines = True
    
    for idx in range(len(s_lines), window_pre + window_post + 1):
        s_lines.append(default_doc)
        
    test_feature_data = []
    logging.info('%d', len(s_lines))
    for line_id in range(len(s_lines)):
        if(include_doc):            
            include_doc.update ({'line_id': line_id});
            
        test_feature_data.append(
            build_section_break_features_spacy(s_lines, line_id, window_pre, window_post, include_doc = include_doc, debug = debug, \
                                               nlp = nlp, using_spacy_lines = using_spacy_lines))
   
    # Predict page breaks
    test_feature_df = pd.DataFrame(test_feature_data).fillna(-1)  
    test_feature_df['line_is_start_of_sent_0'] = test_feature_df.apply(populate_start_sent, axis = 1)    
    test_feature_df['line_is_end_of_sent_0'] = test_feature_df.apply(populate_end_sent, axis = 1)  
    #logging.info(list(test_feature_df.query("debug == -1").index))
            
    return test_feature_df


def annotated_model(model_df):
  
    def populate(row):
        #return pd.Series(s_nlp(doc).vector);["prob_false", "prob_true"]
        trueCase = (row.line_is_first_token_cardinal_0 == True and row.line_token_after_cardinal_title_0 == True \
                    and row.line_is_first_token_cardinal_1 == False and row.line_is_first_token_cardinal_n_1 == False\
                    and row.line_tcount_0 > 0 and row.line_tcount_0 < 11 ) \
        or (row.line_tcount_0 == row.line_tokens_sw_upper_0 and row.line_tcount_0 > 0 and row.line_tcount_0 <= 3 \
            and (row.line_tcount_1 > 3 or row.line_tcount_2 > 3)) \
        or (row.line_tcount_before_colon_0 > 0  and row.line_tcount_before_colon_0 <  7 and \
            (\
             (row.line_special_tcount_before_colon_0 > 0 and row.line_special_tcount_before_colon_0 <= row.line_tcount_before_colon_0)\
             or\
             (row.line_title_tcount_before_colon_0 == row.line_tcount_before_colon_0 and row.line_tcount_before_colon_0 > 0)\
            ))
        
        return pd.Series([1 if (trueCase) else 0])
    
    logging.debug("annotating the model")
    #model_df[['target']] = model_df.apply(populate, axis = 1)    
    model_df['target'] = 0
    return model_df;

def build_annotated_model(text, window_pre = 3, window_post = 3, score_threshold=0.6, include_doc = None, debug = False, nlp = r_nlp.nlp, using_spacy_lines = False, using_existing_model = True, use_mlp = True ):
    
    if(using_spacy_lines):
        lines = []
        for sent in nlp(text).sents:
            lines.append(sent.as_doc())
    else:
        lines = text.splitlines()    
    
    test_feature_df = build_model(lines, window_pre, window_post, score_threshold, include_doc, debug = debug, 
                                  nlp = nlp, using_spacy_lines = using_spacy_lines)
    test_feature_df.replace ({True: 1, False :0}, inplace = True)
    if(using_existing_model):
        droped_columns, test_feature_df = drop_columns (test_feature_df, DEBUG_COLUMNS)
        predicted_df = predict(test_feature_df, use_mlp)  
        droped_columns['target'] = predicted_df['prob_true'] >= score_threshold
        droped_columns['probability'] = predicted_df['prob_true']
        test_feature_df = pd.concat([test_feature_df,droped_columns], axis = 1)
    else:
        test_feature_df = annotated_model(test_feature_df)
        
    test_feature_df.replace ({True: 1, False :0}, inplace = True)
    return test_feature_df

def predict(test_feature_df, use_mlp = True, ):
    if(use_mlp):                
        scaled_features = StandardScaler().fit_transform(test_feature_df.values)  
        scaled_features_df = pd.DataFrame(scaled_features, index=test_feature_df.index, columns=test_feature_df.columns)        
        test_predicted_lines = MLP_SECTION_SEGMENTER_MODEL.predict_proba(scaled_features_df)
    else:
        test_predicted_lines = DT_SECTION_SEGMENTER_MODEL.predict_proba(test_feature_df)
    
    predicted_df = pd.DataFrame(test_predicted_lines, columns=["prob_false", "prob_true"])
    
    return predicted_df;
    

def drop_column (df, column, columns = []):
    columns = columns if (len(columns)>0) else df.columns.values
    #logging.info('{} - Requested {}'.format(column, columns))
    if(column in columns):
        #logging.info('Droping column {}'.format(column));
        return df[[column]], df.drop(columns = [column])
    else:
        return pd.DataFrame(columns = [column]), df;

def drop_columns (df, columns):
    d_df = []
    for column in columns:
        droped_df, df = drop_column(df, column, df.columns.values)
        d_df.append(droped_df)
        
    for i in range(0,len(d_df)):
        if(i > 0):
            d_df[i] = d_df[i-1].join(d_df[i])
            
    d_df = d_df[len(d_df)-1]
    return d_df, df;
    
def get_sections(text, window_pre=3, window_post=3, score_threshold=0.5, debug = False, nlp = r_nlp.nlp, use_mlp = False) -> Generator:
        
    lines = text.splitlines()
       
    test_feature_df = test_feature_df = build_model(lines, window_pre, window_post, score_threshold, include_doc = None, debug = debug, 
                                  nlp = nlp, using_spacy_lines = False)
    
    
    droped_columns, test_feature_df = drop_columns (test_feature_df, DEBUG_COLUMNS)
    #logging.info(list(test_feature_df.columns.values))
    test_feature_df.replace ({True: 1, False :0}, inplace = True)
                
    #yield test_feature_df  
    predicted_df = predict(test_feature_df, use_mlp)       
    
    section_breaks = predicted_df.loc[predicted_df["prob_true"] >= score_threshold, :].index.tolist()
            
    if len(section_breaks) > 0:
        # Get first break
        pos0 = 0
        pos1 = section_breaks[0]
        section = "\n".join(lines[pos0:pos1])
        if len(section.strip()) > 0:
            yield section

        # Iterate through section breaks
        for i in range(len(section_breaks) - 1):
            # Get breaks
            pos0 = section_breaks[i]
            pos1 = section_breaks[i + 1]
            # Get text
            section = "\n".join(lines[pos0:pos1])
            if len(section.strip()) > 0:
                yield section

        # Yield final section
        section = "\n".join(lines[section_breaks[-1]:])
        if len(section.strip()) > 0:
            yield section


import logging
import sqlite3
import pandas as pd
from rcare.readutils import DataFrameUtils as Utils
from rcare.nlp.sections_rcare import build_annotated_model


def build_training_data(dir_path):
    logging.info("Building Training Data .. Start")       
    connex  =  sqlite3.connect(dir_path + "/training.db")  # Opens file if exists, else creates file

    df = pd.concat([Utils.read_data(path = dir_path) ], ignore_index  = True)

    test_feature_df = pd.DataFrame()
    drop_test_feature_df = pd.DataFrame()
    #debug_columns = ['debug','name',  'line_id']

    for index, row in df.iterrows():


        logging.info("**************************************");     

        if(index == 0):
            if_exists = "replace"
        else: 
            if_exists = "append"


        logging.info("Index : %d %s (%s)", index, row['name'], if_exists)

        text = row['data']
        test_feature_df = build_annotated_model(text, include_doc = {'name': row['name']}, debug = True)

        #debug_df = test_feature_df[debug_columns]
        #test_feature_df.drop(columns=debug_columns, inplace= True)

        #drop_test_feature_df, test_feature_df = drop_columns(test_feature_df, debug_columns)
        test_feature_df.to_sql(name  = "training_data", con  = connex, if_exists = if_exists)


        logging.info("-------------------------------------");   
        #break;

    connex.commit()    
    connex.close();
    logging.info("Building Training data complete")          
        
        
    
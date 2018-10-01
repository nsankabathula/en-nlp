import sys;
import logging;
import numpy as np
import  sqlite3;
import pandas as pd
sys.path.append('/home/paperspace/dev/en-nlp/')
sys.path.append('/home/paperspace/dev/en-nlp/rcare/nlp')
sys.path.append('/home/paperspace/dev/en-nlp/rcare')


ES_CONFIG = {
    "host": [
        {
            "host": "localhost",                  
            "port":"9200"
        }
    ],
    "log": "error"
};
SQL_DATA_PATH = "/home/paperspace/dev/sqllite-node/app/data/training.db"
QUERY = "select tf.* from training_features tf join meta_data md on md.txtFileName = tf.name and md.useForTraining = 1"
CHUNK_SIZE = 500

try:
    from rcare.es_helper import ESHelper
    def pushData(sqlPath, query, chunkSize, esConfig):
        
        def get_id(item):
            return "{}_{}".format(item['name'] ,str(item['line_id']))
        
        connex  =  sqlite3.connect(sqlPath)
        esHelper = ESHelper(esConfig)
        df_sql = pd.read_sql_query(sql=query, con=connex, chunksize = chunkSize)
        idx = 0
        print("Starting load...")
        for test_feature_df in df_sql:
            count = test_feature_df[test_feature_df.columns[0]].count()
            print("Processing Index : {} with size {} - {}".format(idx,count, idx * count ));  
    
            idx = idx + 1
            esHelper.bulk_stream_collection( test_feature_df.to_dict(orient='records'), 
                        index = "segmentation", doc_type = "training_features",  get_id = get_id)
            
        print("Complete..")
     
    pushData(SQL_DATA_PATH, QUERY, CHUNK_SIZE, ES_CONFIG)
except Exception as err:   
    print('Exception {}'.format(err))        
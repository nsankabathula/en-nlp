import sys
import logging;
import sqlite3;
import pandas as pd
import os
#MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
#print (MODULE_PATH)
#sys.path.append(MODULE_PATH)
sys.path.append('/home/paperspace/dev/en-nlp/')
sys.path.append('/home/paperspace/dev/en-nlp/rcare/nlp')
sys.path.append('/home/paperspace/dev/en-nlp/rcare')

FILE_PATH = '/home/paperspace/dev/en-nlp/data/lexpredict-contraxsuite-samples/cca_2011_Q3/text/'
#print({'path':sys.path})
errors = []
try:
    # Trying to find module in the parent package
    from rcare.readutils import DataFrameUtils as Utils   
    from rcare.nlp.customize_nlp import rcare
    import spacy;
    from rcare.nlp.cardinal_recognizer import CardinalRecognizer
    from rcare.singleton import SingletonDecorator
    nlp = rcare(spacy.load('en_core_web_sm')).nlp    
    from rcare.nlp.sections_rcare import build_annotated_model, drop_columns
    #r_nlp = rcare()
    #from rcare.nlp.sections_rcare import build_annotated_model
    def build_annotated_model_file(fileName, filePath, sqlLitePath = None, tableName = "predicted_data"):    
        logging.info("Building Training Data .. Start")  
        connex  =  sqlite3.connect(sqlLitePath + "/training.db") if (sqlLitePath) else None;
        test_feature_df = pd.DataFrame()
        
        for index, fileData in enumerate(Utils.read_data_gen(filenames = [fileName], path = filePath)):     
            text =  fileData.text
            name = fileData.name
            logging.info("Index : %d %s", index, name)
            
            if(tableName != "predicted_data" ):
                score_threshold=0.6;
            else:            
                score_threshold=0.55
                            
            test_feature_df = build_annotated_model(text, include_doc = {'name': name}, debug = True, \
                                                    using_spacy_lines = False, nlp = nlp, use_mlp = True, score_threshold=score_threshold)
            
            if(tableName != "predicted_data" ):
                droped_columns, test_feature_df = drop_columns (test_feature_df, ['probability'])
                
            with connex:
                cur = connex.cursor()
                print ("{}".format( tableName))
                # Create table
                print ("delete {}".format( cur.execute("DELETE FROM " + tableName + " WHERE name = '"+ fileName + "'").rowcount))
                connex.commit()
                print ("shape {}".format( test_feature_df.shape));
                test_feature_df.to_sql(name  = tableName, con  = connex, if_exists = 'append', index = False) 
                connex.commit()
            

        logging.info("Building Training data complete") 
        
        return test_feature_df.to_json(orient='records')
    #print(sys.argv)
    build_annotated_model_file(fileName = sys.argv[1], filePath = sys.argv[2], sqlLitePath = sys.argv[3], tableName = sys.argv[4])
except ImportError as err:
    print('Relative import failed {}'.format(err))
except Exception as err:
    print('Exception {}'.format(err))    


    




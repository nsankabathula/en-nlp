from rcare.models import Feature
from rcare.utils import Utils
import pandas as pd
import os
from multiprocessing import Pool
import numpy as np 
import logging;
import timeit
from collections import namedtuple

FileData = namedtuple('FileData', ['name', 'text']);

class DataFrameUtils(Utils):
    
    num_partitions = 10 #number of partitions to split dataframe
    num_cores = 4 #number of cores on your machine

    @staticmethod
    def read_data(filenames = None, path = Utils.DATA_PATH, numfiles = None):
        if(filenames == None):
            filenames = list(filter(lambda x: (x not in Utils.BLACK_LISTED_FILES and \
                                               (x.endswith(".txt") or x.endswith(".TXT")  )), os.listdir(path )))
            logging.info('Number of files {} in {}'.format(len(filenames), path));            

        df = pd.DataFrame(columns=Utils.COLUMNS)
        numfiles = len(filenames) if(numfiles == None) else  numfiles;
        for filename in filenames[:numfiles]:        
            logging.info("Reading file %s", filename);
            data = Utils.read_file_data(path +'/' +filename)
            f_dict = {}
            #f_dict = f.to_dict();
            f_dict['name'] = filename
            f_dict['data'] = data
            #f_dict['clean_data'] =  Utils.clean_document(data)
            logging.info("Appending to dataframe %s", filename);
            df = df.append(f_dict, ignore_index = True)    
            
        logging.info('Reading file data done..');
        return df;

    @staticmethod
    def read_data_gen(filenames = None, path = Utils.DATA_PATH, numfiles = None):
        
        if(filenames == None):
            filenames = sorted(list(filter(lambda x: (x not in Utils.BLACK_LISTED_FILES and\
                                               (x.endswith(".txt") or x.endswith(".TXT")  )), os.listdir(path ))))
           
            logging.info('Number of files {} in {}'.format(len(filenames), path));            

        df = pd.DataFrame(columns=Utils.COLUMNS)
        numfiles = len(filenames) if(numfiles == None) else  numfiles;
        for filename in filenames[:numfiles]:        
            logging.info("Reading file %s", filename);
            data = Utils.read_file_data(path +'/' +filename)
            fileData = FileData(filename, data)  
            yield fileData;
            
        logging.info('Reading file data done..');
   
    @staticmethod
    def read_csv(path = Utils.DATA_PATH, filename = '001.txt', sep = '|', nrows = None, skiprows = None):
        filepath = path + filename
        logging.info("Reading  %s", filepath);
        df = pd.read_csv(filepath , sep = sep, nrows = nrows, skiprows = skiprows)
        return df;
    
    @staticmethod
    def parallelize_dataframe(df, df_function, num_cores = num_cores, num_partitions = num_partitions ):
        start = timeit.timeit()
        logging.info("Pool started  df rows %d", len(df.index));
        
        df_split = np.array_split(df, num_partitions)
        pool = Pool(num_cores)        
        df = pd.concat(pool.map(df_function, df_split))
        pool.close()
        pool.join()
        
        end = timeit.timeit()
        
        logging.info("Pool complete ...%d", end - start);
        return df
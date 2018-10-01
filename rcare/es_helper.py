from elasticsearch import Elasticsearch
from elasticsearch import helpers as esHelper
from rcare.singleton import SingletonDecorator
from collections import namedtuple
import logging;
ESID = "esId"
_ESID = "_id"
_SOURCE = "_source"

ESAlias = namedtuple('ESAlias',"index name action")

class __IESHelper:
    def connection(self): pass
    def info(self): pass
    def bulk_stream(self, iterator):pass
    def create_index(self, index, doc_type, mapping, drop_create = False):pass
    
class __ESHelper(__IESHelper):
    
    def __init__(self, config):       
        self.__esconfig = config
        self.__esconnection = Elasticsearch( hosts= self.__esconfig['host'], timeout=30)
                
    def connection(self):
        return self.__esconnection;

    def search(self, index, doc_type, body, name = "ESResult"):
        try:
            logging.info("{}/{} - {}".format(index, doc_type, body) )
            result = self.__esconnection.search(index=index, doc_type = doc_type, body = body )['hits']
            keys = [key for key in result]
            EsResult = namedtuple(name , keys)
            return EsResult(*[result[key] for key in keys])
        except Exception as err:
            raise err;     
    
    def alias(self, changes):
        
        for change in changes:
            logging.warn ("alias => {}".format(change))
            try:
                if(change.action == "remove"):
                    pass
                else:                
                    self.__esconnection.indices.put_alias(index=change.index, name=change.name)
            except Exception as err:
                raise err;      
                
    def search_nt(self, index, doc_type = None, body = None, name = "ESource" ):
        try:
            
            result = self.search(index, doc_type = doc_type, body = body)
            keys =  list(filter(lambda x: not x.startswith('_'),  [key for key in result.hits[0][_SOURCE]] ))
            source_np = namedtuple(name, keys)
                
            for hit in result.hits:                
                yield source_np(*[hit[_SOURCE][key] for key in keys])
        except Exception as err:
            raise err;               
        
    def create_index(self, index, doc_type, mapping, drop_create = False):
        if(drop_create == True):
            self.__esconnection.indices.delete(index=index, ignore=[400, 404])
           
        self.__esconnection.indices.create(index=index, ignore=400)
        
        if(mapping != None and doc_type != None):
            self.__esconnection.indices.put_mapping(index=index, doc_type=doc_type, body=mapping)
            
    def prep_bulk_iterator(self, collection,index , doc_type, get_id):
            for item in collection:   
                try:                    
                    yield {
                           '_op_type': 'update',
                           '_index': index,
                           '_type': doc_type, 
                           '_id': get_id(item),                        
                            "doc" : item,
                            "doc_as_upsert":True
                    }
                except Exception as err:
                    print(err)
                    yield {
                           '_index': index,
                           '_type': doc_type,                           
                           '_source': item
                    }

    def bulk_stream_collection(self, collection,index , doc_type, get_id):
        logging.info("bulk_stream_collection {}/{} ".format(index, doc_type) )
        self.bulk_stream(self.prep_bulk_iterator(collection, index, doc_type, get_id = get_id))
        
    def bulk_stream(self, iterator, failureThreshold = 100):
        success  = 0;
        failure = 0;
        for ok, result in esHelper.streaming_bulk(self.__esconnection, iterator, raise_on_error = False):
            if not ok:
                failure = failure + 1                                    
                __, result = result.popitem()
                if result['status'] == 409:
                    print('Duplicate event detected, skipping it: {}'.format (result))
                else:
                    print('Failed to record event: {}'.format(result))
                
                if(failure >= failureThreshold): 
                    print('Reached threshold.')
                    raise result;
            else:
                success = success + 1;
                
        print('Success count: {}, Failure count: {}'.format(success, failure))
           
    def info(self):
        return self.__esconnection.info()
    
                    

class ESHelper: pass
ESHelper = SingletonDecorator(__ESHelper)

import  sqlite3
import pandas as pd

class SqlUtils(object):
    
    def __init__(self, sqlDir):
        self.sqlDir = sqlDir
                
    def test(self):
        return self.select("select * from sqlite_master");        
    
    def drop(self, tableName):
        try: 
            cur =  self.__execute__(sqlite3.connect(self.sqlDir), "DROP TABLE {}".format(tableName));
            return cur.fetchall();
        except Exception as err:
            raise err
            
    def create(self, stmt):
        try: 
            cur =  self.__execute__(sqlite3.connect(self.sqlDir), stmt);
            return cur.fetchall();
        except Exception as err:
            raise err
            
    def select(self, stmt, chunksize = None):
        try: 
            df =  pd.read_sql_query(sql=stmt, con=sqlite3.connect(self.sqlDir), chunksize = chunksize)
            return df;
        except Exception as err:
            raise err  
            
    def __execute__(self, connex, stmt):
        result = [];
        with connex:
            try:
                cur = connex.cursor()
                result = cur.execute(stmt)
                connex.commit()
            except Exception as err:
                result = err
                print('Exception {}'.format(err)) 
                connex.rollback()
                raise err
        return result;
                         
            

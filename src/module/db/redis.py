import redis


class Redis:
    def __init__(self, host, port, db):
        self.host = host
        self.port = port
        self.db = db
        self.r = redis.Redis(host=self.host, port=self.port, db=self.db)
    
    def get(self, key):
        return self.r.get(key)
    
    def set(self, key, value):
        self.r.set(key, value)  
        
    def delete(self, key):
        self.r.delete(key)
    
    def size(self):
        return self.r.dbsize()
    
    def get_many(self, keys):
        return self.r.mget(keys)
    
    def keys(self):
        return self.r.keys()
    


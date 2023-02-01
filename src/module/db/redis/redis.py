import redis
from typing import List

from src.module.db.redis.serializers import JSONSerializer


class Redis:
    def __init__(self, host, port, db):
        self.host = host
        self.port = port
        self.db = db
        self._serializer_cls = JSONSerializer
        self._serializer = self._serializer_cls()
        self.r = redis.Redis(host=self.host, port=self.port, db=self.db)

    def decode(self, value):
        """
        Decode the given value.
        """
        try:
            value = int(value)
        except (ValueError, TypeError):
            value = self._serializer.loads(value)
        return value
    
    def encode(self, value):
        """
        Encode the given value.
        """
        if not isinstance(value, int):
            value = self._serializer.dumps(value)
            return value
        return value
    
    def get(self, key):
        return self.decode(self.r.get(key))
    
    def set(self, key, value):
        self.r.set(key, value)  
        
    def delete(self, key):
        self.r.delete(key)
    
    def size(self):
        return self.r.dbsize()
    
    def get_many(self, keys) -> List:
        return [self.decode(v) for v in self.r.mget(keys)]
    
    def keys(self):
        return self.r.keys()
    
    def all(self) -> List:
        return self.get_many(self.keys())


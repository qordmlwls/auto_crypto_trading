import json

from src.module.db.redis.encoders import JSONEncoder


class BaseSerializer:
    def __init__(self):
        pass

    def dumps(self, value):
        raise NotImplementedError

    def loads(self, value):
        raise NotImplementedError


class JSONSerializer(BaseSerializer):
    encoder_class = JSONEncoder

    def dumps(self, value):
        return json.dumps(value, cls=self.encoder_class).encode()

    def loads(self, value):
        return json.loads(value.decode())
    
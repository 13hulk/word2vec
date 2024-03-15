from pydantic import BaseModel


class Sentinel(BaseModel):
    obj: object = None

    class Config:
        arbitrary_types_allowed = True


class ProxyObject:
    def __init__(self, sentinel: Sentinel):
        self.sentinel = sentinel

    def __getattr__(self, attr):
        try:
            return getattr(self.sentinel.obj, attr)
        except AttributeError as error:
            raise AttributeError("Is the proxy object initialized?") from error

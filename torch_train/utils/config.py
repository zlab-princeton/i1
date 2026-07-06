from __future__ import annotations


class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            self[k] = self._wrap(v)

    @classmethod
    def _wrap(cls, v):
        if isinstance(v, ConfigDict):
            return v
        if isinstance(v, dict):
            return cls(v)
        return v

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = self._wrap(value)

    def get(self, key, default=None):
        return self[key] if key in self else default

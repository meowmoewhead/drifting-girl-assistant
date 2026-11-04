# Adapted from LmeSzinc/AzurLaneAutoScript @ 77f4d01fcd2b0acab05a4d89260e8a0a9cd03d21
# Copyright the ALAS contributors. SPDX-License-Identifier: GPL-3.0-only
from .decorator import del_cached_property
VALID_SERVER = ['default']

class Resource:
    # Class property, record all button and templates
    instances = {}
    # Instance property, record cached properties of instance
    cached = []

    def resource_add(self, key):
        Resource.instances[key] = self

    def resource_release(self):
        for cache in self.cached:
            del_cached_property(self, cache)

    @classmethod
    def is_loaded(cls, obj):
        if hasattr(obj, '_image') and obj._image is None:
            return False
        elif hasattr(obj, 'image') and obj.image is None:
            return False
        return True

    @classmethod
    def resource_show(cls):
        from alas_core.log import logger
        logger.hr('Show resource')
        for key, obj in cls.instances.items():
            if cls.is_loaded(obj):
                continue
            logger.info(f'{obj}: {key}')

    @staticmethod
    def parse_property(data, s=None):
        """
        Parse properties of Button or Template object input.
        Such as `area`, `color` and `button`.

        Args:
            data: Dict or str
            s (str): Load from given a server or load from global attribute `server.server`
        """
        if s is None:
            s = 'default'
        if isinstance(data, dict):
            return data[s]
        else:
            return data

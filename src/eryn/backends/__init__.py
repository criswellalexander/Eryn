# -*- coding: utf-8 -*-

from .backend import Backend

from .hdfbackend import HDFBackend, TempHDFBackend

from .supbackend import SupplementalBackend

__all__ = ["Backend", "HDFBackend", "SupplementalBackend", "TempHDFBackend", "get_test_backends"]


def get_test_backends():
    backends = [Backend]

    try:
        import h5py  # NOQA
    except ImportError:
        pass
    else:
        backends.append(TempHDFBackend)

    return backends

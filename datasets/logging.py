#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Logging."""

import time
import os
import logging
import functools
import sys
from fvcore.common.file_io import PathManager


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return PathManager.open(filename, "a")


def setup_logging(output_dir=None):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

    logging.root.handlers = []

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(plain_formatter)
    logger.addHandler(ch)

    if output_dir is not None:
        filename = os.path.join(output_dir, "stdout.log")
        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)

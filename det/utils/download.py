#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :download.py
@Author :CodeCat
@Date   :2025/11/5 10:22
"""
import os
import os.path as osp
import sys
import yaml
import time
import shutil
import requests
import tqdm
import hashlib
import base64
import binascii
import tarfile
import zipfile
import errno
from loguru import logger
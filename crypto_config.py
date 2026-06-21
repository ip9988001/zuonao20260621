# -*- coding: utf-8 -*-
"""左脑统一密钥管理 — 本地部署版"""
from _keys import (
    SECRET_KEY, AES_KEY_SALT,
    derive_engine_key, generate_universal_sign_prefix, derive_backup_key,
)

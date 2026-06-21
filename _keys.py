# -*- coding: utf-8 -*-
"""左脑密钥模块 — 本地部署版（已移除DRM授权依赖）"""
import hashlib
import hmac

# 本地生成的密钥（用于数据签名和备份加密，非DRM）
SECRET_KEY = hashlib.sha256(b"ZUONAO_LOCAL_ENGINE_KEY_V3").digest()
AES_KEY_SALT = hashlib.sha256(b"ZUONAO_AES_SALT_LOCAL").digest()[:16]

def derive_engine_key(sign_prefix: str) -> bytes:
    """派生引擎密钥"""
    return hashlib.sha256(sign_prefix.encode("utf-8") + AES_KEY_SALT).digest()

def generate_universal_sign_prefix() -> str:
    """生成通用签名前缀"""
    return hmac.new(SECRET_KEY, b"ZUONAO_UNIVERSAL_SIGN_V3", hashlib.sha256).hexdigest()[:32]

def derive_backup_key() -> bytes:
    """派生备份加密密钥"""
    return hashlib.sha256(AES_KEY_SALT + b"_BACKUP_DERIVE_V3").digest()

from datetime import timedelta

import alibabacloud_oss_v2 as oss
from dotenv import load_dotenv
from fastapi import APIRouter

from app.common.config import require_env

load_dotenv()
router = APIRouter()

# 部署关键：OSS 连接/凭证必须显式配置，缺失即启动失败。
# 注意：之前 credential 由 SDK 的 EnvironmentVariableCredentialsProvider 隐式读 env，
# 代码里看不到取值路径、漏配也不报错——这里改为显式 require_env + StaticCredentialsProvider 注入。
OSS_ACCESS_KEY_ID = require_env("OSS_ACCESS_KEY_ID")
OSS_ACCESS_KEY_SECRET = require_env("OSS_ACCESS_KEY_SECRET")
OSS_REGION = require_env("OSS_REGION")
OSS_BUCKET = require_env("OSS_BUCKET")
OSS_ENDPOINT = require_env("OSS_ENDPOINT")

_client = None


def _get_oss_client():
    global _client
    if _client is None:
        cfg = oss.config.load_default()
        cfg.credentials_provider = oss.credentials.StaticCredentialsProvider(
            access_key_id=OSS_ACCESS_KEY_ID,
            access_key_secret=OSS_ACCESS_KEY_SECRET,
        )
        cfg.region = OSS_REGION
        _client = oss.Client(cfg)
    return _client


@router.get("/oss/presign")
def chat_endpoint(filename: str):
    content_type_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
        "mp4": "video/mp4",
        "webm": "video/webm",
        "mov": "video/quicktime",
    }
    ext = filename.split(".")[-1].lower() if "." in filename else "jpg"
    content_type = content_type_map.get(ext, "application/octet-stream")

    pre_result = _get_oss_client().presign(
        oss.PutObjectRequest(
            bucket=OSS_BUCKET,
            key=filename,
            content_type=content_type,
        ),
        expires=timedelta(seconds=3600),
    )

    return {
        "uploadUrl": pre_result.url.strip('"'),
        "contentType": content_type,
        "accessUrl": f"https://{OSS_BUCKET}.{OSS_ENDPOINT}/{filename}",
    }
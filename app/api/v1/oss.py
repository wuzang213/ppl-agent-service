import os
from datetime import timedelta

import alibabacloud_oss_v2 as oss
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException

load_dotenv()
router = APIRouter()

_client = None


def _get_oss_client():
    global _client
    if _client is None:
        credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
        cfg = oss.config.load_default()
        cfg.credentials_provider = credentials_provider
        cfg.region = os.getenv("OSS_REGION")
        _client = oss.Client(cfg)
    return _client


@router.get("/oss/presign")
def chat_endpoint(filename: str):
    bucket = os.getenv("OSS_BUCKET")
    if not bucket:
        raise HTTPException(status_code=500, detail="OSS_BUCKET 未配置")

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

    endpoint = os.getenv("OSS_ENDPOINT")
    pre_result = _get_oss_client().presign(
        oss.PutObjectRequest(
            bucket=bucket,
            key=filename,
            content_type=content_type,
        ),
        expires=timedelta(seconds=3600),
    )

    return {
        "uploadUrl": pre_result.url.strip('"'),
        "contentType": content_type,
        "accessUrl": f"https://{bucket}.{endpoint}/{filename}",
    }
"""Local configuration. Credentials never belong in displayable objects."""
from dataclasses import dataclass, field
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    base_url: str = "https://financialmodelingprep.com"
    auth_mode: str = "query"
    ca_bundle: str | None = None
    daily_history_days: int = 730
    intraday_history_days: int = 35
    cache_ttl_seconds: int = 300
    request_timeout_seconds: int = 20


def load_settings() -> Settings:
    """Only load this project's .env; environment values take precedence."""
    load_dotenv(PROJECT_DIR / ".env", override=False)
    base_url = os.environ.get("FMP_BASE_URL", "https://financialmodelingprep.com").strip().rstrip("/")
    parsed = urlsplit(base_url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or
            parsed.password or parsed.query or parsed.fragment or parsed.path):
        raise ValueError("FMP_BASE_URL 必须是无路径、无凭据的 HTTPS 地址。")
    auth_mode = os.environ.get("FMP_AUTH_MODE", "query" if parsed.hostname == "financialmodelingprep.com" else "proxy")
    if auth_mode not in {"query", "proxy"}:
        raise ValueError("FMP_AUTH_MODE 必须为 query 或 proxy。")
    ca_bundle = os.environ.get("FMP_CA_BUNDLE") or None
    if ca_bundle and not Path(ca_bundle).is_file():
        raise ValueError("FMP_CA_BUNDLE 指定的证书文件不存在。")
    return Settings(api_key=os.environ.get("FMP_API_KEY", "").strip(),
                    base_url=base_url, auth_mode=auth_mode, ca_bundle=ca_bundle)

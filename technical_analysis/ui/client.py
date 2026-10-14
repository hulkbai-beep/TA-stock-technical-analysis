"""One market-data client per server process, shared by every page."""
import streamlit as st

from config import load_settings
from data import fmp_client
from models import DataError


@st.cache_resource
def get_client() -> fmp_client.FMPClient:
    # Configuration is read internally, so credential strings are not cache keys.
    # Looked up on the module so tests can substitute an offline adapter.
    return fmp_client.FMPClient(load_settings())


def require_client() -> fmp_client.FMPClient:
    """Return the shared client, or raise a user-safe error when no key is set."""
    try:
        settings = load_settings()
    except ValueError as error:
        raise DataError(str(error), "invalid_configuration") from None
    if not settings.api_key or settings.api_key == "your_fmp_api_key":
        raise DataError("尚未配置 FMP_API_KEY。请在本项目的 .env 文件中设置密钥后重启应用。", "missing_key")
    return get_client()

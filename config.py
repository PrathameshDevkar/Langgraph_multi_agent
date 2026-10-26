from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[0]

class Settings(BaseSettings):
    gemini_api_key: SecretStr
    gemini_model: str

    RESEARCHER_MAX_STEPS: int = 8
    RESEARCHER_MAX_TOOL_CALLS: int = 6
    RESEARCHER_TOKEN_BUDGET: int = 60000

    COPYWRITER_MAX_STEPS: int= 6
    COPYWRITER_MAX_TOOL_CALLS: int= 5
    COPYWRITER_TOKEN_BUDGET: int = 60000

    MAX_AGENT_HANDOFFS: int = 6

    TOTAL_TOKEN_BUDGET: int = 150000

    model_config = SettingsConfigDict(
        env_file = f"{PROJECT_ROOT_PATH}/.env",
        env_file_encoding = "utf-8",
        extra = "ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
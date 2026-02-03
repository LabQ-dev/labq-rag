"""설정 로더 모듈

YAML 설정 파일(config.yaml, secrets.yaml)을 로드하고
환경변수 치환을 지원합니다.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


def _substitute_env_vars(content: str) -> str:
    """${VAR:-default} 패턴을 환경변수 값으로 치환"""

    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default_value = match.group(3) if match.group(3) else ""
        return os.getenv(var_name, default_value)

    return re.sub(r"\$\{(\w+)(:-([^}]*))?\}", replacer, content)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """YAML 파일 로드 (환경변수 치환 지원)

    Args:
        path: YAML 파일 경로

    Returns:
        파싱된 설정 딕셔너리

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
    """
    with open(path, encoding="utf-8") as f:
        content = f.read()

    substituted = _substitute_env_vars(content)
    return yaml.safe_load(substituted) or {}


class EmbeddingConfig(BaseModel):
    """임베딩 모델 설정"""

    model_name: str = "BAAI/bge-m3"
    device: str = "cpu"
    normalize: bool = True


class SplitterConfig(BaseModel):
    """텍스트 분할 설정"""

    chunk_size: int = 1000
    chunk_overlap: int = 200


class RetrieverConfig(BaseModel):
    """검색 설정"""

    top_k: int = 5
    search_type: str = "similarity"


class QdrantConfig(BaseModel):
    """Qdrant 벡터 DB 설정"""

    collection_name: str = "labq_docs"
    host: str = "localhost"
    port: int = 6333


class LLMConfig(BaseModel):
    """LLM 설정"""

    provider: str = "google"  # openai, google, anthropic
    temperature: float = 0.0

    # 프로바이더별 모델 (가성비 기준, 2025-2026)
    openai_model: str = "gpt-5-mini"
    google_model: str = "gemini-3.0-flash"
    anthropic_model: str = "claude-haiku-4-5"


class AppConfig(BaseModel):
    """애플리케이션 전체 설정"""

    embedding: EmbeddingConfig = EmbeddingConfig()
    splitter: SplitterConfig = SplitterConfig()
    retriever: RetrieverConfig = RetrieverConfig()
    qdrant: QdrantConfig = QdrantConfig()
    llm: LLMConfig = LLMConfig()


class SecretsConfig(BaseModel):
    """비밀 정보 설정"""

    openai_api_key: str = ""
    google_api_key: str = ""
    anthropic_api_key: str = ""
    notion_api_key: str = ""


class Settings:
    """통합 설정 관리자"""

    def __init__(
        self,
        config_path: str | Path = "configs/config.yaml",
        secrets_path: str | Path = "configs/secrets.yaml",
    ):
        self._config_path = Path(config_path)
        self._secrets_path = Path(secrets_path)
        self._load()

    def _load(self) -> None:
        """설정 파일 로드"""
        # config.yaml 로드
        if self._config_path.exists():
            config_data = load_yaml(self._config_path)
            self.config = AppConfig(**config_data)
        else:
            self.config = AppConfig()

        # secrets.yaml 로드
        if self._secrets_path.exists():
            secrets_data = load_yaml(self._secrets_path)
            self.secrets = SecretsConfig(
                openai_api_key=secrets_data.get("openai", {}).get("api_key", ""),
                google_api_key=secrets_data.get("google", {}).get("api_key", ""),
                anthropic_api_key=secrets_data.get("anthropic", {}).get("api_key", ""),
                notion_api_key=secrets_data.get("notion", {}).get("api_key", ""),
            )
        else:
            self.secrets = SecretsConfig()

    def reload(self) -> None:
        """설정 다시 로드"""
        self._load()


@lru_cache
def get_settings() -> Settings:
    """싱글톤 Settings 인스턴스 반환"""
    return Settings()

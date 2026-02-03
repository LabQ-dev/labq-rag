"""로깅 설정 모듈

QueueHandler와 RotatingFileHandler를 사용한 스레드 안전 로깅
"""

from __future__ import annotations

import logging
import logging.config
import logging.handlers
import queue
from pathlib import Path

import yaml

# 큐
log_queue: queue.Queue = queue.Queue(-1)


def setup_logging() -> None:
    """YAML 설정 파일에서 로깅 초기화"""
    config_path = Path(__file__).parent.parent / "configs" / "logging.yaml"

    if not config_path.exists():
        # 기본 설정
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s|%(module)s|L%(lineno)d] %(message)s",
        )
        return

    # 로그 디렉토리 생성
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # YAML 로드
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # QueueListener 설정
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(logging.WARNING)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s|%(module)s|L%(lineno)d] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    stderr_handler.setFormatter(formatter)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "labq_rag.log",
        maxBytes=10485760,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # QueueListener
    listener = logging.handlers.QueueListener(
        log_queue,
        stderr_handler,
        file_handler,
        respect_handler_level=True,
    )
    listener.start()

    # 로깅 설정 로드
    logging.config.dictConfig(config)

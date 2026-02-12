"""Retriever 전략별 실험 자동화 스크립트

각 retriever 전략(similarity, mmr, threshold, hybrid, rerank)별로
실험을 실행하고 결과를 JSON 파일로 저장합니다.

Usage:
    # 모든 전략 실행
    uv run python scripts/run_retriever_experiments.py

    # 특정 전략만 실행
    uv run python scripts/run_retriever_experiments.py --strategies similarity mmr

    # API URL 지정
    uv run python scripts/run_retriever_experiments.py --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 기본 경로
CONFIG_PATH = Path("configs/config.yaml")
CONFIG_BACKUP_PATH = Path("configs/config.yaml.backup")
EVAL_DATASET_PATH = Path("scripts/eval_dataset.json")
RESULTS_DIR = Path("scripts/results")

# 지원하는 retriever 전략
SUPPORTED_STRATEGIES = [
    "similarity",
    "mmr",
    "threshold",
    "hybrid",
    "rerank",
]

# API 타임아웃
API_TIMEOUT = 90.0
HEALTH_CHECK_TIMEOUT = 10.0


def backup_config() -> None:
    """config.yaml 백업"""
    if CONFIG_PATH.exists():
        shutil.copy2(CONFIG_PATH, CONFIG_BACKUP_PATH)
        logger.info(f"설정 파일 백업: {CONFIG_BACKUP_PATH}")


def restore_config() -> None:
    """config.yaml 복원"""
    if CONFIG_BACKUP_PATH.exists():
        shutil.copy2(CONFIG_BACKUP_PATH, CONFIG_PATH)
        CONFIG_BACKUP_PATH.unlink()
        logger.info("설정 파일 복원 완료")


def update_config_search_type(search_type: str) -> None:
    """config.yaml의 retriever.search_type 업데이트

    Args:
        search_type: 검색 전략 (similarity, mmr, threshold, hybrid, rerank)
    """
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["retriever"]["search_type"] = search_type

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    logger.info(f"설정 업데이트: retriever.search_type = {search_type}")


def check_api_health(api_url: str) -> bool:
    """API 헬스체크

    Args:
        api_url: API 기본 URL

    Returns:
        API가 정상이면 True
    """
    try:
        with httpx.Client(timeout=HEALTH_CHECK_TIMEOUT) as client:
            resp = client.get(f"{api_url}/health")
            resp.raise_for_status()
            logger.info(f"API 헬스체크 성공: {api_url}")
            return True
    except Exception as e:
        logger.error(f"API 헬스체크 실패: {e}")
        return False


def run_experiment_for_strategy(
    strategy: str,
    api_url: str,
) -> None:
    """특정 전략에 대한 실험 실행

    Args:
        strategy: retriever 전략
        api_url: API 기본 URL
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"실험 시작: retriever strategy = {strategy}")
    logger.info(f"{'='*60}")

    # 1. 설정 파일 업데이트
    update_config_search_type(strategy)

    # 2. API 헬스체크 (설정 변경 후 서버 재시작 필요할 수 있음)
    if not check_api_health(api_url):
        logger.warning(
            f"API 헬스체크 실패. 서버가 실행 중인지 확인하세요. "
            f"설정 변경 후 서버 재시작이 필요할 수 있습니다."
        )
        logger.warning("계속 진행합니다...")

    # 3. 실험 실행 (run_experiment.py 사용)
    variant = f"retriever_{strategy}"
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/run_experiment.py",
        "--variant",
        variant,
        "--api-url",
        api_url,
    ]

    logger.info(f"실험 명령어 실행: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("실험 완료")
        if result.stdout:
            logger.debug(f"출력: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"실험 실행 실패: {e}")
        logger.error(f"에러 출력: {e.stderr}")
        raise

    # 4. 잠시 대기 (다음 실험 전)
    time.sleep(2)


def run_all_experiments(
    strategies: list[str],
    api_url: str,
) -> None:
    """모든 전략에 대한 실험 실행

    Args:
        strategies: 실행할 retriever 전략 리스트
        api_url: API 기본 URL
    """
    # 설정 파일 백업
    backup_config()

    try:
        for strategy in strategies:
            if strategy not in SUPPORTED_STRATEGIES:
                logger.warning(f"지원하지 않는 전략: {strategy}. 건너뜁니다.")
                continue

            try:
                run_experiment_for_strategy(strategy, api_url)
            except Exception as e:
                logger.error(f"전략 {strategy} 실험 실패: {e}")
                logger.error("다음 전략으로 계속 진행합니다...")
                continue

        logger.info(f"\n{'='*60}")
        logger.info("모든 실험 완료!")
        logger.info(f"{'='*60}")

    finally:
        # 설정 파일 복원
        restore_config()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retriever 전략별 실험 자동화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 모든 전략 실행
  uv run python scripts/run_retriever_experiments.py

  # 특정 전략만 실행
  uv run python scripts/run_retriever_experiments.py --strategies similarity mmr

  # API URL 지정
  uv run python scripts/run_retriever_experiments.py --api-url http://localhost:8000
        """,
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=SUPPORTED_STRATEGIES,
        choices=SUPPORTED_STRATEGIES,
        help=f"실행할 retriever 전략 (기본값: 모든 전략)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API URL (기본값: http://localhost:8000)",
    )
    args = parser.parse_args()

    # 전략 중복 제거 및 정렬
    strategies = sorted(list(set(args.strategies)))

    logger.info(f"실행할 전략: {strategies}")
    logger.info(f"API URL: {args.api_url}")

    # API 헬스체크
    if not check_api_health(args.api_url):
        logger.error("API가 응답하지 않습니다. 서버가 실행 중인지 확인하세요.")
        logger.error("서버 실행: uv run uvicorn src.main:app --reload")
        return

    # 실험 실행
    run_all_experiments(strategies, args.api_url)


if __name__ == "__main__":
    main()




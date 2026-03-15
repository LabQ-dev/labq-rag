"""RAG 비교 실험 자동화 스크립트

실험 변형(variant)별로 평가 데이터셋의 질문들을 실행하고
결과를 JSON 파일로 저장합니다.

Usage:
    uv run python scripts/run_experiment.py --variant baseline
    uv run python scripts/run_experiment.py --variant citation
    uv run python scripts/run_experiment.py --variant combined --api-url http://100.109.146.90:8000
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 기본 경로
EVAL_DATASET_PATH = Path("scripts/eval_dataset.json")
RESULTS_DIR = Path("scripts/results")
CONFIG_PATH = Path("configs/config.yaml")

# API 타임아웃 (LLM 응답 대기)
API_TIMEOUT = 90.0


def get_git_commit() -> str | None:
    """현재 git commit hash (short) 반환

    Returns:
        7자리 commit hash 또는 None
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_config_snapshot(path: Path) -> dict | None:
    """config.yaml을 읽어 스냅샷으로 반환

    Args:
        path: config.yaml 경로

    Returns:
        설정 딕셔너리 또는 None (파일 없을 때)
    """
    if not path.exists():
        logger.warning(f"설정 파일을 찾을 수 없습니다: {path}")
        return None

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_eval_dataset(path: Path) -> dict:
    """평가 데이터셋 로드

    Args:
        path: eval_dataset.json 경로

    Returns:
        평가 데이터셋 딕셔너리

    Raises:
        FileNotFoundError: 데이터셋 파일이 없을 때
    """
    if not path.exists():
        raise FileNotFoundError(f"평가 데이터셋을 찾을 수 없습니다: {path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def calculate_keyword_score(answer: str, expected_keywords: list[str]) -> float:
    """키워드 매칭 점수 계산

    Args:
        answer: LLM 응답
        expected_keywords: 예상 키워드 리스트

    Returns:
        0.0 ~ 1.0 사이의 점수
    """
    if not expected_keywords:
        return 0.0

    hits = sum(1 for kw in expected_keywords if kw in answer)
    return round(hits / len(expected_keywords), 2)


def run_single_query(
    client: httpx.Client,
    api_url: str,
    question: str,
) -> tuple[dict, float]:
    """단일 질문 실행

    Args:
        client: httpx 클라이언트
        api_url: API 기본 URL
        question: 질문 문자열

    Returns:
        (응답 딕셔너리, 응답 시간 ms)
    """
    start = time.perf_counter()
    resp = client.post(
        f"{api_url}/query",
        json={"question": question},
        timeout=API_TIMEOUT,
    )
    elapsed_ms = round((time.perf_counter() - start) * 1000)

    resp.raise_for_status()
    return resp.json(), elapsed_ms


def run_experiment(
    variant: str,
    api_url: str,
    eval_dataset: dict,
) -> dict:
    """실험 실행

    Args:
        variant: 실험 변형 이름
        api_url: API 기본 URL
        eval_dataset: 평가 데이터셋

    Returns:
        실험 결과 딕셔너리
    """
    queries = eval_dataset["queries"]
    results = []
    total_elapsed = 0

    logger.info(f"실험 시작: variant={variant}, queries={len(queries)}")

    with httpx.Client() as client:
        # 헬스체크
        health = client.get(f"{api_url}/health", timeout=10)
        health.raise_for_status()
        logger.info(f"API 연결 확인: {health.json()}")

        for query in queries:
            query_id = query["id"]
            question = query["question"]
            expected_keywords = query.get("expected_keywords", [])

            logger.info(f"  [{query_id}] {question[:50]}...")

            try:
                data, elapsed_ms = run_single_query(client, api_url, question)
                answer = data.get("answer", "")
                keyword_score = calculate_keyword_score(answer, expected_keywords)
                error = None
            except Exception as e:
                logger.error(f"  [{query_id}] 실패: {e}")
                answer = ""
                elapsed_ms = 0
                keyword_score = 0.0
                error = str(e)

            total_elapsed += elapsed_ms

            results.append(
                {
                    "id": query_id,
                    "question": question,
                    "difficulty": query.get("difficulty", "unknown"),
                    "answer": answer,
                    "elapsed_ms": elapsed_ms,
                    "answer_length": len(answer),
                    "keyword_score": keyword_score,
                    "expected_keywords": expected_keywords,
                    "matched_keywords": [kw for kw in expected_keywords if kw in answer],
                    "error": error,
                }
            )

            logger.info(
                f"  [{query_id}] {elapsed_ms}ms, "
                f"keyword_score={keyword_score}, "
                f"length={len(answer)}"
            )

    # 요약 통계
    successful = [r for r in results if r["error"] is None]
    summary = {
        "total_queries": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "avg_elapsed_ms": (
            round(sum(r["elapsed_ms"] for r in successful) / len(successful)) if successful else 0
        ),
        "avg_keyword_score": (
            round(sum(r["keyword_score"] for r in successful) / len(successful), 2)
            if successful
            else 0.0
        ),
        "avg_answer_length": (
            round(sum(r["answer_length"] for r in successful) / len(successful))
            if successful
            else 0
        ),
        "total_elapsed_ms": total_elapsed,
    }

    logger.info(
        f"실험 완료: {summary['successful']}/{summary['total_queries']} 성공, "
        f"avg={summary['avg_elapsed_ms']}ms, "
        f"keyword_score={summary['avg_keyword_score']}"
    )

    # 재현성을 위한 config 스냅샷 및 git 정보
    config_snapshot = load_config_snapshot(CONFIG_PATH)
    git_commit = get_git_commit()
    now = datetime.now(UTC)

    return {
        "experiment": {
            "variant": variant,
            "date": now.strftime("%Y-%m-%d"),
            "timestamp": now.isoformat(),
            "api_url": api_url,
            "git_commit": git_commit,
            "config": config_snapshot,
        },
        "summary": summary,
        "results": results,
    }


def save_results(experiment: dict, output_dir: Path) -> Path:
    """실험 결과 저장

    Args:
        experiment: 실험 결과 딕셔너리
        output_dir: 저장 디렉토리

    Returns:
        저장된 파일 경로
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    date = experiment["experiment"]["date"]
    variant = experiment["experiment"]["variant"]
    git_commit = experiment["experiment"].get("git_commit")

    # git commit을 포함하여 브랜치 간 결과 충돌 방지
    if git_commit:
        base_name = f"{date}_{variant}_{git_commit}"
    else:
        base_name = f"{date}_{variant}"

    # 같은 커밋에서 재실행 시 기존 결과 보존 (시각 suffix 추가)
    output_path = output_dir / f"{base_name}.json"
    if output_path.exists():
        time_suffix = datetime.now(UTC).strftime("%H%M%S")
        output_path = output_dir / f"{base_name}_{time_suffix}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment, f, ensure_ascii=False, indent=2)

    logger.info(f"결과 저장: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 비교 실험 자동화")
    parser.add_argument(
        "--variant",
        required=True,
        help="실험 변형 이름 (예: baseline, citation, reordering, structured, combined)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API URL (기본값: http://localhost:8000)",
    )
    parser.add_argument(
        "--eval-dataset",
        default=str(EVAL_DATASET_PATH),
        help=f"평가 데이터셋 경로 (기본값: {EVAL_DATASET_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR),
        help=f"결과 저장 디렉토리 (기본값: {RESULTS_DIR})",
    )
    args = parser.parse_args()

    eval_dataset = load_eval_dataset(Path(args.eval_dataset))
    experiment = run_experiment(args.variant, args.api_url, eval_dataset)
    save_results(experiment, Path(args.output_dir))


if __name__ == "__main__":
    main()

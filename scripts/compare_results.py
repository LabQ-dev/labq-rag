"""실험 결과 비교 스크립트

저장된 실험 결과 JSON 파일들을 비교하여
마크다운 테이블로 출력합니다.

Usage:
    # 디렉토리 내 모든 결과 비교
    uv run python scripts/compare_results.py

    # 특정 파일들만 비교
    uv run python scripts/compare_results.py --files results/a.json results/b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path("scripts/results")


def load_results(results_dir: Path) -> list[dict]:
    """결과 디렉토리에서 모든 JSON 파일 로드

    Args:
        results_dir: 결과 디렉토리 경로

    Returns:
        실험 결과 리스트 (날짜순 정렬)
    """
    results = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            data["_source_file"] = path.name
            results.append(data)
    return results


def load_result_files(file_paths: list[str]) -> list[dict]:
    """지정된 파일들만 로드

    Args:
        file_paths: JSON 파일 경로 리스트

    Returns:
        실험 결과 리스트
    """
    results = []
    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            print(f"  경고: 파일 없음 - {path}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            data["_source_file"] = path.name
            results.append(data)
    return results


def get_label(r: dict) -> str:
    """비교 테이블에서 사용할 라벨 생성

    Args:
        r: 실험 결과 딕셔너리

    Returns:
        "variant(commit)" 형태의 라벨
    """
    exp = r["experiment"]
    variant = exp["variant"]
    commit = exp.get("git_commit", "")
    return f"{variant}({commit})" if commit else variant


def print_experiment_info(results: list[dict]) -> None:
    """각 실험의 메타 정보 출력"""
    print("\n## 실험 메타 정보\n")
    print("| File | Variant | Commit | Date " "| LLM Provider | LLM Model | Top-K |")
    print("|------|---------|--------|------|-------------|-----------|-------|")

    for r in results:
        exp = r["experiment"]
        config = exp.get("config") or {}
        llm = config.get("llm", {})
        retriever = config.get("retriever", {})

        provider = llm.get("provider", "-")
        model = llm.get("model", "(default)")
        top_k = retriever.get("top_k", "-")

        print(
            f"| {r.get('_source_file', '-'):<30} "
            f"| {exp['variant']:<12} "
            f"| {exp.get('git_commit', '-'):<10} "
            f"| {exp.get('date', '-')} "
            f"| {provider:<13} "
            f"| {model:<11} "
            f"| {top_k:<5} |"
        )


def print_summary_table(results: list[dict]) -> None:
    """요약 비교 테이블 출력"""
    labels = [get_label(r) for r in results]

    print("\n## 실험 요약 비교\n")
    print("| Variant | Avg Time (ms) | Avg Keyword Score | Avg Length | Success |")
    print("|---------|---------------|-------------------|------------|---------|")

    for label, r in zip(labels, results):
        s = r["summary"]
        print(
            f"| {label:<20} "
            f"| {s['avg_elapsed_ms']:>13} "
            f"| {s['avg_keyword_score']:>17} "
            f"| {s['avg_answer_length']:>10} "
            f"| {s['successful']}/{s['total_queries']:<5} |"
        )


def print_detail_table(results: list[dict]) -> None:
    """질문별 상세 비교 테이블 출력"""
    if not results:
        return

    query_ids = [r["id"] for r in results[0]["results"]]
    labels = [get_label(r) for r in results]

    print("\n## 질문별 키워드 점수 비교\n")

    header = "| ID | Difficulty | " + " | ".join(labels) + " |"
    separator = "|" + "|".join(["----"] * (len(labels) + 2)) + "|"
    print(header)
    print(separator)

    for qid in query_ids:
        row_data = []
        difficulty = ""
        for r in results:
            for q in r["results"]:
                if q["id"] == qid:
                    row_data.append(q["keyword_score"])
                    difficulty = q.get("difficulty", "")
                    break

        scores = " | ".join(f"{s:.2f}" for s in row_data)
        print(f"| {qid:<2} | {difficulty:<10} | {scores} |")

    print("\n## 질문별 응답 시간 비교 (ms)\n")

    header = "| ID | Difficulty | " + " | ".join(labels) + " |"
    print(header)
    print(separator)

    for qid in query_ids:
        row_data = []
        difficulty = ""
        for r in results:
            for q in r["results"]:
                if q["id"] == qid:
                    row_data.append(q["elapsed_ms"])
                    difficulty = q.get("difficulty", "")
                    break

        times = " | ".join(f"{t:>4}" for t in row_data)
        print(f"| {qid:<2} | {difficulty:<10} | {times} |")


def print_baseline_diff(results: list[dict]) -> None:
    """첫 번째 결과를 기준으로 나머지와 비교

    --files로 지정 시 첫 번째 파일이 기준이 됩니다.
    디렉토리 전체 비교 시 첫 번째 baseline이 기준.
    """
    if len(results) < 2:
        return

    # 첫 번째 결과를 기준으로 사용
    base = results[0]
    base_label = get_label(base)

    print(f"\n## 기준 대비 변화 (기준: {base_label})\n")
    print("| Variant | Time Diff | Keyword Diff | Length Diff |")
    print("|---------|-----------|--------------|------------|")

    bs = base["summary"]

    for r in results[1:]:
        s = r["summary"]
        label = get_label(r)
        time_diff = s["avg_elapsed_ms"] - bs["avg_elapsed_ms"]
        kw_diff = s["avg_keyword_score"] - bs["avg_keyword_score"]
        len_diff = s["avg_answer_length"] - bs["avg_answer_length"]

        time_sign = "+" if time_diff >= 0 else ""
        kw_sign = "+" if kw_diff >= 0 else ""
        len_sign = "+" if len_diff >= 0 else ""

        print(
            f"| {label:<30} "
            f"| {time_sign}{time_diff:>7}ms "
            f"| {kw_sign}{kw_diff:>10.2f} "
            f"| {len_sign}{len_diff:>8} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="실험 결과 비교")
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        help=f"결과 디렉토리 경로 (기본값: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="비교할 결과 파일들 (예: results/a.json results/b.json)",
    )
    args = parser.parse_args()

    if args.files:
        results = load_result_files(args.files)
    else:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"결과 디렉토리를 찾을 수 없습니다: {results_dir}")
            return
        results = load_results(results_dir)

    if not results:
        print("결과 파일이 없습니다.")
        return

    print(f"# RAG 실험 비교 ({len(results)}개 결과)\n")

    print_experiment_info(results)
    print_summary_table(results)
    print_baseline_diff(results)
    print_detail_table(results)


if __name__ == "__main__":
    main()

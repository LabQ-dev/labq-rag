"""실험 결과 시각화 스크립트

저장된 실험 결과 JSON 파일들을 비교하여
시각화 그래프를 생성하고 이미지로 저장합니다.

Usage:
    # 디렉토리 내 모든 결과 시각화
    uv run python scripts/plot_results.py

    # 특정 파일들만 시각화
    uv run python scripts/plot_results.py --files results/a.json results/b.json

    # 출력 디렉토리 지정
    uv run python scripts/plot_results.py --output-dir plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
except ImportError:
    print("matplotlib와 seaborn이 필요합니다.")
    print("설치: uv add matplotlib seaborn")
    raise

# compare_results.py의 함수 재사용
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


def plot_summary_comparison(results: list[dict], output_path: Path) -> None:
    """요약 비교 바 차트 생성

    Args:
        results: 실험 결과 리스트
        output_path: 저장 경로
    """
    labels = [get_label(r) for r in results]
    summaries = [r["summary"] for r in results]

    # 2x2 서브플롯
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Experiment Summary Comparison", fontsize=16, fontweight="bold")

    # 1. 평균 응답 시간
    ax1 = axes[0, 0]
    times = [s["avg_elapsed_ms"] for s in summaries]
    bars1 = ax1.bar(range(len(labels)), times, color="steelblue", alpha=0.7)
    ax1.set_xlabel("Variant")
    ax1.set_ylabel("Average Response Time (ms)")
    ax1.set_title("Average Response Time")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.grid(axis="y", alpha=0.3)
    # 값 표시
    for i, (bar, val) in enumerate(zip(bars1, times)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 2. 평균 키워드 점수
    ax2 = axes[0, 1]
    scores = [s["avg_keyword_score"] for s in summaries]
    bars2 = ax2.bar(range(len(labels)), scores, color="coral", alpha=0.7)
    ax2.set_xlabel("Variant")
    ax2.set_ylabel("Average Keyword Score")
    ax2.set_title("Average Keyword Score")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis="y", alpha=0.3)
    # 값 표시
    for i, (bar, val) in enumerate(zip(bars2, scores)):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 3. 평균 응답 길이
    ax3 = axes[1, 0]
    lengths = [s["avg_answer_length"] for s in summaries]
    bars3 = ax3.bar(range(len(labels)), lengths, color="mediumseagreen", alpha=0.7)
    ax3.set_xlabel("Variant")
    ax3.set_ylabel("Average Answer Length (chars)")
    ax3.set_title("Average Answer Length")
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=45, ha="right")
    ax3.grid(axis="y", alpha=0.3)
    # 값 표시
    for i, (bar, val) in enumerate(zip(bars3, lengths)):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 4. 성공률
    ax4 = axes[1, 1]
    success_rates = [
        s["successful"] / s["total_queries"] * 100 for s in summaries
    ]
    bars4 = ax4.bar(range(len(labels)), success_rates, color="gold", alpha=0.7)
    ax4.set_xlabel("Variant")
    ax4.set_ylabel("Success Rate (%)")
    ax4.set_title("Success Rate")
    ax4.set_xticks(range(len(labels)))
    ax4.set_xticklabels(labels, rotation=45, ha="right")
    ax4.set_ylim(0, 110)
    ax4.grid(axis="y", alpha=0.3)
    # 값 표시
    for i, (bar, val) in enumerate(zip(bars4, success_rates)):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"저장: {output_path}")


def plot_keyword_scores_by_query(results: list[dict], output_path: Path) -> None:
    """질문별 키워드 점수 비교 라인 차트 생성

    Args:
        results: 실험 결과 리스트
        output_path: 저장 경로
    """
    if not results:
        return

    labels = [get_label(r) for r in results]
    query_ids = sorted([r["id"] for r in results[0]["results"]])

    # 각 variant별 질문별 점수 추출
    data = {label: [] for label in labels}
    for r, label in zip(results, labels):
        scores_by_id = {q["id"]: q["keyword_score"] for q in r["results"]}
        data[label] = [scores_by_id.get(qid, 0.0) for qid in query_ids]

    # 라인 차트
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, scores in data.items():
        ax.plot(query_ids, scores, marker="o", label=label, linewidth=2, markersize=6)

    ax.set_xlabel("Query ID", fontsize=11)
    ax.set_ylabel("Keyword Score", fontsize=11)
    ax.set_title("Keyword Score by Query", fontsize=14, fontweight="bold")
    ax.set_xticks(query_ids)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"저장: {output_path}")


def plot_elapsed_time_by_query(results: list[dict], output_path: Path) -> None:
    """질문별 응답 시간 비교 라인 차트 생성

    Args:
        results: 실험 결과 리스트
        output_path: 저장 경로
    """
    if not results:
        return

    labels = [get_label(r) for r in results]
    query_ids = sorted([r["id"] for r in results[0]["results"]])

    # 각 variant별 질문별 시간 추출
    data = {label: [] for label in labels}
    for r, label in zip(results, labels):
        times_by_id = {q["id"]: q["elapsed_ms"] for q in r["results"]}
        data[label] = [times_by_id.get(qid, 0) for qid in query_ids]

    # 라인 차트
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, times in data.items():
        ax.plot(query_ids, times, marker="s", label=label, linewidth=2, markersize=6)

    ax.set_xlabel("Query ID", fontsize=11)
    ax.set_ylabel("Response Time (ms)", fontsize=11)
    ax.set_title("Response Time by Query", fontsize=14, fontweight="bold")
    ax.set_xticks(query_ids)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"저장: {output_path}")


def plot_baseline_diff(results: list[dict], output_path: Path) -> None:
    """기준 대비 변화 바 차트 생성

    Args:
        results: 실험 결과 리스트
        output_path: 저장 경로
    """
    if len(results) < 2:
        return

    # 첫 번째 결과를 기준으로 사용
    base = results[0]
    base_label = get_label(base)
    bs = base["summary"]

    # 기준 대비 차이 계산
    variants = []
    time_diffs = []
    kw_diffs = []
    len_diffs = []

    for r in results[1:]:
        s = r["summary"]
        label = get_label(r)
        variants.append(label)
        time_diffs.append(s["avg_elapsed_ms"] - bs["avg_elapsed_ms"])
        kw_diffs.append(s["avg_keyword_score"] - bs["avg_keyword_score"])
        len_diffs.append(s["avg_answer_length"] - bs["avg_answer_length"])

    # 3개 서브플롯
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Difference from Baseline (Baseline: {base_label})", fontsize=14, fontweight="bold"
    )

    x = np.arange(len(variants))
    width = 0.6

    # 1. 응답 시간 차이
    ax1 = axes[0]
    colors1 = ["red" if d > 0 else "green" for d in time_diffs]
    bars1 = ax1.bar(x, time_diffs, width, color=colors1, alpha=0.7)
    ax1.set_xlabel("Variant")
    ax1.set_ylabel("Response Time Difference (ms)")
    ax1.set_title("Response Time Difference")
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, rotation=45, ha="right")
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax1.grid(axis="y", alpha=0.3)
    # 값 표시
    for bar, val in zip(bars1, time_diffs):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{val:+,.0f}",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontsize=9,
        )

    # 2. 키워드 점수 차이
    ax2 = axes[1]
    colors2 = ["green" if d > 0 else "red" for d in kw_diffs]
    bars2 = ax2.bar(x, kw_diffs, width, color=colors2, alpha=0.7)
    ax2.set_xlabel("Variant")
    ax2.set_ylabel("Keyword Score Difference")
    ax2.set_title("Keyword Score Difference")
    ax2.set_xticks(x)
    ax2.set_xticklabels(variants, rotation=45, ha="right")
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax2.grid(axis="y", alpha=0.3)
    # 값 표시
    for bar, val in zip(bars2, kw_diffs):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{val:+.2f}",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontsize=9,
        )

    # 3. 응답 길이 차이
    ax3 = axes[2]
    colors3 = ["green" if d > 0 else "red" for d in len_diffs]
    bars3 = ax3.bar(x, len_diffs, width, color=colors3, alpha=0.7)
    ax3.set_xlabel("Variant")
    ax3.set_ylabel("Answer Length Difference (chars)")
    ax3.set_title("Answer Length Difference")
    ax3.set_xticks(x)
    ax3.set_xticklabels(variants, rotation=45, ha="right")
    ax3.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax3.grid(axis="y", alpha=0.3)
    # 값 표시
    for bar, val in zip(bars3, len_diffs):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{val:+d}",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"저장: {output_path}")


def plot_heatmap_keyword_scores(results: list[dict], output_path: Path) -> None:
    """질문별 키워드 점수 히트맵 생성

    Args:
        results: 실험 결과 리스트
        output_path: 저장 경로
    """
    if not results:
        return

    labels = [get_label(r) for r in results]
    query_ids = sorted([r["id"] for r in results[0]["results"]])

    # 히트맵 데이터 준비
    heatmap_data = []
    for r in results:
        scores_by_id = {q["id"]: q["keyword_score"] for q in r["results"]}
        heatmap_data.append([scores_by_id.get(qid, 0.0) for qid in query_ids])

    # 히트맵 생성
    fig, ax = plt.subplots(figsize=(max(10, len(query_ids) * 0.8), max(6, len(labels) * 0.6)))
    im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    # 축 레이블 설정
    ax.set_xticks(range(len(query_ids)))
    ax.set_xticklabels([f"Q{qid}" for qid in query_ids])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    # 값 표시
    for i in range(len(labels)):
        for j in range(len(query_ids)):
            text = ax.text(
                j,
                i,
                f"{heatmap_data[i][j]:.2f}",
                ha="center",
                va="center",
                color="black" if heatmap_data[i][j] < 0.5 else "white",
                fontsize=8,
            )

    ax.set_xlabel("Query ID", fontsize=11)
    ax.set_ylabel("Variant", fontsize=11)
    ax.set_title("Keyword Score Heatmap by Query", fontsize=14, fontweight="bold")

    # 컬러바
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Keyword Score", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"저장: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="실험 결과 시각화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 디렉토리 내 모든 결과 시각화
  uv run python scripts/plot_results.py

  # 특정 파일들만 시각화
  uv run python scripts/plot_results.py --files results/a.json results/b.json

  # 출력 디렉토리 지정
  uv run python scripts/plot_results.py --output-dir plots
        """,
    )
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
    parser.add_argument(
        "--output-dir",
        default="scripts/plots",
        help="출력 디렉토리 경로 (기본값: scripts/plots)",
    )
    args = parser.parse_args()

    # 결과 로드
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

    print(f"시각화 시작: {len(results)}개 결과")

    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # matplotlib 스타일 설정
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        try:
            plt.style.use("seaborn-darkgrid")
        except OSError:
            plt.style.use("default")
    sns.set_palette("husl")

    # 그래프 생성
    plot_summary_comparison(results, output_dir / "summary_comparison.png")
    plot_keyword_scores_by_query(results, output_dir / "keyword_scores_by_query.png")
    plot_elapsed_time_by_query(results, output_dir / "elapsed_time_by_query.png")
    plot_heatmap_keyword_scores(results, output_dir / "keyword_scores_heatmap.png")

    if len(results) >= 2:
        plot_baseline_diff(results, output_dir / "baseline_diff.png")

    print(f"\n모든 그래프 저장 완료: {output_dir}")


if __name__ == "__main__":
    main()


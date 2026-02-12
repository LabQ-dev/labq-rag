"""벡터 검색 모듈

Qdrant에서 유사한 문서 청크를 검색합니다.
다양한 검색 전략을 지원합니다: MMR, Score Threshold, Hybrid Search, Re-ranking, Metadata Filtering.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from langchain_qdrant import QdrantVectorStore

from src.config import get_settings
from src.indexer import get_embeddings, get_qdrant_client

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger()


def get_vectorstore() -> QdrantVectorStore:
    """Qdrant 벡터 스토어 인스턴스 반환"""
    settings = get_settings()
    embeddings = get_embeddings()
    client = get_qdrant_client()

    return QdrantVectorStore(
        client=client,
        collection_name=settings.config.qdrant.collection_name,
        embedding=embeddings,
    )


def get_retriever() -> VectorStoreRetriever:
    """검색기 인스턴스 반환

    Returns:
        설정된 top_k와 search_type을 사용하는 retriever
    """
    settings = get_settings()
    vectorstore = get_vectorstore()

    return vectorstore.as_retriever(
        search_type=settings.config.retriever.search_type,
        search_kwargs={"k": settings.config.retriever.top_k},
    )


def retrieve_advanced(
    query: str,
    strategy: str = "similarity",
    top_k: int | None = None,
    **kwargs,
) -> list[Document]:
    """고급 검색 전략을 사용한 통합 검색 함수

    다양한 검색 전략을 하나의 인터페이스로 사용할 수 있습니다.

    Args:
        query: 검색 쿼리
        strategy: 검색 전략
            - "similarity": 기본 유사도 검색
            - "mmr": MMR 검색 (kwargs: lambda_mult, fetch_k)
            - "threshold": Score Threshold 검색 (kwargs: score_threshold)
            - "hybrid": Hybrid Search (kwargs: vector_weight)
            - "rerank": Re-ranking 검색 (kwargs: rerank_top_n)
            - "metadata": Metadata Filtering (kwargs: metadata_filter)
        top_k: 반환할 문서 수
        **kwargs: 전략별 추가 파라미터
            - mmr: lambda_mult (float), fetch_k (int)
            - threshold: score_threshold (float)
            - hybrid: vector_weight (float)
            - rerank: rerank_top_n (int)
            - metadata: metadata_filter (dict)

    Returns:
        검색된 Document 리스트

    Examples:
        >>> # MMR 검색
        >>> docs = retrieve_advanced("질문", strategy="mmr", lambda_mult=0.5)
        >>>
        >>> # Score Threshold 검색
        >>> docs = retrieve_advanced("질문", strategy="threshold", score_threshold=0.8)
        >>>
        >>> # Hybrid Search
        >>> docs = retrieve_advanced("질문", strategy="hybrid", vector_weight=0.7)
        >>>
        >>> # Re-ranking
        >>> docs = retrieve_advanced("질문", strategy="rerank", rerank_top_n=10)
        >>>
        >>> # Metadata Filtering
        >>> docs = retrieve_advanced(
        ...     "질문",
        ...     strategy="metadata",
        ...     metadata_filter={"source": "document.pdf"}
        ... )
    """
    strategy = strategy.lower()

    if strategy == "similarity":
        return retrieve(query)

    elif strategy == "mmr":
        lambda_mult = kwargs.get("lambda_mult", 0.5)
        fetch_k = kwargs.get("fetch_k")
        return retrieve_mmr(query, top_k=top_k, fetch_k=fetch_k, lambda_mult=lambda_mult)

    elif strategy == "threshold":
        score_threshold = kwargs.get("score_threshold", 0.7)
        return retrieve_with_threshold(query, score_threshold=score_threshold, top_k=top_k)

    elif strategy == "hybrid":
        vector_weight = kwargs.get("vector_weight", 0.7)
        return retrieve_hybrid(query, top_k=top_k, vector_weight=vector_weight)

    elif strategy == "rerank":
        rerank_top_n = kwargs.get("rerank_top_n")
        return retrieve_with_rerank(query, top_k=top_k, rerank_top_n=rerank_top_n)

    elif strategy == "metadata":
        metadata_filter = kwargs.get("metadata_filter")
        if not metadata_filter:
            raise ValueError("metadata_filter는 필수입니다.")
        return retrieve_with_metadata_filter(
            query, metadata_filter=metadata_filter, top_k=top_k
        )

    else:
        raise ValueError(
            f"지원하지 않는 전략: {strategy}. "
            f"가능한 값: similarity, mmr, threshold, hybrid, rerank, metadata"
        )


def retrieve(query: str) -> list[Document]:
    """쿼리와 유사한 문서 검색

    Args:
        query: 검색 쿼리

    Returns:
        유사한 Document 리스트
    """
    logger.debug(f"검색 시작: {query[:50]}...")
    retriever = get_retriever()
    docs = retriever.invoke(query)
    logger.info(f"검색 완료: {len(docs)}개 문서 검색됨")
    return docs


def retrieve_with_scores(query: str) -> list[tuple[Document, float]]:
    """쿼리와 유사한 문서를 점수와 함께 검색

    Args:
        query: 검색 쿼리

    Returns:
        (Document, score) 튜플 리스트
    """
    logger.debug(f"점수와 함께 검색: {query[:50]}...")
    settings = get_settings()
    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search_with_score(
        query=query,
        k=settings.config.retriever.top_k,
    )
    logger.info(f"검색 완료: {len(results)}개 결과")
    return results


def retrieve_mmr(
    query: str,
    top_k: int | None = None,
    fetch_k: int | None = None,
    lambda_mult: float = 0.5,
) -> list[Document]:
    """MMR (Maximal Marginal Relevance) 검색

    관련성과 다양성을 균형있게 고려하여 검색합니다.
    lambda_mult: 0.0=다양성 우선, 1.0=관련성 우선, 0.5=균형

    Args:
        query: 검색 쿼리
        top_k: 반환할 문서 수 (None이면 설정값 사용)
        fetch_k: 다양성을 위해 가져올 후보 수 (None이면 top_k * 2)
        lambda_mult: 관련성/다양성 균형 조절 (0.0-1.0)

    Returns:
        검색된 Document 리스트
    """
    logger.debug(f"MMR 검색: {query[:50]}..., lambda_mult={lambda_mult}")
    settings = get_settings()
    vectorstore = get_vectorstore()

    if top_k is None:
        top_k = settings.config.retriever.top_k
    if fetch_k is None:
        fetch_k = top_k * 2

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        },
    )

    docs = retriever.invoke(query)
    logger.info(f"MMR 검색 완료: {len(docs)}개 문서 (lambda_mult={lambda_mult})")
    return docs


def retrieve_with_threshold(
    query: str,
    score_threshold: float = 0.7,
    top_k: int | None = None,
) -> list[Document]:
    """Score Threshold를 사용한 검색

    최소 유사도 점수 이상인 문서만 반환합니다.

    Args:
        query: 검색 쿼리
        score_threshold: 최소 유사도 점수 (0-1, cosine similarity)
        top_k: 최대 반환 문서 수 (None이면 설정값 사용)

    Returns:
        임계값을 통과한 Document 리스트
    """
    logger.debug(f"Threshold 검색: {query[:50]}..., threshold={score_threshold}")
    settings = get_settings()
    vectorstore = get_vectorstore()

    if top_k is None:
        top_k = settings.config.retriever.top_k

    # 더 많이 가져온 후 필터링
    # Qdrant는 cosine similarity를 사용하므로 1.0에 가까울수록 유사
    # threshold는 최소 점수이므로 더 많이 가져와서 필터링
    fetch_k = top_k * 3

    results = vectorstore.similarity_search_with_score(
        query=query,
        k=fetch_k,
    )

    # 점수 필터링 (cosine similarity는 1.0에 가까울수록 유사)
    filtered = [(doc, score) for doc, score in results if score >= score_threshold]

    # top_k만큼만 반환
    docs = [doc for doc, _ in filtered[:top_k]]
    logger.info(
        f"Threshold 검색 완료: {len(docs)}개 문서 "
        f"(threshold={score_threshold}, 후보 {len(results)}개 중 선택)"
    )
    return docs


def _simple_keyword_search(
    query: str, docs: list[Document], top_k: int
) -> list[Document]:
    """간단한 키워드 기반 검색 (Hybrid Search용)

    문서 내용에서 쿼리 키워드가 많이 포함된 문서를 우선 선택합니다.

    Args:
        query: 검색 쿼리
        docs: 검색 대상 문서 리스트
        top_k: 반환할 문서 수

    Returns:
        키워드 매칭 점수가 높은 Document 리스트
    """
    # 쿼리를 키워드로 분리
    keywords = set(re.findall(r"\w+", query.lower()))

    # 각 문서의 키워드 매칭 점수 계산
    scored_docs = []
    for doc in docs:
        content_lower = doc.page_content.lower()
        # 키워드가 포함된 횟수
        matches = sum(1 for keyword in keywords if keyword in content_lower)
        # 문서 길이로 정규화 (짧은 문서에 키워드가 많으면 높은 점수)
        score = matches / max(len(content_lower.split()), 1)
        scored_docs.append((doc, score))

    # 점수 순으로 정렬
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scored_docs[:top_k]]


def retrieve_hybrid(
    query: str,
    top_k: int | None = None,
    vector_weight: float = 0.7,
) -> list[Document]:
    """Hybrid Search (벡터 검색 + 키워드 검색)

    벡터 검색과 키워드 검색 결과를 결합합니다.

    Args:
        query: 검색 쿼리
        top_k: 반환할 문서 수 (None이면 설정값 사용)
        vector_weight: 벡터 검색 가중치 (0.0-1.0, 키워드 가중치 = 1 - vector_weight)

    Returns:
        하이브리드 검색 결과 Document 리스트
    """
    logger.debug(f"Hybrid 검색: {query[:50]}..., vector_weight={vector_weight}")
    settings = get_settings()
    vectorstore = get_vectorstore()

    if top_k is None:
        top_k = settings.config.retriever.top_k

    # 벡터 검색 (더 많이 가져옴)
    vector_results = vectorstore.similarity_search_with_score(
        query=query,
        k=top_k * 2,
    )
    vector_docs = [doc for doc, _ in vector_results]

    # 키워드 검색 (벡터 검색 결과에서 키워드 매칭)
    keyword_docs = _simple_keyword_search(query, vector_docs, top_k)

    # 결과 결합 (Reciprocal Rank Fusion 방식)
    # 각 문서에 순위 점수 부여
    doc_scores: dict[str, float] = {}

    # 벡터 검색 점수 (순위 기반)
    for rank, (doc, score) in enumerate(vector_results, 1):
        doc_id = id(doc)  # 간단한 식별자
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + vector_weight / rank

    # 키워드 검색 점수 (순위 기반)
    keyword_weight = 1.0 - vector_weight
    for rank, doc in enumerate(keyword_docs, 1):
        doc_id = id(doc)
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + keyword_weight / rank

    # 모든 문서를 점수 순으로 정렬
    all_docs = {id(doc): doc for doc in vector_docs}
    sorted_docs = sorted(
        doc_scores.items(), key=lambda x: x[1], reverse=True
    )[:top_k]

    result = [all_docs[doc_id] for doc_id, _ in sorted_docs]
    logger.info(f"Hybrid 검색 완료: {len(result)}개 문서")
    return result


def retrieve_with_rerank(
    query: str,
    top_k: int | None = None,
    rerank_top_n: int | None = None,
) -> list[Document]:
    """Re-ranking을 사용한 검색

    초기 검색 후 더 정교한 재정렬을 수행합니다.
    현재는 간단한 휴리스틱 기반 재정렬 (향후 LLM 기반 reranker로 확장 가능)

    Args:
        query: 검색 쿼리
        top_k: 최종 반환할 문서 수 (None이면 설정값 사용)
        rerank_top_n: 재정렬할 후보 수 (None이면 top_k * 2)

    Returns:
        재정렬된 Document 리스트
    """
    logger.debug(f"Re-ranking 검색: {query[:50]}...")
    settings = get_settings()
    vectorstore = get_vectorstore()

    if top_k is None:
        top_k = settings.config.retriever.top_k
    if rerank_top_n is None:
        rerank_top_n = top_k * 2

    # 1단계: 초기 검색 (더 많이 가져옴)
    initial_results = vectorstore.similarity_search_with_score(
        query=query,
        k=rerank_top_n,
    )

    # 2단계: 재정렬 (간단한 휴리스틱)
    # 쿼리 키워드와 문서 내용의 매칭도 추가 고려
    query_keywords = set(re.findall(r"\w+", query.lower()))

    reranked = []
    for doc, score in initial_results:
        # 벡터 유사도 점수
        vector_score = score

        # 키워드 매칭 보너스
        content_lower = doc.page_content.lower()
        keyword_matches = sum(1 for kw in query_keywords if kw in content_lower)
        keyword_bonus = keyword_matches / max(len(query_keywords), 1) * 0.1

        # 최종 점수 = 벡터 점수 + 키워드 보너스
        final_score = vector_score + keyword_bonus
        reranked.append((doc, final_score))

    # 재정렬된 점수로 정렬
    reranked.sort(key=lambda x: x[1], reverse=True)

    # top_k만 반환
    result = [doc for doc, _ in reranked[:top_k]]
    logger.info(
        f"Re-ranking 검색 완료: {len(result)}개 문서 "
        f"(후보 {rerank_top_n}개 중 재정렬)"
    )
    return result


def retrieve_with_metadata_filter(
    query: str,
    metadata_filter: dict[str, str | int | list],
    top_k: int | None = None,
) -> list[Document]:
    """Metadata Filtering을 사용한 검색

    특정 메타데이터 조건을 만족하는 문서만 검색합니다.

    Args:
        query: 검색 쿼리
        metadata_filter: 메타데이터 필터 딕셔너리
            예: {"source": "document.pdf"} 또는 {"page": 1}
        top_k: 반환할 문서 수 (None이면 설정값 사용)

    Returns:
        필터 조건을 만족하는 Document 리스트
    """
    logger.debug(
        f"Metadata 필터 검색: {query[:50]}..., filter={metadata_filter}"
    )
    settings = get_settings()
    vectorstore = get_vectorstore()

    if top_k is None:
        top_k = settings.config.retriever.top_k

    # Qdrant의 필터 형식으로 변환
    # Qdrant 필터는 복잡하므로, 여기서는 검색 후 필터링하는 방식 사용
    # (실제 프로덕션에서는 Qdrant의 필터 기능을 직접 사용하는 것이 효율적)

    # 더 많이 가져온 후 필터링
    results = vectorstore.similarity_search_with_score(
        query=query,
        k=top_k * 3,  # 필터링을 위해 더 많이 가져옴
    )

    # 메타데이터 필터링
    filtered = []
    for doc, score in results:
        match = True
        for key, value in metadata_filter.items():
            doc_value = doc.metadata.get(key)
            if isinstance(value, list):
                # 리스트인 경우: doc_value가 value에 포함되어야 함
                if doc_value not in value:
                    match = False
                    break
            else:
                # 단일 값인 경우: 정확히 일치해야 함
                if doc_value != value:
                    match = False
                    break

        if match:
            filtered.append((doc, score))

    # top_k만 반환
    result = [doc for doc, _ in filtered[:top_k]]
    logger.info(
        f"Metadata 필터 검색 완료: {len(result)}개 문서 "
        f"(필터: {metadata_filter}, 후보 {len(results)}개 중 선택)"
    )
    return result

"""벡터 검색 모듈

Qdrant에서 유사한 문서 청크를 검색합니다.
"""

from __future__ import annotations

import logging
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

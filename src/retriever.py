"""벡터 검색 모듈

Qdrant에서 유사한 문서 청크를 검색합니다.
MMR (Maximal Marginal Relevance) 검색을 사용합니다.
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
    """MMR 검색기 인스턴스 반환

    MMR (Maximal Marginal Relevance) 검색을 사용하여
    관련성과 다양성을 균형있게 고려한 문서를 검색합니다.

    Returns:
        MMR 검색을 사용하는 retriever
    """
    settings = get_settings()
    vectorstore = get_vectorstore()
    top_k = settings.config.retriever.top_k

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": top_k * 2,  # 다양성을 위해 더 많은 후보 검색
            "lambda_mult": 0.5,  # 관련성/다양성 균형 (0.5 = 균형)
        },
    )



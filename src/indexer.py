"""PDF 인덱싱 모듈

PDF 파일을 로드하고, 텍스트를 분할한 후,
임베딩을 생성하여 Qdrant에 저장합니다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import get_settings

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = logging.getLogger()


def get_embeddings() -> HuggingFaceEmbeddings:
    """임베딩 모델 인스턴스 반환"""
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.config.embedding.model_name,
        model_kwargs={"device": settings.config.embedding.device},
        encode_kwargs={"normalize_embeddings": settings.config.embedding.normalize},
    )


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """텍스트 분할기 인스턴스 반환"""
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.config.splitter.chunk_size,
        chunk_overlap=settings.config.splitter.chunk_overlap,
    )


def get_qdrant_client() -> QdrantClient:
    """Qdrant 클라이언트 반환"""
    settings = get_settings()
    return QdrantClient(
        host=settings.config.qdrant.host,
        port=settings.config.qdrant.port,
        timeout=settings.config.qdrant.timeout,
    )


def ensure_collection_exists(client: QdrantClient, embeddings: HuggingFaceEmbeddings) -> None:
    """Qdrant 컬렉션이 없으면 생성"""
    settings = get_settings()
    collection_name = settings.config.qdrant.collection_name

    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if collection_name not in collection_names:
        # 임베딩 차원 확인 (bge-m3는 1024 차원)
        sample_embedding = embeddings.embed_query("test")
        vector_size = len(sample_embedding)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )


def load_pdf(file_path: str | Path) -> list[Document]:
    """PDF 파일 로드

    Args:
        file_path: PDF 파일 경로

    Returns:
        Document 리스트 (페이지별)
    """
    logger.debug(f"PDF 로드 시작: {file_path}")
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    logger.info(f"PDF 로드 완료: {len(docs)}페이지")
    return docs


def index_documents(documents: list[Document]) -> int:
    """문서를 분할하고 Qdrant에 인덱싱

    Args:
        documents: LangChain Document 리스트

    Returns:
        인덱싱된 청크 수
    """
    settings = get_settings()

    # 텍스트 분할
    logger.debug("텍스트 분할 시작")
    text_splitter = get_text_splitter()
    chunks = text_splitter.split_documents(documents)
    logger.info(f"텍스트 분할 완료: {len(chunks)}개 청크")

    if not chunks:
        logger.warning("분할된 청크가 없음")
        return 0

    # 임베딩 모델
    logger.debug(f"임베딩 모델 로드: {settings.config.embedding.model_name}")
    embeddings = get_embeddings()

    # Qdrant 클라이언트 및 컬렉션 확인
    client = get_qdrant_client()
    ensure_collection_exists(client, embeddings)

    # 벡터 스토어에 저장
    logger.debug(f"벡터 인덱싱 시작: {len(chunks)}개 청크")
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=f"http://{settings.config.qdrant.host}:{settings.config.qdrant.port}",
        collection_name=settings.config.qdrant.collection_name,
    )
    logger.info(f"벡터 인덱싱 완료: {len(chunks)}개 청크")

    return len(chunks)


def index_pdf(file_path: str | Path) -> int:
    """PDF 파일을 인덱싱

    Args:
        file_path: PDF 파일 경로

    Returns:
        인덱싱된 청크 수
    """
    logger.debug(f"PDF 인덱싱 시작: {file_path}")
    documents = load_pdf(file_path)
    chunks_count = index_documents(documents)
    logger.info(f"PDF 인덱싱 완료: {chunks_count}개 청크")
    return chunks_count

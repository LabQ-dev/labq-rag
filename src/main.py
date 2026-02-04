"""FastAPI 애플리케이션

RAG 서비스의 REST API 엔드포인트를 제공합니다.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.config import get_settings
from src.generator import agenerate
from src.indexer import index_pdf
from src.logger import setup_logging
from src.schemas import HealthResponse, IndexResponse, QueryRequest, QueryResponse

# 로깅 초기화
setup_logging()
logger = logging.getLogger()

app = FastAPI(
    title="LabQ RAG Service",
    description="PDF 문서 기반 RAG 서비스 - 내부 지식을 활용한 질의응답",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# 데이터 디렉토리
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

logger.info("LabQ RAG Service 초기화됨")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="헬스체크",
    tags=["health"],
)
async def health_check() -> HealthResponse:
    """헬스체크 엔드포인트"""
    logger.info("헬스체크 요청")
    return HealthResponse(
        status="healthy",
        version="0.1.0",
    )


@app.post(
    "/index",
    response_model=IndexResponse,
    summary="PDF 인덱싱",
    description="PDF 파일을 업로드하여 벡터 데이터베이스에 인덱싱합니다. 텍스트를 청크로 분할하고 임베딩을 생성합니다.",
    tags=["indexing"],
)
async def index_document(
    file: Annotated[UploadFile, File(description="인덱싱할 PDF 파일")],
) -> IndexResponse:
    """PDF 파일 업로드 및 인덱싱

    Args:
        file: 업로드된 PDF 파일

    Returns:
        인덱싱 결과

    Raises:
        HTTPException: PDF 파일이 아니거나 인덱싱 실패 시
    """
    logger.info(f"PDF 인덱싱 시작: {file.filename}")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        logger.warning(f"잘못된 파일 형식: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail="PDF 파일만 업로드 가능합니다.",
        )

    # 파일 저장
    file_path = DATA_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.debug(f"파일 저장 완료: {file_path}")
    except Exception as e:
        logger.error(f"파일 저장 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"파일 저장 실패: {e}",
        ) from e

    # 인덱싱
    try:
        chunks_count = index_pdf(file_path)
        logger.info(f"인덱싱 완료: {chunks_count}개 청크")
    except Exception as e:
        # 실패 시 파일 삭제
        file_path.unlink(missing_ok=True)
        logger.error(f"인덱싱 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"인덱싱 실패: {e}",
        ) from e

    return IndexResponse(
        filename=file.filename,
        chunks_indexed=chunks_count,
        message=f"{chunks_count}개의 청크가 인덱싱되었습니다.",
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="RAG 질의응답",
    description="질문을 입력하면 인덱싱된 문서에서 관련 내용을 검색하고 LLM이 답변을 생성합니다.",
    tags=["retrieval"],
)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """RAG 기반 질의응답

    Args:
        request: 질문이 포함된 요청

    Returns:
        질문과 생성된 답변

    Raises:
        HTTPException: 응답 생성 실패 시
    """
    logger.info(f"쿼리 처리 시작: {request.question[:50]}...")

    settings = get_settings()
    provider = settings.config.llm.provider

    # 프로바이더별 API 키 검증
    api_key_map = {
        "openai": settings.secrets.openai_api_key,
        "google_genai": settings.secrets.google_api_key,
        "anthropic": settings.secrets.anthropic_api_key,
    }

    if provider not in api_key_map:
        logger.error(f"지원하지 않는 프로바이더: {provider}")
        raise HTTPException(
            status_code=500,
            detail=f"지원하지 않는 프로바이더: {provider}",
        )

    if not api_key_map[provider]:
        logger.error(f"{provider} API 키가 설정되지 않음")
        raise HTTPException(
            status_code=500,
            detail=f"{provider} API 키가 설정되지 않았습니다. secrets.yaml을 확인하세요.",
        )

    try:
        answer = await agenerate(request.question)
        logger.info(f"쿼리 처리 완료: {len(answer)} 글자 응답")
    except Exception as e:
        logger.error(f"응답 생성 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"응답 생성 실패: {e}",
        ) from e

    return QueryResponse(
        question=request.question,
        answer=answer,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

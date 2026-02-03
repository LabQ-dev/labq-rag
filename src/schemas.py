"""API 요청/응답 스키마

Pydantic 모델을 사용한 요청/응답 스키마 정의
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """RAG 쿼리 요청"""

    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        example="LabQ의 주요 프로젝트는 무엇인가?",
        description="질문 문자열",
    )


class QueryResponse(BaseModel):
    """RAG 쿼리 응답"""

    question: str = Field(..., example="LabQ의 주요 프로젝트는 무엇인가?")
    answer: str = Field(..., example="LabQ는 여러 프로젝트를 진행 중입니다...")


class IndexResponse(BaseModel):
    """PDF 인덱싱 응답"""

    filename: str = Field(..., example="document.pdf", description="인덱싱된 파일명")
    chunks_indexed: int = Field(..., example=42, description="생성된 청크 수", ge=0)
    message: str = Field(..., example="42개의 청크가 인덱싱되었습니다.")


class HealthResponse(BaseModel):
    """헬스체크 응답"""

    status: str = Field(..., example="healthy", description="서비스 상태")
    version: str = Field(..., example="0.1.0", description="서비스 버전")

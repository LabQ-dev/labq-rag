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


class RAGAnswer(BaseModel):
    """Structured Output용 LLM 응답 스키마

    with_structured_output()에서 사용되는 Pydantic 모델.
    LLM이 이 스키마에 맞춰 JSON 응답을 생성합니다.
    """

    answer: str = Field(..., description="질문에 대한 답변. 출처를 [문서N] 형태로 명시")
    confidence: float = Field(..., ge=0.0, le=1.0, description="답변 신뢰도 (0.0~1.0)")
    sources: list[str] = Field(
        default_factory=list, description='참조한 문서 번호 (예: ["문서1", "문서3"])'
    )


class QueryResponse(BaseModel):
    """RAG 쿼리 응답"""

    question: str = Field(..., example="LabQ의 주요 프로젝트는 무엇인가?")
    answer: str = Field(..., example="LabQ는 여러 프로젝트를 진행 중입니다...")
    confidence: float | None = Field(
        None, example=0.85, description="답변 신뢰도 (structured_output 활성 시)"
    )
    sources: list[str] = Field(
        default_factory=list,
        example=["문서1", "문서3"],
        description="참조 문서 목록 (structured_output 활성 시)",
    )


class IndexResponse(BaseModel):
    """PDF 인덱싱 응답"""

    filename: str = Field(..., example="document.pdf", description="인덱싱된 파일명")
    chunks_indexed: int = Field(..., example=42, description="생성된 청크 수", ge=0)
    message: str = Field(..., example="42개의 청크가 인덱싱되었습니다.")


class HealthResponse(BaseModel):
    """헬스체크 응답"""

    status: str = Field(..., example="healthy", description="서비스 상태")
    version: str = Field(..., example="0.1.0", description="서비스 버전")

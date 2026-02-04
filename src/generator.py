"""LLM 응답 생성 모듈

검색된 문서 컨텍스트를 기반으로 LLM 응답을 생성합니다.
LCEL (LangChain Expression Language) 패턴을 사용합니다.

지원 프로바이더: openai, google_genai, anthropic
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.config import get_settings
from src.retriever import get_retriever

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import Runnable

logger = logging.getLogger()

# RAG 프롬프트 템플릿
RAG_PROMPT_TEMPLATE = """당신은 LabQ의 내부 지식을 기반으로 질문에 답변하는 AI 어시스턴트입니다.
아래 제공된 컨텍스트만을 사용하여 질문에 답변하세요.
컨텍스트에 답변이 없다면 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답변하세요.

컨텍스트:
{context}

질문: {question}

답변:"""

# 지원 프로바이더 목록 (init_chat_model의 model_provider 값)
# 참고: https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
SUPPORTED_PROVIDERS = ("openai", "google_genai", "anthropic")

# 프로바이더별 기본 모델
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "google_genai": "gemini-3-flash-preview",
    "anthropic": "claude-3-5-haiku-latest",
}


def format_docs(docs: list[Document]) -> str:
    """Document 리스트를 문자열로 포맷팅

    Args:
        docs: Document 리스트

    Returns:
        줄바꿈으로 구분된 문서 내용 문자열
    """
    return "\n\n".join(doc.page_content for doc in docs)


def get_llm() -> BaseChatModel:
    """설정된 프로바이더에 따른 LLM 인스턴스 반환

    init_chat_model을 사용하여 provider 문자열로 동적 생성.
    model이 지정되지 않으면 프로바이더별 기본 모델 사용.

    Returns:
        BaseChatModel: LLM 인스턴스

    Raises:
        ValueError: 지원하지 않는 프로바이더인 경우
    """
    settings = get_settings()
    provider = settings.config.llm.provider
    temperature = settings.config.llm.temperature

    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"지원하지 않는 LLM 프로바이더: {provider}. " f"가능한 값: {SUPPORTED_PROVIDERS}"
        )

    # model이 None이면 프로바이더별 기본값 사용
    model = settings.config.llm.model or DEFAULT_MODELS[provider]

    # 프로바이더별 API 키 매핑 (secrets.yaml의 키 이름은 간결하게 유지)
    api_key_map = {
        "openai": settings.secrets.openai_api_key,
        "google_genai": settings.secrets.google_api_key,
        "anthropic": settings.secrets.anthropic_api_key,
    }

    timeout = settings.config.llm.timeout

    logger.debug(f"LLM 초기화: provider={provider}, model={model}, timeout={timeout}s")

    return init_chat_model(
        model=model,
        model_provider=provider,
        temperature=temperature,
        api_key=api_key_map[provider],
        timeout=timeout,
    )


def get_rag_chain() -> Runnable:
    """RAG 체인 반환

    LCEL을 사용한 RAG 파이프라인:
    1. 질문으로 관련 문서 검색
    2. 검색된 문서를 컨텍스트로 포맷팅
    3. 프롬프트에 컨텍스트와 질문 삽입
    4. LLM으로 응답 생성
    5. 문자열로 파싱

    Returns:
        실행 가능한 RAG 체인
    """
    retriever = get_retriever()
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def generate(question: str) -> str:
    """질문에 대한 RAG 응답 생성

    Args:
        question: 사용자 질문

    Returns:
        LLM이 생성한 응답
    """
    logger.debug(f"응답 생성 시작: {question[:50]}...")
    chain = get_rag_chain()
    answer = chain.invoke(question)
    logger.info(f"응답 생성 완료: {len(answer)} 글자")
    return answer


async def agenerate(question: str) -> str:
    """질문에 대한 RAG 응답 비동기 생성

    Args:
        question: 사용자 질문

    Returns:
        LLM이 생성한 응답
    """
    logger.debug(f"비동기 응답 생성 시작: {question[:50]}...")
    chain = get_rag_chain()
    answer = await chain.ainvoke(question)
    logger.info(f"비동기 응답 생성 완료: {len(answer)} 글자")
    return answer

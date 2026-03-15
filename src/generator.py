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
from langchain_core.runnables import RunnablePassthrough

from src.config import get_settings
from src.prompts import build_prompt, format_context
from src.retriever import get_retriever
from src.schemas import RAGAnswer

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import Runnable

logger = logging.getLogger()

# 지원 프로바이더 목록 (init_chat_model의 model_provider 값)
SUPPORTED_PROVIDERS = ("openai", "google_genai", "anthropic")

# 프로바이더별 기본 모델
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "google_genai": "gemini-3-flash-preview",
    "anthropic": "claude-3-5-haiku-latest",
}


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

    # 프로바이더별 API 키 매핑
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
    2. 검색된 문서를 재배치(config) + [문서N] 태깅
    3. Citation 프롬프트에 컨텍스트와 질문 삽입
    4. LLM으로 응답 생성
    5. structured_output 설정에 따라 RAGAnswer 또는 문자열로 파싱

    Returns:
        실행 가능한 RAG 체인
    """
    settings = get_settings()
    gen_config = settings.config.generation

    retriever = get_retriever()
    llm = get_llm()
    prompt = build_prompt()

    # config 기반 문서 포맷터: reorder + tag_docs
    def _format_docs_with_config(docs: list) -> str:
        return format_context(docs, reorder=gen_config.reorder_docs)

    # structured_output 분기: RAGAnswer 또는 StrOutputParser
    output_parser: Runnable
    if gen_config.structured_output:
        output_parser = llm.with_structured_output(RAGAnswer)
    else:
        output_parser = llm | StrOutputParser()

    return (
        {
            "context": retriever | _format_docs_with_config,
            "question": RunnablePassthrough(),
        }
        | prompt
        | output_parser
    )


def generate(question: str) -> RAGAnswer | str:
    """질문에 대한 RAG 응답 생성

    Args:
        question: 사용자 질문

    Returns:
        structured_output=true: RAGAnswer, false: str
    """
    logger.debug(f"응답 생성 시작: {question[:50]}...")
    chain = get_rag_chain()
    result = chain.invoke(question)
    logger.info(f"응답 생성 완료: {type(result).__name__}")
    return result


async def agenerate(question: str) -> RAGAnswer | str:
    """질문에 대한 RAG 응답 비동기 생성

    Args:
        question: 사용자 질문

    Returns:
        structured_output=true: RAGAnswer, false: str
    """
    logger.debug(f"비동기 응답 생성 시작: {question[:50]}...")
    chain = get_rag_chain()
    result = await chain.ainvoke(question)
    logger.info(f"비동기 응답 생성 완료: {type(result).__name__}")
    return result

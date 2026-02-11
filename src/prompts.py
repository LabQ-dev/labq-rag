"""프롬프트 및 문서 처리 모듈

Citation 프롬프트(항상 적용), 문서 번호 태깅, Lost in the Middle 재배치를 담당합니다.
빌더 패턴으로 설계되어 추후 CoT 등 기법을 플래그 토글로 확장 가능합니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from langchain_core.documents import Document

# --- 프롬프트 구성 블록 ---

BASE_INSTRUCTION = (
    "당신은 LabQ의 내부 지식을 기반으로 질문에 답변하는 AI 어시스턴트입니다.\n"
    "아래 제공된 컨텍스트만을 사용하여 질문에 답변하세요.\n"
    "컨텍스트에 답변이 없다면 "
    '"제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답변하세요.'
)

CITATION_INSTRUCTION = (
    "답변에 반드시 출처를 [문서N] 형태로 명시하세요.\n"
    "여러 문서를 참조한 경우 각각의 출처를 모두 표기하세요."
)

# 추후 확장 예시:
# COT_INSTRUCTION = (
#     "답변하기 전에 단계별로 사고 과정을 보여주세요.\n"
#     "각 단계에서 어떤 문서를 참조했는지 명시하세요."
# )

CONTEXT_AND_QUESTION = "컨텍스트:\n{context}\n\n질문: {question}\n\n답변:"


# --- 문서 처리 ---


def tag_docs(docs: list[Document]) -> str:
    """문서에 [문서N] 태그를 붙여 포맷팅

    Args:
        docs: Document 리스트 (순서대로 태깅)

    Returns:
        [문서1] ... 형태의 포맷팅된 문자열
    """
    return "\n\n".join(f"[문서{i + 1}] {doc.page_content}" for i, doc in enumerate(docs))


def reorder_docs(docs: list[Document]) -> list[Document]:
    """Lost in the Middle 대응: 관련도 높은 문서를 앞/뒤에 배치

    LLM은 컨텍스트의 앞부분과 뒷부분에 더 주의를 기울이므로,
    1위 → 맨 앞, 2위 → 맨 뒤, 나머지 → 중간에 배치합니다.

    Args:
        docs: 관련도 순으로 정렬된 Document 리스트

    Returns:
        재배치된 Document 리스트
    """
    if len(docs) <= 2:
        return docs
    return [docs[0], *docs[2:], docs[1]]


def format_context(docs: list[Document], *, reorder: bool = True) -> str:
    """문서 재배치 + 태깅을 통합한 컨텍스트 포맷팅

    Args:
        docs: Document 리스트
        reorder: Lost in the Middle 재배치 적용 여부

    Returns:
        [문서N] 태그가 붙은 포맷팅된 컨텍스트 문자열
    """
    ordered = reorder_docs(docs) if reorder else docs
    return tag_docs(ordered)


# --- 프롬프트 빌더 ---


def build_prompt() -> ChatPromptTemplate:
    """Citation이 내장된 RAG 프롬프트 생성

    빌더 패턴으로 프롬프트 블록을 동적 조합합니다.
    Citation은 항상 포함되며, 추후 CoT 등은 config 플래그로 토글 가능.

    Returns:
        {context}, {question} 변수를 포함하는 ChatPromptTemplate
    """
    sections = [BASE_INSTRUCTION, CITATION_INSTRUCTION]
    # 추후 확장: if config.cot: sections.append(COT_INSTRUCTION)
    sections.append(CONTEXT_AND_QUESTION)
    template = "\n\n".join(sections)
    return ChatPromptTemplate.from_template(template)

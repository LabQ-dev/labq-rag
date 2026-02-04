# LabQ RAG Service - AI Agent Context

이 문서는 AI 코딩 어시스턴트(Cursor, Claude Code 등)가 프로젝트 컨텍스트를 이해하는 데 사용됩니다.

## 프로젝트 개요

PDF 문서 기반 RAG(Retrieval-Augmented Generation) 서비스입니다.
LabQ 내부 지식을 활용하여 질문에 답변하는 것이 목표입니다.

## 기술 스택

- **언어**: Python 3.12 (`.python-version`으로 고정)
- **패키지 관리**: uv (`uv.lock`으로 버전 잠금)
- **웹 프레임워크**: FastAPI
- **RAG 프레임워크**: LangChain 1.0+, LangGraph
- **벡터 DB**: Qdrant
- **임베딩 모델**: bge-m3 (HuggingFace, 로컬 CPU)
- **LLM**: OpenAI / Google Gemini / Anthropic Claude (설정으로 선택)
- **컨테이너**: Docker, Docker Compose

## 프로젝트 구조

```
labq-rag/
├── src/                    # 소스 코드 (플랫 구조)
│   ├── main.py             # FastAPI 엔드포인트
│   ├── config.py           # YAML 설정 로더
│   ├── logger.py           # 로깅 설정
│   ├── schemas.py          # Pydantic 스키마
│   ├── indexer.py          # PDF 인덱싱
│   ├── retriever.py        # 벡터 검색
│   └── generator.py        # RAG 응답 생성 (LCEL)
├── configs/                # 설정 파일
│   ├── config.yaml         # 앱 설정 (Git 관리)
│   ├── logging.yaml        # 로깅 설정 (Git 관리)
│   ├── secrets.yaml        # 비밀 정보 (Git 제외)
│   └── secrets.example.yaml# 비밀 정보 템플릿
├── .github/                # GitHub 템플릿
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── .cursor/rules/          # Cursor IDE 규칙
├── pyproject.toml          # 의존성 정의
├── uv.lock                 # 버전 잠금 파일
├── .python-version         # Python 버전 (3.12)
├── compose.yaml            # Docker Compose
├── Dockerfile
├── data/                   # PDF 저장 디렉토리
├── logs/                   # 로그 파일
├── examples/               # 테스트용 샘플 데이터
└── tests/                  # 테스트
```

## 코딩 컨벤션

### Python 스타일

- **Type Hints 필수**: 모든 함수의 인자와 반환값에 타입 힌트 작성
- **Docstring**: Google Style 사용
- **린터/포매터**: ruff (line-length: 100)
- **네이밍**:
  - 함수/변수: snake_case
  - 클래스: PascalCase
  - 상수: UPPER_SNAKE_CASE

### 코드 예시

```python
def process_document(file_path: str | Path, chunk_size: int = 1000) -> list[Document]:
    """문서를 처리하여 청크로 분할합니다.

    Args:
        file_path: 처리할 파일 경로
        chunk_size: 청크 크기 (기본값: 1000)

    Returns:
        분할된 Document 리스트

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
    """
    ...
```

### 금지 사항

- `print()` 사용 금지 → `logging` 사용
- 하드코딩된 API 키, 비밀 정보 금지
- `Any` 타입 사용 최소화 → `unknown`이면 타입 좁히기 사용
- 주석으로 "What" 설명 금지 → "Why" 설명만

### 권장 패턴

- **설정**: `configs/config.yaml` (앱 설정), `configs/secrets.yaml` (비밀 정보)
- **RAG Chain**: LCEL (LangChain Expression Language) 사용
- **비동기**: FastAPI 엔드포인트는 async/await 사용
- **의존성 주입**: `get_settings()` 같은 팩토리 함수 사용

## 커밋 메시지 규칙 (Conventional Commits)

```
<type>: <description>

예시:
feat: add PDF multi-page support
fix: handle empty query error in retriever
docs: update installation guide
refactor: separate YAML loader
test: add RAG chain unit tests
chore: add sentence-transformers dependency
```

**참고:** scope는 사용하지 않습니다. `feat(indexer):` ❌ → `feat:` ✅

**타입**:
- `feat`: 새 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 포맷팅, 세미콜론 등 (코드 변경 없음)
- `refactor`: 리팩토링
- `perf`: 성능 개선
- `test`: 테스트 추가/수정
- `chore`: 빌드, 설정 변경

## 주요 의존성 버전

- langchain >= 1.0.0
- langgraph >= 1.0.0
- fastapi >= 0.128.0
- qdrant-client >= 1.16.0
- pypdf >= 6.6.0
- numpy == 1.26.4 (2.x 호환성 이슈 방지)

## 설정 관리

모든 설정을 `configs/` 폴더에서 YAML로 관리합니다.

### 설정 파일

- `configs/config.yaml`: 앱 설정 (Git 관리)
- `configs/secrets.yaml`: 비밀 정보 (Git 제외)
- `configs/secrets.example.yaml`: 비밀 정보 템플릿 (Git 관리)
- `configs/logging.yaml`: 로깅 설정 (Git 관리)

### LLM 프로바이더 설정

`configs/config.yaml`에서 프로바이더 선택 (model 미지정 시 기본값 사용):

```yaml
llm:
  provider: "google_genai"   # openai, google_genai, anthropic
  # model: "gemini-3-flash-preview"  # 미지정 시 기본값 사용
  temperature: 0.0
  timeout: 30  # 초

qdrant:
  host: "${QDRANT_HOST:-localhost}"  # Docker: qdrant, 로컬: localhost
  timeout: 5  # 초
```

**프로바이더별 기본 모델:**
- `openai` → gpt-4o-mini
- `google_genai` → gemini-3-flash-preview
- `anthropic` → claude-3-5-haiku-latest

**타임아웃 설정:**
- Qdrant: 5초 (연결/검색)
- LLM: 30초 (응답 생성)

`init_chat_model`을 사용하여 provider 문자열로 동적 LLM 생성.
provider 이름은 LangChain의 `init_chat_model` 규격을 따릅니다.

### 로깅

QueueHandler + RotatingFileHandler를 사용한 스레드 안전 로깅:

```python
import logging
from src.logger import setup_logging

# 앱 시작 시 (main.py에서 한 번만)
setup_logging()
logger = logging.getLogger()  # 루트 로거 사용

# 사용
logger.debug("디버그 정보")
logger.info("일반 정보")
logger.warning("경고")
logger.error("에러", exc_info=True)  # 스택트레이스 포함
```

로그는 `logs/labq_rag.log`에 저장되며, 10MB마다 자동 로테이션됩니다.

## 에러 처리

### FastAPI 에러 처리

- 구체적 HTTP 상태 코드 사용
- 예외 체이닝으로 원인 추적
- 사용자 친화적 메시지

```python
try:
    result = await some_operation()
except SpecificError as e:
    logger.error(f"작업 실패: {e}", exc_info=True)
    raise HTTPException(
        status_code=400,
        detail="작업 실패: 이유 설명"
    ) from e  # 원인 예외 추적
```

## 성능 최적화

- **비동기**: FastAPI 엔드포인트는 async/await 필수
  - `await retriever.ainvoke()` (async 버전)
- **캐싱**: 자주 로드되는 모델은 싱글톤
  - `get_settings()`, `get_embeddings()` 등
- **배치**: 여러 쿼리는 배치 처리 고려
- **타임아웃**: 장시간 작업에 제한 시간 설정

```python
# 비동기 임베딩
answer = await chain.ainvoke(query)

# 싱글톤 패턴
@lru_cache
def get_llm():
    return ChatOpenAI(...)
```

## PR 제출 전 자가 리뷰 체크리스트

### 코드 품질
- [ ] Type hints 모든 함수에 포함?
- [ ] Docstring (Args, Returns, Raises) 완성?
- [ ] `uv run ruff check src/` 통과?
- [ ] 불필요한 import 제거?
- [ ] `Any` 타입 없음?
- [ ] `print()` 사용 안 함?

### 기능성
- [ ] 주요 로직 async 처리?
- [ ] 에러 처리 (try-except) 있음?
- [ ] 민감 정보 하드코딩 안 함?
- [ ] 로거 추가됨 (info/debug/error)?

### RAG 특화
- [ ] LCEL 파이프라인 사용?
- [ ] retriever는 적절한 top_k 설정?
- [ ] prompt 템플릿 명확?

### 커밋
- [ ] Conventional Commits 형식?
- [ ] 커밋 메시지 설명적?

## LCEL (LangChain Expression Language)

LangChain 1.0+의 핵심 개념. 파이프라인을 함수형 프로그래밍 스타일로 구성합니다.

### 예시: RAG 체인

```python
chain = (
    {
        "context": retriever | format_docs,  # 검색 후 포맷
        "question": RunnablePassthrough()     # 질문 통과
    }
    | prompt                                  # 프롬프트 적용
    | llm                                     # LLM 호출
    | StrOutputParser()                       # 파싱
)

answer = chain.invoke(query)        # 동기 실행
answer = await chain.ainvoke(query) # 비동기 실행
```

### 장점
- **가독성**: 파이프라인이 시각적으로 명확 (`|`로 연결)
- **유연성**: 각 단계를 독립적으로 테스트 가능
- **성능**: 비동기 지원 자동

## 주석 및 문서화

### Why 주석 (필수)

복잡한 로직에는 **"Why" 주석** 필수:

```python
# Qdrant와의 네트워크 지연을 고려하여
# 미리 top_k*2로 검색 후 필터링
raw_results = vectorstore.similarity_search(query, k=top_k*2)
```

**하지 말 것:**
```python
# ❌ 코드가 이미 말해주는 "What"
# 리스트를 순회한다
for item in items:
    process(item)
```

### 코드 변경 시 문서 동기화

**핵심: 코드 변경 → 영향받는 모든 문서 업데이트**

- 함수 시그니처 변경 → Docstring
- API 엔드포인트 변경 → schemas.py + AGENTS.md
- 설정값 변경 → config.yaml + AGENTS.md
- 의존성 추가 → pyproject.toml + AGENTS.md

**체크리스트 (코드 변경 후):**
- [ ] Docstring (Args, Returns, Raises) 최신?
- [ ] AGENTS.md API 섹션 최신?
- [ ] 설정 파일 변경 시 문서 반영?
- [ ] 주석의 "Why"가 명확한가?


## 실행 명령어

```bash
# 개발 서버
uv run uvicorn src.main:app --reload

# Docker
docker compose up -d

# 테스트
uv run pytest

# 린트
uv run ruff check src/
uv run ruff format src/

# Pre-commit 설치
uv run pre-commit install
```

## API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | /health | 헬스체크 |
| POST | /index | PDF 업로드 및 인덱싱 |
| POST | /query | RAG 질의응답 |

## 환경 설정

1. `configs/secrets.example.yaml`을 `configs/secrets.yaml`로 복사
2. 사용할 프로바이더의 API 키 입력
3. `configs/config.yaml`에서 `llm.provider` 설정
4. Docker로 Qdrant 실행 또는 `docker compose up` 사용

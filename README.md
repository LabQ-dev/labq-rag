# LabQ RAG Service

PDF 문서 기반 RAG(Retrieval-Augmented Generation) 서비스입니다.

## 기술 스택

- **Python 3.12**
- **LangChain 1.0+** - RAG 파이프라인
- **Qdrant** - 벡터 데이터베이스
- **FastAPI** - REST API
- **bge-m3** - 임베딩 모델
- **LLM**: OpenAI / Google Gemini / Anthropic Claude (선택 가능)
- **uv** - 패키지 관리

## 프로젝트 구조

```
labq-rag/
├── pyproject.toml          # 의존성 정의
├── uv.lock                 # 버전 잠금 파일
├── .python-version         # Python 버전 고정
├── configs/
│   ├── config.yaml         # 앱 설정 (Git 관리)
│   ├── logging.yaml        # 로깅 설정
│   ├── secrets.yaml        # 비밀 정보 (Git 제외)
│   └── secrets.example.yaml# 비밀 정보 템플릿
├── Dockerfile
├── compose.yaml
├── src/
│   ├── __init__.py
│   ├── main.py             # FastAPI 엔드포인트
│   ├── config.py           # 설정 로더
│   ├── logger.py           # 로깅 설정
│   ├── schemas.py          # Pydantic 스키마
│   ├── indexer.py          # PDF 인덱싱
│   ├── retriever.py        # 벡터 검색
│   └── generator.py        # LLM 응답 생성
├── data/                   # PDF 저장 디렉토리
├── logs/                   # 로그 파일
├── .github/
│   ├── ISSUE_TEMPLATE/     # 이슈 템플릿
│   └── PULL_REQUEST_TEMPLATE.md
└── tests/
```

## 빠른 시작

### 1. 사전 요구사항

- Docker & Docker Compose
- API 키 (OpenAI / Google / Anthropic 중 하나)

### 2. 설정

```bash
# secrets.yaml 생성
cp configs/secrets.example.yaml configs/secrets.yaml

# API 키 입력
# configs/secrets.yaml 파일을 편집하여 사용할 프로바이더의 api_key 값을 입력하세요
```

### 3. 실행 (Docker)

```bash
# 서비스 시작
docker compose up -d

# 로그 확인
docker compose logs -f app
```

서비스가 시작되면:
- API: http://localhost:8000
- API 문서: http://localhost:8000/docs
- Qdrant Dashboard: http://localhost:6333/dashboard

### 4. 로컬 개발 (Docker 없이)

```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync

# Qdrant 실행 (별도 터미널)
docker run -p 6333:6333 qdrant/qdrant

# 서버 실행
uv run uvicorn src.main:app --reload
```

## API 사용법

### 헬스체크

```bash
curl http://localhost:8000/health
```

### PDF 인덱싱

```bash
curl -X POST http://localhost:8000/index \
  -F "file=@/path/to/document.pdf"
```

### 질의응답

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "문서에서 찾고 싶은 내용"}'
```

## 설정

### configs/config.yaml

앱 설정 파일입니다. Git으로 관리됩니다.

```yaml
embedding:
  model_name: "BAAI/bge-m3"  # 임베딩 모델
  device: "cpu"              # cpu 또는 cuda

splitter:
  chunk_size: 1000           # 청크 크기
  chunk_overlap: 200         # 청크 오버랩

retriever:
  top_k: 5                   # 검색 결과 수

qdrant:
  collection_name: "labq_docs"
  host: "localhost"
  port: 6333

llm:
  provider: "google"         # openai, google, anthropic
  temperature: 0.0
  openai_model: "gpt-5-mini"
  google_model: "gemini-3.0-flash"
  anthropic_model: "claude-haiku-4-5"
```

### configs/secrets.yaml

비밀 정보 파일입니다. **Git에 커밋하지 마세요.**

```yaml
openai:
  api_key: "sk-your-api-key"

google:
  api_key: "your-gemini-api-key"

anthropic:
  api_key: "your-claude-api-key"
```

## 개발

### 테스트 실행

```bash
uv run pytest
```

### 린트

```bash
uv run ruff check src/
uv run ruff format src/
```

### Pre-commit 훅 설치

```bash
uv run pre-commit install
```

## 라이선스

MIT License

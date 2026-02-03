FROM python:3.12-slim

# uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# 의존성 설치 (캐시 최적화)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 설정 파일 복사
COPY config.yaml ./
COPY secrets.yaml ./

# 소스 코드 복사
COPY src ./src

# 데이터 디렉토리 생성
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

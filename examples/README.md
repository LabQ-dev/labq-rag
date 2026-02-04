# RAG 테스트 예시

MVP 테스트를 위한 샘플 데이터와 질문입니다.

## 파일 구성

| 파일 | 설명 |
|------|------|
| `협업 및 커뮤니케이션 컨벤션.pdf` | 테스트용 PDF 문서 |
| `test_queries.json` | 테스트 질문 목록 |

## 테스트 방법

### 1. PDF 인덱싱

```bash
curl -X POST http://localhost:8000/index \
  -F 'file=@examples/협업 및 커뮤니케이션 컨벤션.pdf'
```

### 2. 질문 테스트

**질문 1 (쉬움):** 단순 사실 확인
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "코드 리뷰는 PR 생성 후 몇 시간 이내에 해야 하나요?"}'
```

**질문 2 (중간):** 여러 정보 조합
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "실험 브랜치 이름은 어떤 형식으로 만들어야 하고, 실험 결과는 어디에 저장하나요?"}'
```

**질문 3 (어려움):** 추상적 질문
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "새로운 기능을 merge하려면 어떤 조건들을 충족해야 하나요?"}'
```

## 예상 답변

### 질문 1
> PR 생성 후 **24시간 이내** 1차 리뷰, **48시간 이내** approve/request changes

### 질문 2
> 브랜치: `exp/{담당자명}/{실험내용}` 형식 (예: `exp/kim/bge-m3-finetuning`)
> 저장 위치: `/docs/experiments/` 폴더

### 질문 3
> - 최소 **5% 이상** 성능 향상 (통계적 유의성 확인)
> - 최소 **1명 approve** 필수 (핵심 변경은 2명)
> - 테스트 통과

## 평가 기준

| 난이도 | 기준 |
|--------|------|
| easy | 예상 키워드 중 2개 이상 포함 |
| medium | 예상 키워드 중 3개 이상 포함 |
| hard | 예상 키워드 중 2개 이상 + 논리적 연결 |

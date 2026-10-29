# 파이썬 3.11 슬림 이미지 기반
FROM python:3.11-slim

# 표준 출력/에러 버퍼링 방지 및 pyc 생성 억제
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 빌드 및 c++ 의존성(ChromaDB 빌드용 컴파일러 라이브러리) 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 종속성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY main.py .

# Cloud Run 기본 포트 (환경변수 PORT에 맞춰 실행)
ENV PORT=8080
EXPOSE 8080

# uvicorn 실행 (PORT 환경변수를 받아 실행)
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}

# 베이스 이미지 설정
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 환경 파일 복사
COPY requirement.txt .

# Conda 환경 생성 및 의존성 설치
RUN pip install -r requirement.txt

# FastAPI 앱 소스 복사
COPY ./app .

# FastAPI 앱 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]

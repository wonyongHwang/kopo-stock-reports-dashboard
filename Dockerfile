# 베이스 이미지: Python 3.11 slim
FROM python:3.11-slim

# 시스템 업데이트 및 필수 패키지 설치 (타임존, 로케일 등)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 종속성 복사 & 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스 복사 직전/직후에 .streamlit도 함께 복사
COPY .streamlit/ ./.streamlit/

# 앱 소스 복사
COPY app_dashboard.py ./
# 필요 시 다른 모듈도 같이 복사
# COPY firestore_store.py extract_fields_openai.py ./

# Streamlit 기본 포트
ENV PORT=8080

# Streamlit 설정: 서버 주소/포트 고정
CMD ["streamlit", "run", "app_dashboard.py", "--server.port=8080", "--server.address=0.0.0.0"]

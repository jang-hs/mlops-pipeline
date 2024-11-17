# Base image
FROM python:3.10-slim

# 작업 디렉터리 설정
WORKDIR /app

# 패키지 설치
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# 추론 실행 파일 복사
COPY logger.py /app/logger.py
COPY inference.py /app/inference.py

CMD ["python", "inference.py"]
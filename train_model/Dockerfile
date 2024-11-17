# Base image
FROM python:3.10-slim

# 작업 디렉터리 설정
WORKDIR /app

# 패키지 설치
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# 학습 모델 실행 파일 복사
COPY logger.py /app/logger.py
COPY train_model.py /app/train_model.py

# 모델 학습 실행
CMD ["python", "train_model.py"]
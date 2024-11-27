docker stop dami-server || true
docker rm dami-server || true

# 새 컨테이너 실행
docker run -d \
  --env-file .env \
  --name dami-server \
  --restart unless-stopped \
  -v user-data:/app/data \
  -p 8000:8000 \
  shortboy710/dami-server:latest
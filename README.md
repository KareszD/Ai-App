docker build -t nnnapi .  
docker run -d -p  5000:5000 --gpus=all --name foliage_app_container nnnapi

docker build -t nnnapi .  

Api container indítása
docker run -d -p  5000:5000 --gpus=all --name foliage_app_container nnnapi

Alkalmazás buildelése:
npm run dist

Eloszott tanítást megvalúsító image
docker pull karesz/yoloddp:1.0.1

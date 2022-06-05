# build
docker build -f Dockerfile -t docker_centos_prod .

# docker-compose
docker-compose -f cnn_prod.yaml up

# docker exec -ti centos_container_prod /bin/bash
# docker-compose -f cnn_prod.yaml down
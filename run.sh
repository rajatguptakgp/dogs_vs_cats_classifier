# build
docker build -f Dockerfile -t docker_centos .
docker build -f Dockerfile3 -t docker_tensorflow .

# docker-compose
docker-compose -f cnn.yaml up

# docker exec -ti tf_container /bin/bash
# docker exec -ti centos_container /bin/bash
# docker-compose -f cnn.yaml down
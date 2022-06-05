# Python Package to create Vision Classifier

Dataset: Dogs v/s Cats 

### Docker
#### Steps:
1. Build Docker images: 
    1. CentOS: <code>docker build -f Dockerfile -t docker_centos .</code>
    2. Tensorflow: <code>docker build -f Dockerfile3 -t docker_tensorflow .</code>
2. Running Docker-Compose: <code>docker-compose -f cnn.yaml up</code>
3. Enter containers:
    1. <code>docker exec -ti tf_container /bin/bash</code>
    2. <code>docker exec -ti centos_container /bin/bash</code>
4. Shut-down all containers: <code>docker-compose -f cnn.yaml down</code>

#### Additional: 
1. Run docker container: <code>docker run -ti docker_centos</code>
2. Stop all containers: <code>docker kill $(docker ps -q)</code>

# To build the image
docker build -t hakab/chm:latest .

# To run the image in interactive mode
docker run -it hakab/chm:latest bash

# To execute a command inside a running container
docker exec -it hakab/chm:latest bash

docker images

# show all running images
docker ps -all

# Remove all unused containers, networks, images (both dangling and unreferenced), and optionally, volumes.
docker system prune

# To stop all running containers use the docker container stop command followed by a list of all containers IDs.
docker container stop $(docker container ls -aq)

# Once all containers are stopped, you can remove them using the docker container rm command followed by the containers ID list.
docker container rm $(docker container ls -aq)

docker container stop $(docker container ls -aq) && docker container rm $(docker container ls -aq)


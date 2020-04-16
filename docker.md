# To build the image
docker build -t corona_hakab_image .

# To run the image in interactive mode
docker run -it corona_hakab_image bash

# To execute a command inside a running container
docker exec -it corona_hakab_image bash

# show all running images
docker ps -all

# Remove all unused containers, networks, images (both dangling and unreferenced), and optionally, volumes.
docker system prune

# To stop all running containers use the docker container stop command followed by a list of all containers IDs.
docker container stop $(docker container ls -aq)

# Once all containers are stopped, you can remove them using the docker container rm command followed by the containers ID list.
docker container rm $(docker container ls -aq)

docker container stop $(docker container ls -aq) && docker container rm $(docker container ls -aq)


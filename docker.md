# To build the image
sudo docker build -t hakab/chm:latest .

# To run the image in interactive mode
sudo docker run -it hakab/chm:latest bash

# To execute a command inside a running container
sudo docker exec -it hakab/chm:latest bash

sudo docker images

# show all running images
sudo docker ps -all

# Remove all unused containers, networks, images (both dangling and unreferenced), and optionally, volumes.
sudo docker system prune

# To stop all running containers use the docker container stop command followed by a list of all containers IDs.
sudo docker container stop $(docker container ls -aq)

# Once all containers are stopped, you can remove them using the docker container rm command followed by the containers ID list.
sudo docker container rm $(docker container ls -aq)

sudo docker container stop $(docker container ls -aq) && docker container rm $(docker container ls -aq)


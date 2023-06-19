if [ -x "$(command -v docker)" ]; then
    echo "Docker is installed"
else
    echo "You should Install docker"
fi
# built the container
docker build -t tutorial_2023 .

# run the container
docker run --name tutorial_2023 \
            -p 8888:8888 \
            --user root \
            tutorial_2023
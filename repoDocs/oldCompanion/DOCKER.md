# A quick reference guide for Docker on the Jetson

## Getting a proper docker
```
sudo apt -y purge docker-ce docker-ce-cli containerd.io
sudo apt -y install docker.io nvidia-container-toolkit
sudo usermod -aG docker $USER
newgrp docker
```

## Docker credentials
Obtain nvcr credentials (https://ngc.nvidia.com/signin) and then run this as your user and not `root`
```
docker login nvcr.io
$oauthtoken:<Your creds>
```

## Build docker
docker build -f Dockerfile.jetson -t baseline.docker .
docker build -f Dockerfile.jetson -t baseline.docker --network=host . # workaround for iptable_raw

## Temp
docker run -it --rm \
  --runtime nvidia \
  --network host \
  --privileged \
  -v /tmp:/tmp \
  -v /dev:/dev \
  -v /etc/enctune.conf:/etc/enctune.conf \
  -v /var/run/dbus:/var/run/dbus \
  --env DISPLAY=$DISPLAY \
  --env QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  baseline.docker

## Perm
docker run -it --name jetson-dev \
  --runtime nvidia \
  --network host \
  --privileged \
  -v /tmp:/tmp \
  -v /dev:/dev \
  -v /etc/enctune.conf:/etc/enctune.conf \
  -v /var/run/dbus:/var/run/dbus \
  --env DISPLAY=$DISPLAY \
  --env QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  baseline.docker


## Thoughts on vid:
--device /dev/video0 \
--device /dev/v4l-subdev0 \
--device /dev/media0

## then exit and if you want back do
docker start -ai jetson-dev

## purge
docker rm jetson-dev

## merge
docker commit jetson-dev jetson-dev-img

## export
docker save -o jetson-dev-img.tar jetson-dev-img

## load
docker load -i jetson-dev-img.tar


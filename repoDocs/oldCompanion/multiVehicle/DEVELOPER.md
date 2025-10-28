## Starting from scratch
This is a reference file based on a repeatability test with no set of detailed instructions other than following the `.md` trail.

This tutorial has been created around a Laptop running `Ubuntu 22.04.5 LTS`.  This is a cheaper alternative to a Jetson with the idea being that while we need to mimic the real-world scenario as much as possible, sometimes we don't always have the higher end gear physically present.

## System prep
pip can sometimes muddy the waters.  Before beginning this tutorial make sure you backup anything pip as it will restore your Python installation so that it only has apt based modules.  Once you've backed up anything needed go ahead and run `cleanPip.sh` as a normal user and without sudo.  It will first clean that user's cache and then ask you to sudo for root.

## Initial packages
Install Jammy and then:
```
sudo apt update
sudo apt -y dist-upgrade
sudo apt -y install build-essential\
                    python3-pip\
                    ros-humble-geographic-msgs\
                    ros-humble-mavros\
                    ros-humble-mavros-msgs\
                    ros-humble-tf-transformations\
                    ros-humble-angles\
                    ninja-build\
                    python3-cffi\
                    arduino\
                    libcusparselt0\ libcusparselt-dev python3-pygame libgirepository1.0-dev libcairo2-dev gir1.2-gtk-3.0 python3-dev meson

sudo apt install libgstrtspserver-1.0-dev gstreamer1.0-tools git build-essential meson ninja-build libgstreamer1.0-dev

## Workaround for video_receiver probems
sudo apt -y install ros-humble-rmw-cyclonedds-cpp

## tutorials
ros-humble-turtlesim

## video_receiver tracing below
sudo apt -y install gstreamer1.0-tools

sudo apt -y install gir1.2-gst-rtsp-server-1.0

sudo apt -y install ros-humble-ros-gz-bridge

sudo apt -y install python3-rosbag

sudo apt -y install ros-humble-rosgraph-msgs

sudo apt -y install ros-humble-rqt-image-view

sudo apt -y install ros-humble-ros-gz
```

## Cleanout all prior CUDA and Nvidia things.
```
sudo apt -y purge 'cuda-*' 'nvidia-*'
sudo apt -y autoremove
for pkg in libcudnn8 libcudnn8-dev libnvidia-compute-535:amd64; do
    if dpkg -s "$pkg" >/dev/null 2>&1; then
        sudo apt -y purge "$pkg"
    fi
done
dpkg --list | grep -Ei 'cuda|nvidia'

## If nothing cuda or nvidia remains, proceed.  If things remain cleanup as needed
reboot
```

## Install the Nvidia driver for your workstation
The directions below are explicit for an RTX 3050.  Anything other than this should be researched before proceeding as Nvidia and CUDA can be very, unfun.
```
sudo apt update
sudo apt -y install nvidia-driver-535
reboot
```

## Add CUDA repo and do any needed updates
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm -f cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y dist-upgrade
```

## Error handling
For whatever reason there is an issue with the upgrade process as of July 2025.

dpkg: error processing archive /tmp/apt-dpkg-install-6auGMF/03-libnvidia-extra-535_535.261.03-0ubuntu1_amd64.deb (--unpack):
 trying to overwrite '/usr/lib/x86_64-linux-gnu/libnvidia-api.so.1', which is also in package libnvidia-gl-535:amd64 535.247.01-0ubuntu0.22.04.1
Errors were encountered while processing:
 /tmp/apt-dpkg-install-S5WcYr/03-libnvidia-extra-535_535.261.03-0ubuntu1_amd64.deb

To work around this proceed as follows:
```
sudo apt -yf install
sudo apt -y autoremove
reboot
```

## Install CUDA
```
sudo apt -y install cuda-toolkit-12-1
sudo ln -s /usr/local/cuda-12.1/nvvm/bin/cicc /usr/local/cuda-12.1/bin/cicc
echo 'export PATH=/usr/local/cuda/bin:$PATH' | sudo tee /etc/profile.d/cuda.sh
sudo chmod +x /etc/profile.d/cuda.sh
reboot
```

## Create a test
```
cat << EOF > test.cu
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel() {
    printf("Hello from CUDA kernel!\n");
}

int main() {
    kernel<<<1, 1>>>();
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
EOF
```

## Test nvcc
Running the below should yield `Hello from CUDA kernel!`
```
nvcc -o test test.cu
```

## Steps for cudnn
Below are the steps for cudnn.  At the end of these steps you should see output from ldconfig
```
sudo apt install libcudnn8 libcudnn8-dev
sudo ldconfig
ldconfig -p | grep cudnn
```

## Testing the ROS src
Due to the use of a virtual environment the shebang for every package has to be rewritten so that ROS knows which Python to use.  If you are using this outside of a virtual environment you can disregard the postBuild.sh portion of these instructions:
```
cd ~/internal-fw-companion-ros2/ros2_ws
colcon build --symlink-install --event-handlers console_direct+ --parallel-workers $(( $(nproc) * 75 / 100 )) 2>&1 | tee build.log
bash ./postBuild.sh
```

The shebang needs to be modified for venv purposes on these files:
```
ros2_ws/src/sands/sands/control_data.py
ros2_ws/src/sands/sands/control_node.py
ros2_ws/src/sands/sands/control_report.py
ros2_ws/src/sands/sands/control_status.py
ros2_ws/src/video_receiver/video_receiver/receive_rtsp_gst.py
```

## Create a development environment
The following steps setup a virtual environment for Python.  This is done because if these python packages were installed outside of the virtual environment you will likely experience issues building the ROS source code.  Perhaps with further testing this can be avoided, until then though, this.

During development it was noticed that the gi module becomes very tricky to load, even in a virtual environment.  In the end the solution I found was to use --system-site-packages and let the gi requirement be handled by apt.

Again, this is not pretty and definitely undesirable but with time it can be fixed and made better.
```
python3 -m venv --system-site-packages ~/testEnv
source ~/testEnv/bin/activate
```

## RustDesk steps go here
Depending on the environment RustDesk is authorized for use as of now for non-CUI situations.  Make sure to discuss with leadership your chosen implementation for a remote desktop before proceeding.  If RustDesk is not an acceptable solution then you should refer to the VNC steps as listed in `SETUP.md`:
```
Steps will go here

Things to be added are Wireguard and some syntax

During testing it exceeded expectations
```

## Steps for Ultralytics
Ensuring you have sourced your virtual environment, proceed:
```
pip install ultralytics[export]
yolo version
pip uninstall -y torch torchvision
```

## Setting up Torch
What follows below is the "closest" we can get to Jetson parity:
```
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/test/cu124
```

## Other Python dependencies
```
pip install pipdeptree==2.2.0 numpy==1.23.5 pymavlink==2.4.48 transforms3d pygeodesy geographiclib pygame
```

## Dependency concerns
Numpy at 1.23.5 seems to work for the most part but there are definite conflicts going on:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
treescope 0.1.9 requires numpy>=1.25.2, but you have numpy 1.23.5 which is incompatible.
tensorflow 2.19.0 requires numpy<2.2.0,>=1.26.0, but you have numpy 1.23.5 which is incompatible.
opencv-python 4.12.0.88 requires numpy<2.3.0,>=2; python_version >= "3.9", but you have numpy 1.23.5 which is incompatible.
jaxlib 0.6.2 requires numpy>=1.26, but you have numpy 1.23.5 which is incompatible.
jax 0.6.2 requires numpy>=1.26, but you have numpy 1.23.5 which is incompatible.
chex 0.1.89 requires numpy>=1.24.1, but you have numpy 1.23.5 which is incompatible.
Successfully installed numpy-1.23.5
```

## Obtain the geographiclib_datasets
```
wget https://raw.githubusercontent.com/mavlink/mavros/ros2/mavros/scripts/install_geographiclib_datasets.sh
chmod +x install_geographiclib_datasets.sh
sudo ./install_geographiclib_datasets.sh
rm -f ./install_geographiclib_datasets.sh
```

## Testing the environment
```bash
python3 -c "import torch; import torchvision; import ultralytics; print(f'\npytorch: {torch.__version__}'); print(f'torchvision: {torchvision.__version__}'); print(f'CUDA Available? {torch.cuda.is_available()}'); print(f'cuda devices: {torch.cuda.device_count()}' if torch.cuda.is_available() else ''); ultralytics.checks();"
```

The above should yield something that resembles:
```
pytorch: 2.4.1+cu124
torchvision: 0.19.1+cu124
CUDA Available? True
cuda devices: 1
Ultralytics 8.3.168 🚀 Python-3.10.12 torch-2.4.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3050 Ti Laptop GPU, 3902MiB)
Setup complete ✅ (16 CPUs, 15.3 GB RAM, 136.0/464.3 GB disk)
```

## Sourcing the environment
There are a few ways to go about this as far as .bashrc and whatnot.  For the purposes of this tutorial what follows is the raw sourcing needed, implement as you see fit into your developer environment:
```
source ~/internal-fw-companion-ros2/ros2_ws/install/setup.bash
source ~/testEnv/bin/activate
```

## Sands ENV
After making any changes to `internal-sim` remember to:
```
cd ~/internal-sim
cd gazebo_plugin
mkdir build
cd build
make clean
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j$(( $(nproc) * 75 / 100 ))
```

After making any changes to `nvac` remember to:
```
cd ~/nvac
./waf configure --board sitl
./waf copter
```

Open a shell for Gazebo and do:
```
## Ignore if not On-Demand for Nvidia
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

export PATH=/home/dev/nvac/Tools/autotest:$PATH
source ~/nvac/Tools/completion/completion.bash
export GZ_VERSION=harmonic
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/internal-sim/gazebo_plugin/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}
export GZ_SIM_RESOURCE_PATH=$HOME/internal-sim/gazebo_plugin/models:$HOME/internal-sim/gazebo_plugin/worlds:${GZ_SIM_RESOURCE_PATH}
cd ~
gz sim -v4 -r custom_world.sdf
```

After Gazbeo renders the vehicle open a shell for SITL - quad0 for CUI GCS :
```
export PATH=/home/dev/nvac/Tools/autotest:$PATH
source ~/nvac/Tools/completion/completion.bash
export GZ_VERSION=harmonic
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/internal-sim/gazebo_plugin/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}
export GZ_SIM_RESOURCE_PATH=$HOME/internal-sim/gazebo_plugin/models:$HOME/internal-sim/gazebo_plugin/worlds:${GZ_SIM_RESOURCE_PATH}
cd ~/nvac

## tic
Tools/autotest/sim_vehicle.py -v ArduCopter -f tic --console -I0 --model JSON --out=tcpin:0.0.0.0:10000 --mavproxy-args="--source-system=101"

## tac
Tools/autotest/sim_vehicle.py -v ArduCopter -f tac --console -I1 --model JSON --out=tcpin:0.0.0.0:10010 --mavproxy-args="--source-system=102"

## toe
Tools/autotest/sim_vehicle.py -v ArduCopter -f toe --console -I2 --model JSON --out=tcpin:0.0.0.0:10020 --mavproxy-args="--source-system=103"

```

Open a new shell for serial sanity -- Perhaps sim_vehicle can be modified for all in one?
```
mavproxy.py --master=tcp:127.0.0.1:5760 --out=udp:127.0.0.1:14555 --streamrate=-1 --source-system=104

mavproxy.py --master=tcp:127.0.0.1:5770 --out=udp:127.0.0.1:14556 --streamrate=-1 --source-system=105

mavproxy.py --master=tcp:127.0.0.1:5780 --out=udp:127.0.0.1:14557 --streamrate=-1 --source-system=106
```

Run the companion for Arduino float:
```
cd ~/Desktop/marcoDownloads
source pyEnv/bin/activate
python3 ./companion_tic_mcu_driver.py

cd ~/Desktop/marcoDownloads
source pyEnv/bin/activate
python3 ./companion_tac_mcu_driver.py

cd ~/Desktop/marcoDownloads
source pyEnv/bin/activate
python3 ./companion_toe_mcu_driver.py
```

Run mavros:
```
source ~/internal-fw-companion-ros2/ros2_ws/install/setup.bash
source ~/testEnv/bin/activate
ros2 launch mavros node.launch fcu_url:=udp://:14555@:14555\
 gcs_url:="\"\""\
 namespace:=/mavros/uas1\
 tgt_system:=1\
 tgt_component:=1\
 respawn_mavros:=true\
 config_yaml:=/home/dev/internal-fw-companion-ros2/ros2_ws/src/sands/config/naviator_config.yaml\
 pluginlists_yaml:=/home/dev/internal-fw-companion-ros2/ros2_ws/src/sands/config/naviator_pluginlists.yaml
```

Open a new shell to interact with MAVROS topics/services ensuring the appropriate MAVLink rates:
```
source ~/internal-fw-companion-ros2/ros2_ws/install/setup.bash
source ~/testEnv/bin/activate
ros2 run sands control_status
```

Interact with MAVROS topics and output the cummulative data in a custom topic:
```
source ~/internal-fw-companion-ros2/ros2_ws/install/setup.bash
source ~/testEnv/bin/activate
ros2 run sands control_data
```

Set up the video_receiver:
```
source ~/internal-fw-companion-ros2/ros2_ws/install/setup.bash
source ~/testEnv/bin/activate
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 run video_receiver receive_rtsp_gst --ros-args -p rtsp_url:=rtsp://127.0.0.1:8554/down_camera -p output_topic:=image
```

Set up the detector:
```
source ~/internal-fw-companion-ros2/ros2_ws/install/setup.bash
source ~/testEnv/bin/activate
ros2 run detector detect --ros-args -p input_topic:=/video_receiver/image -p detection_type:=color
ros2 run detector detect --ros-args -p input_topic:=/video_receiver/image -p detection_type:=yolo
```

Control the node:
```
source ~/internal-fw-companion-ros2/ros2_ws/install/setup.bash
source ~/testEnv/bin/activate
ros2 run sands control_node
```

Control the vehicle
```
source ~/internal-fw-companion-ros2/ros2_ws/install/setup.bash
source ~/testEnv/bin/activate

ros2 service call /sands/unlock std_srvs/srv/Trigger "{}"

ros2 service call /sands/takeoff interfaces_subuas/srv/DoTakeoff "{altitude: 5}"

ros2 service call /sands/relative interfaces_subuas/srv/DoRelative "{x: 3.5, y: 12, z: 0}"

ros2 service call /sands/waypoint interfaces_subuas/srv/DoWaypoint "{lat: 30.1769685, lon: -85.7352026, alt: 5}"

ros2 service call /sands/splash interfaces_subuas/srv/DoSplash

ros2 service call /sands/lars_hold interfaces_subuas/srv/DoLaunchRecoveryHold "{depth: 2.3, duration: 10}"

ros2 service call /sands/lars_recover interfaces_subuas/srv/DoLaunchRecoveryRecover "{estimated_depth: 2.3}"

ros2 service call /sands/land interfaces_subuas/srv/DoLand

ros2 service call /sands/transition interfaces_subuas/srv/DoTransition "{altitude: 5.0}"

ros2 service call /sands/waypoint interfaces_subuas/srv/DoWaypoint "{lat: 30.176900345051163, lon: -85.735265598682, alt: 10, alt_type: 'rel'}"

ros2 service call /sands/waypoint interfaces_subuas/srv/DoWaypoint "{lat: 30.1773108146325, lon: -85.73531389606717, alt: 15, heading: 180}"
```

## Reports
ros2 topic echo /sands/control_status/report


## todo
- made it to the beacon, kept trying to go low, 2.3?  play with -
- jetson time!



~~~~~~~~~~~~~~ JETSON BELOW ~~~~~~~~~~~~~~~

## When all hope is lost cam wise:
# sudo apt install --reinstall nvidia-l4t-kernel nvidia-l4t-kernel-dtbs nvidia-l4t-kernel-headers
# nvidia-l4t-jetson-io


sudo apt update
sudo apt -y install nvidia-jetpack
sudo apt -y dist-upgrade
sudo apt -y install tmux python3.10-venv v4l-utils
reboot

sudo apt install nano tree firefox vlc ffmpeg python3-pip
python3 -m pip install --upgrade pip
python3 -m pip install numpy==1.26.1
sudo apt -y install curl
pip install pipdeptree==2.2.0
sudo apt -y remove modemmanager

# skip vnc steps
# skip hist changes
# skip hostname changes
# set power mode to 25W
# skip global github naming
# skip .git access

touch ~/.gitignore_global
echo "COLCON_IGNORE" >> ~/.gitignore_global
git config --global core.excludesfile ~/.gitignore_global
git config --global core.excludesfile

# skip nvidia setup as nvidia-jetpack covers it
# skip cudnn for the same reasons

sudo apt install -y docker.io
sudo usermod -aG docker $USER
newgrp docker
docker run --rm --runtime=nvidia nvcr.io/nvidia/l4t-ml:r36.2.0-py3 bash -c "echo 'GPU OK on Jetson';"
docker rmi -f $(docker images -aq)


docker build -f Dockerfile.baseline -t baseline.docker .


Now you can go build
# syntax=docker/dockerfile:1.4
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y git openssh-client

# Use SSH agent for private repo clone
RUN --mount=type=ssh git clone git@github.com:your/private-repo.git /app

## Call f whatever you want and t is target
docker build -f Dockerfile.baseline -t baseline.docker .

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






## ensure you're on the correct branch before doing this
rsync -av --exclude='.git' ~/nvac subuas@192.168.100.100:/home/subuas/dBuild/
rsync -av --exclude='.git' ~/internal-fw-companion-ros2 subuas@192.168.100.100:/home/subuas/dBuild/




## Camera work below ##

https://forums.developer.nvidia.com/t/picamera3-imx708-and-j40header-config-for-pwm-servo-on-jetpack6-solution/295489




git clone https://gitlab.freedesktop.org/gstreamer/gst-rtsp-server.git
cd gst-rtsp-server
meson build
ninja -C build

mkdir mtx && cd mtx
wget https://github.com/bluenviron/mediamtx/releases/download/v1.13.1/mediamtx_v1.13.1_linux_arm64.tar.gz
tar zxf mediamtx_v1.13.1_linux_arm64.tar.gz 

sudo apt install python3-opencv


cd ~
wget https://github.com/ArduCAM/MIPI_Camera/releases/download/v0.0.3/install_full.sh
chmod +x install_full.sh
./install_full.sh -m imx708

Edit Pin Headers using arducam io tool
sudo /opt/arducam/jetson-io/jetson-io.py
IMPORTANT, do not click save and reboot after making changes, instead click the option to export as device tree overlay. Copy the overlay filepath it gives you to your clipboard

Edit /boot/extlinux/extlinux.conf to append overlay
At the bottom of the file you should see where the IMX708 overlay is. Add a comma and paste the additional overlay

Reboot

Notes:
This will create the folder /boot/arducam that contains the a new kernel image with the drivers, as well as update the /boot/extlinux/extlinux.conf to point to the new kernel image and add the overlay for the IMX708 dtbo file in /boot/arducam/dtb . After rebooting, the IMX708 sensor should be recognized and working.
However any exisiting pin configurations were set back to default for me.

2. Update J40 Pin Headers:
This is the part that I struggled with the most and it drove me insane. I tried updating the pin headers using sudo /opt/nvidia/jetson-io/jetson-io.py but that would always cause the orin nano to no longer recognize the IMX708 sensor which put me back at square one. After digging around in the root file system, I realized that arducam also installs a seperate set of tools to manage io config, which are found at /opt/arducam/jetson-io/jetson-io.py . The nvidia io tool seemed to rewrite the /boot/extlinux/extlinux.conf to point back to the old kernel image under /boot/Image, as opposed to the new image at /boot/arducam/Image . However the arducam io tool was also throwing an error when I saved and rebooted (seemed to remove the IMX708 overlay from the extlinux.conf file). Long Story short, my way around it was to use the /opt/arducam/jetson-io/jetson-io.py to make the pin changes, and save those changes to a new overlay in /boot/arducam/, and then manually edited the /boot/extlinux/extlinux.conf file to append the new overlay after the IMX708 overlay. After saving the extlinux.conf file and rebooting, both the camera and pin changes were working.

## Camera work above ##



## Docker and SSH problems with Github
```
mkdir -p ~/.docker/cli-plugins
curl -L https://github.com/docker/buildx/releases/download/v0.13.1/buildx-v0.13.1.linux-arm64 -o ~/.docker/cli-plugins/docker-buildx
chmod +x ~/.docker/cli-plugins/docker-buildx
```

ssh-keyscan github.com >> ~/.ssh/known_hosts

pkill ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa


# Ignore below until SSH perms are figured out, hence our COPY above #

DOCKER_BUILDKIT=1 docker build --ssh default -f Dockerfile.baseline -t baseline.docker .


# SSH key security
# ssh-keyscan github.com > github_hostkey
# ssh-keygen -lf github_hostkey
# github.com ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCj7ndNxQowgcQnjshcLrqPEiiphnt+VTTvDP6mHBL9j1aNUkY4Ue1gvwnGLVlOhGeYrnZaMgRK6+PKCUXaDbC7qtbW8gIkhL7aGCsOr/C56SJMy/BCZfxd1nWzAOxSDPgVsmerOBYfNqltV9/hWCqBywINIR+5dIg6JTJ72pcEpEjcYgXkE2YEFXV1JHnsKgbLWNlhScqb2UmyRkQyytRLtL+38TGxkxCflmO+5Z8CSSNY7GidjMIZ7Q4zMjA2n1nGrlTDkzwDCsw+wqFPGQA179cnfGWOWRVruj16z6XyvxvjJwbz0wQZ75XK5tKSb7FNyeIEs4TT4jk+S4dhPeAUC5y+bDYirYgM4GC7uEnztnZyaVWQ7B381AK4Qdrwt51ZqExKbQpTUNn+EjqoTwvqNj4kqx5QUCI0ThS/YkOxJCXmPUWZbhjpCg56i+2aB6CmK2JGhn57K5mj0MNdBXA4/WnwH6XoPWJzK5Nyu2zB3nAZp+S5hpQs+p1vN1/wsjk=
# github.com ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBEmKSENjQEezOmxkZMy7opKgwFB9nkt5YRrYMjNuG5N87uRgg6CLrbo5wAdT/y6v0mKV0U2w0WZ2YB/++Tpockg=
# github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl
# Strict host key check with proper permissions
# RUN --mount=type=ssh ssh -T git@github.com || echo "SSH auth failed"
# RUN mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
#     echo "github.com ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCj7ndNxQowgcQnjshcLrqPEiiphnt+VTTvDP6mHBL9j1aNUkY4Ue1gvwnGLVlOhGeYrnZaMgRK6+PKCUXaDbC7qtbW8gIkhL7aGCsOr/C56SJMy/BCZfxd1nWzAOxSDPgVsmerOBYfNqltV9/hWCqBywINIR+5dIg6JTJ72pcEpEjcYgXkE2YEFXV1JHnsKgbLWNlhScqb2UmyRkQyytRLtL+38TGxkxCflmO+5Z8CSSNY7GidjMIZ7Q4zMjA2n1nGrlTDkzwDCsw+wqFPGQA179cnfGWOWRVruj16z6XyvxvjJwbz0wQZ75XK5tKSb7FNyeIEs4TT4jk+S4dhPeAUC5y+bDYirYgM4GC7uEnztnZyaVWQ7B381AK4Qdrwt51ZqExKbQpTUNn+EjqoTwvqNj4kqx5QUCI0ThS/YkOxJCXmPUWZbhjpCg56i+2aB6CmK2JGhn57K5mj0MNdBXA4/WnwH6XoPWJzK5Nyu2zB3nAZp+S5hpQs+p1vN1/wsjk=" > /tmp/expected_hostkey && \
#     echo "github.com ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBEmKSENjQEezOmxkZMy7opKgwFB9nkt5YRrYMjNuG5N87uRgg6CLrbo5wAdT/y6v0mKV0U2w0WZ2YB/++Tpockg=" >> /tmp/expected_hostkey && \
#     echo "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl" >> /tmp/expected_hostkey && \
#     ssh-keyscan github.com > /tmp/actual_hostkey && \
#     sort /tmp/expected_hostkey > /tmp/expected_sorted && \
#     sort /tmp/actual_hostkey > /tmp/actual_sorted && \
#     diff -q /tmp/expected_sorted /tmp/actual_sorted || \
#     (echo "SSH host key mismatch! ABORTING BUILD." && exit 1) && \
#     mv /tmp/actual_hostkey ~/.ssh/known_hosts && \
#     chmod 600 ~/.ssh/known_hosts

# RUN mkdir -p ~/.ssh && chmod 700 ~/.ssh
# RUN ssh-keyscan github.com >> ~/.ssh/known_hosts && chmod 0600 ~/.ssh/known_hosts
# RUN --mount=type=ssh ssh -T git@github.com || echo "SSH auth failed"

## nvac
# RUN --mount=type=ssh ssh -vT git@github.com || echo "SSH FAILED"
# RUN --mount=type=ssh env | grep SSH_AUTH_SOCK || echo "SSH_AUTH_SOCK missing"
# RUN --mount=type=ssh git clone --recurse-submodules --branch Naviator-4.2.3-sim-dev git@github.com:subuas-llc/fw-nv-ap-ardupilot-legacy.git ./nvac;

# RUN --mount=type=ssh ssh-keyscan github.com >> ~/.ssh/known_hosts && chmod 0600 ~/.ssh/known_hosts && git clone --recurse-submodules --branch Naviator-4.2.3-sim-dev git@github.com:subuas-llc/fw-nv-ap-ardupilot-legacy.git ./nvac;
    # cd nvac;\
# RUN --mount=type=ssh git reset --hard 5eec604;\
# RUN --mount=type=ssh git submodule update --init --recursive
# Ignore above until SSH perms are figured out #


## Multi
https://github.com/subuas-llc/internal-sim/tree/internal-sim-testing-dev                                           356c68c (most recent)
https://github.com/subuas-llc/internal-fw-companion-ros2/tree/sands-b-drone-testing-merge_d5467f3-multiVehicle-dev c529fd5 (most recent)
https://github.com/subuas-llc/fw-nv-ap-ardupilot-legacy/tree/Naviator-4.2.3-sim-multiVehicle-dev                   d5c1a6b (most recent)

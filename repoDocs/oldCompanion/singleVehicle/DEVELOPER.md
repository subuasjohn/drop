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
sudo apt -y install software-properties-common curl git
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt update
sudo apt -y install arduino \
                    build-essential \
                    gcc-arm-none-eabi \
                    geographiclib-tools \
                    gir1.2-gst-rtsp-server-1.0 \
                    gir1.2-gtk-3.0 \
                    git \
                    git-lfs \
                    gstreamer1.0-gl \
                    gstreamer1.0-libav \
                    gstreamer1.0-plugins-bad \
                    gstreamer1.0-plugins-base-apps \
                    gstreamer1.0-plugins-ugly \
                    gstreamer1.0-tools \
                    ignition-fortress \
                    libcairo2-dev \
                    libcusparse11 \
                    libgirepository1.0-dev \
                    libgstreamer1.0-dev \
                    libgstreamer-plugins-base1.0-dev \
                    libgstrtspserver-1.0-dev \
                    libgz-sim8-dev \
                    libopencv-dev \
                    locales \
                    meson \
                    ninja-build \
                    python3-cffi \
                    python3-colcon-common-extensions \
                    python3-dev \
                    python3-gdal \
                    python3-pip \
                    python3-pygame \
                    rapidjson-dev \
                    ros-dev-tools \
                    ros-humble-angles \
                    ros-humble-desktop \
                    ros-humble-geographic-msgs \
                    ros-humble-mavros \
                    ros-humble-mavros-msgs \
                    ros-humble-ros-gz-bridge \
                    ros-humble-rmw-cyclonedds-cpp \
                    ros-humble-rqt-tf-tree \
                    ros-humble-tf-transformations \
                    terminator \
                    tmux \
                    ubuntu-drivers-common
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

## compiz if wanted
```
sudo apt -y install compiz-mate compizconfig-settings-manager compiz-plugins
```

## rosdep
sudo mkdir -p /etc/ros/rosdep/sources.list.d/
sudo bash -c 'wget https://raw.githubusercontent.com/osrf/osrf-rosdep/master/gz/00-gazebo.list -O /etc/ros/rosdep/sources.list.d/00-gazebo.list'
rosdep update
rosdep resolve gz-harmonic

## Env
echo '' >> ~/.bashrc
echo '' >> ~/.bashrc
echo '## CUSTOM' >> ~/.bashrc
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
echo "export RCUTILS_COLORIZED_OUTPUT=1" >> ~/.bashrc
echo "export LIBGL_ALWAYS_SOFTWARE=1" >> ~/.bashrc
echo "export ROS_DOMAIN_ID="$(( $RANDOM % 100 + 1 )) >> ~/.bashrc
echo "export ROS_LOCALHOST_ONLY=1" >> ~/.bashrc
echo "source /usr/share/colcon_cd/function/colcon_cd.sh" >> ~/.bashrc
echo "export _colcon_cd_root=/opt/ros/humble/" >> ~/.bashrc
echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ~/.bashrc

## Setup ArduPilot
cd ~
git clone https://github.com/ArduPilot/ardupilot.git --recurse-submodules
cd ardupilot
Tools/environment_install/install-prereqs-ubuntu.sh -y

## ArduPilot cleanup
In ~/.profile find "export PATH=/home/dev/ardupilot/Tools/autotest:$PATH" and delete

## Setup ardupilot_gazebo
```
cd ~
git clone https://github.com/ArduPilot/ardupilot_gazebo.git
export GZ_VERSION=harmonic
cd ardupilot_gazebo
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j4
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

Currently experimenting with nvidia-driver-575
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
For whatever reason there is an issue with the upgrade process as of September 2025.

dpkg: error processing archive /tmp/apt-dpkg-install-6auGMF/03-libnvidia-extra-535_535.261.03-0ubuntu1_amd64.deb (--unpack):
 trying to overwrite '/usr/lib/x86_64-linux-gnu/libnvidia-api.so.1', which is also in package libnvidia-gl-535:amd64 535.247.01-0ubuntu0.22.04.1
Errors were encountered while processing:
 /tmp/apt-dpkg-install-S5WcYr/03-libnvidia-extra-535_535.261.03-0ubuntu1_amd64.deb
 /tmp/apt-dpkg-install-uzDA5G/04-libnvidia-extra-535_535.261.03-0ubuntu1_amd64.deb

To work around this proceed as follows:
```
sudo apt -yf install
sudo apt -y autoremove
reboot
```

## OBS Studio
```
sudo add-apt-repository ppa:obsproject/obs-studio
sudo apt update
sudo apt install obs-studio

https://github.com/iamscottxu/obs-rtspserver/releases/download/v3.1.0/obs-rtspserver-v3.1.0-linux-qt6.deb
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
./test
rm -f test test.cu
```

## Steps for cudnn
Below are the steps for cudnn.  At the end of these steps you should see output from ldconfig
```
sudo apt install libcudnn8 libcudnn8-dev
sudo ldconfig
ldconfig -p | grep cudnn
```

## Testing the ROS src
Depending on a system layout a virtual environment may be required.  `ros2_ws/postBuild.sh` was created to handle keeping the source code clean while allowing a user to quickly add, update or remove shebangs as needed for Python purposes.

If you are using this outside of a virtual environment you can disregard the postBuild.sh portion of these instructions:
```
python3 -m pip install setuptools==65.5.1
cd ~/companion/ros2_ws
colcon build --symlink-install --event-handlers console_direct+ --parallel-workers $(( $(nproc) * 75 / 100 )) 2>&1 | tee build.log
colcon build --packages-select detector interfaces_subuas libmavconn mavlink mavros_extras mavros_msgs sands video_receiver --symlink-install --event-handlers console_direct+ --parallel-workers $(( $(nproc) * 75 / 100 )) 2>&1 | tee build.log
bash ./postBuild.sh
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
python3 -m pip install --upgrade pip wheel
pip install pipdeptree==2.2.0 numpy==1.23.5 pymavlink==2.4.48 transforms3d pygeodesy geographiclib pygame tqdm
python3 -m pip install setuptools==65.5.1
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
source ~/companion/ros2_ws/install/setup.bash
source ~/testEnv/bin/activate
```

## Selector app
```
sudo apt install nvidia-prime
```

## Final pips
pymavlink==2.4.17 was previously seen as the way to go but for source matching we now pair to mavproxy
```
python3 -m pip install setuptools==65.5.1
pip install numpy==1.23.5
pip3 install --user --no-cache-dir "pymavlink==2.4.40" "MAVProxy==1.8.17"
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

export PATH=/home/$(whoami)/nvac/Tools/autotest:$PATH
source ~/nvac/Tools/completion/completion.bash
export GZ_VERSION=harmonic
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/internal-sim/gazebo_plugin/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}
export GZ_SIM_RESOURCE_PATH=$HOME/internal-sim/gazebo_plugin/models:$HOME/internal-sim/gazebo_plugin/worlds:${GZ_SIM_RESOURCE_PATH}
cd ~
gz sim -v4 -r custom_world.sdf
```

After Gazbeo renders the vehicle open a shell for SITL - .173 for Non-CUI GCS and quad0 for CUI GCS :
```
export PATH=/home/$(whoami)/nvac/Tools/autotest:$PATH
source ~/nvac/Tools/completion/completion.bash
cd ~/nvac
Tools/autotest/sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --console --out 127.0.0.1:14550 --mavproxy-args="--source-system=101"
```

Open a new shell for the 2nd mavproxy instance:
```
mavproxy.py --master=tcp:127.0.0.1:5762 --out=udp:127.0.0.1:14555 --streamrate=-1 --source-system=102
```

Run the companion for Arduino float:
```
cd ~/Desktop/marcoDownloads
source pyEnv/bin/activate
python3 ./companion_mcu_driver.py
```

Run mavros:
```
source ~/companion/ros2_ws/install/setup.bash
source ~/testEnv/bin/activate
ros2 launch mavros node.launch \
    fcu_url:=udp://:14555@:14555 \
    gcs_url:=udp://@ \
    namespace:=/mavros/uas1 \
    tgt_system:=1 \
    tgt_component:=1 \
    respawn_mavros:=true \
    config_yaml:=/home/dev/companion/ros2_ws/src/sands/config/naviator_config.yaml \
    pluginlists_yaml:=/home/dev/companion/ros2_ws/src/sands/config/naviator_pluginlists.yaml
```

Open a new shell to interact with MAVROS topics/services ensuring the appropriate MAVLink rates:
```
source ~/companion/ros2_ws/install/setup.bash
#source ~/testEnv/bin/activate
ros2 run sands control_status
```

Interact with MAVROS topics and output the cummulative data in a custom topic:
```
source ~/companion/ros2_ws/install/setup.bash
#source ~/testEnv/bin/activate
ros2 run sands control_data
```

Set up the video_receiver:
```
source ~/companion/ros2_ws/install/setup.bash
#source ~/testEnv/bin/activate
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 run video_receiver receive_rtsp_gst --ros-args -p rtsp_url:=rtsp://127.0.0.1:8554/down_camera -p output_topic:=image
```

Set up the detector:
```
source ~/companion/ros2_ws/install/setup.bash
#source ~/testEnv/bin/activate
ros2 run detector detect --ros-args -p input_topic:=/video_receiver/image -p detection_type:=color

ros2 run detector detect --ros-args -p input_topic:=/video_receiver/image -p detection_type:=yolo
```

Control the node:
```
source ~/companion/ros2_ws/install/setup.bash
#source ~/testEnv/bin/activate
ros2 run sands control_node
```

Control the vehicle
```
source ~/companion/ros2_ws/install/setup.bash
#source ~/testEnv/bin/activate

ros2 service call /sands/unlock std_srvs/srv/Trigger "{}"

ros2 service call /sands/relative interfaces_subuas/srv/DoRelative "{x: 3.5, y: 12, z: 0}"
ros2 service call /sands/waypoint interfaces_subuas/srv/DoWaypoint "{lat: 30.1769685, lon: -85.7352026, alt: 5}"
ros2 service call /sands/splash interfaces_subuas/srv/DoSplash
ros2 service call /sands/lars_hold interfaces_subuas/srv/DoLaunchRecoveryHold "{depth: 2.3, duration: 10}"

ros2 service call /sands/lars_recover interfaces_subuas/srv/DoLaunchRecoveryRecover "{estimated_depth: 2.3}"

ros2 service call /sands/land interfaces_subuas/srv/DoLand
ros2 service call /sands/transition interfaces_subuas/srv/DoTransition "{altitude: 5.0}"
ros2 service call /sands/waypoint interfaces_subuas/srv/DoWaypoint "{lat: 30.176900345051163, lon: -85.735265598682, alt: 10, alt_type: 'rel'}"
ros2 service call /sands/waypoint interfaces_subuas/srv/DoWaypoint "{lat: 30.1773108146325, lon: -85.73531389606717, alt: 15, heading: 180}"


ros2 service call /sands/takeoff interfaces_subuas/srv/DoTakeoff "{altitude: 5}"


ros2 service call /detector/enable std_srvs/srv/SetBool "{data: false}"

```

## Reports
ros2 topic echo /sands/control_status/report

## Single -> Current
https://github.com/subuas-llc/internal-sim/tree/internal-sim-testing-dev                                           1d1b399
https://github.com/subuas-llc/internal-fw-companion-ros2/tree/sands-b-drone-testing-merge_d5467f3-dev              a9dae7b (most recent)
https://github.com/subuas-llc/fw-nv-ap-ardupilot-legacy/tree/Naviator-4.2.3-sim-dev                                5eec604
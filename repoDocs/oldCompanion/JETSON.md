# Setting up your Jetson after flashing
After you have followed the steps in FLASHING.md the below steps will get your Jetson setup as a Companion Computer.

## Sync your clocks
```
sudo -s
echo 'nameserver 8.8.8.8' > /etc/resolv.conf
systemctl restart systemd-timesyncd
timedatectl
```

## Setting up for CUDA.
```
sudo apt update
sudo apt -y remove modemmanager
sudo apt -y dist-upgrade
sudo apt -y install nvidia-jetpack build-essential
sudo ln -s /usr/local/cuda-12.6/nvvm/bin/cicc /usr/local/cuda-12.6/bin/cicc
echo 'export PATH=/usr/local/cuda/bin:$PATH' | sudo tee /etc/profile.d/cuda.sh
sudo chmod +x /etc/profile.d/cuda.sh
sudo reboot
```

## Testing CUDA
The goal is that you see `Hello from CUDA kernel!`
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
nvcc -o test test.cu
./test
rm -rf test test.cu
```

## Setting up a Docker environment
```
sudo apt -y purge docker-ce docker-ce-cli containerd.io
sudo apt -y install docker.io nvidia-container-toolkit
sudo usermod -aG docker $USER
newgrp docker
```

## Setting up Docker credentials
If doing this for a reusable image skip this part.

Obtain nvcr credentials (https://ngc.nvidia.com/signin) and then run this as your user and not `root`
```
docker login nvcr.io
$oauthtoken:<Your creds>
```

## Environment and further debs
Installing necessary .debs and such.
```
sudo apt -y install python3.10-venv \
                    curl \
                    tmux \
                    v4l-utils \
                    nano \
                    tree \
                    vlc \
                    ffmpeg \
                    python3-pip
python3 -m pip install --upgrade pip wheel
python3 -m pip install numpy==1.26.1 tqdm pygeodesy
python3 -m pip install setuptools==65.5.1
python3 -m pip install pipdeptree==2.2.0
touch ~/.gitignore_global
echo "COLCON_IGNORE" >> ~/.gitignore_global
git config --global core.excludesfile ~/.gitignore_global
git config --global core.excludesfile
sudo apt -y install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
sudo apt -y install libgstrtspserver-1.0-dev \
                    gstreamer1.0-tools \
                    git \
                    meson \
                    ninja-build \
                    libgstreamer1.0-dev \
                    software-properties-common
sudo add-apt-repository -y universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt update
sudo apt -y install arduino \
                    gcc-arm-none-eabi \
                    geographiclib-tools \
                    gir1.2-gtk-3.0 \
                    git-lfs \
                    gstreamer1.0-gl \
                    gstreamer1.0-libav \
                    gstreamer1.0-plugins-bad \
                    ignition-fortress \
                    libasio-dev \
                    libcairo2-dev \
                    libgirepository1.0-dev \
                    libglib2.0-0 \
                    libgstreamer1.0-dev \
                    libgstreamer-plugins-base1.0-dev \
                    libgz-sim8-dev \
                    libjpeg-dev \
                    libpng-dev \
                    ninja-build \
                    rapidjson-dev \
                    ros-dev-tools \
                    ros-humble-angles \
                    ros-humble-ros-base \
                    ros-humble-cv-bridge \
                    ros-humble-diagnostic-updater \
                    ros-humble-eigen-stl-containers \
                    ros-humble-ros-base \
                    ros-humble-geographic-msgs \
                    ros-humble-mavlink \
                    ros-humble-mavros \
                    ros-humble-mavros-msgs \
                    ros-humble-rmw-cyclonedds-cpp \
                    ros-humble-rosidl-default-generators \
                    ros-humble-ros-gz-bridge \
                    ros-humble-rqt-tf-tree \
                    ros-humble-tf-transformations \
                    python3-colcon-common-extensions \
                    python3-cffi \
                    python3-dev \
                    python3-pip \
                    python3-pygame \
                    python3-rosdep \
                    python3.10-venv \
                    tmux \
                    ubuntu-drivers-common \
                    v4l-utils \
                    zlib1g-dev
sudo mkdir -p /etc/ros/rosdep/sources.list.d/
sudo bash -c 'wget https://raw.githubusercontent.com/osrf/osrf-rosdep/master/gz/00-gazebo.list -O /etc/ros/rosdep/sources.list.d/00-gazebo.list'
rosdep update
rosdep resolve gz-harmonic
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
git clone https://github.com/ArduPilot/ardupilot.git --recurse-submodules
# cd ardupilot && Tools/environment_install/install-prereqs-ubuntu.sh -y ## Skipping due to potential issues caused on companion by this step
cd ~ && rm -rf Documents Music Pictures Public Templates Videos
wget https://raw.githubusercontent.com/mavlink/mavros/ros2/mavros/scripts/install_geographiclib_datasets.sh
chmod +x install_geographiclib_datasets.sh
sudo ./install_geographiclib_datasets.sh
rm -f ./install_geographiclib_datasets.sh
```

## ArduPilot cleanup
In `~/.profile` find `export PATH=/home/subuas/ardupilot/Tools/autotest:$PATH` and delete.

## GPU finalizations
The wheels are located [here](https://subuasllc.sharepoint.com/:u:/s/%5BSubUASEngineering%5D/Efe0LraIvbVIs2aINmXbJp0BQvUl65VQmYcvSnZt-7Q5fQ?e=WALxGS) for archive purposes.
```
pip install ultralytics --no-deps
pip uninstall -y torch torchvision torchaudio
pip install opencv-python>=4.6.0 polars ultralytics-thop>=2.0.0
wget -O torch-2.3.0-cp310-cp310-linux_aarch64.whl https://nvidia.box.com/shared/static/zvultzsmd4iuheykxy17s4l2n91ylpl8.whl
wget -O torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl https://nvidia.box.com/shared/static/9si945yrzesspmg9up4ys380lqxjylc3.whl
wget -O torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl https://nvidia.box.com/shared/static/u0ziu01c0kyji4zz3gxam79181nebylf.whl
pip install \
    torch-2.3.0-cp310-cp310-linux_aarch64.whl \
    torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl \
    torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
```

## Final pips
pymavlink==2.4.17 was previously seen as the way to go but for source matching we now pair to mavproxy
```
# pip3 install --user --no-cache-dir "pymavlink==2.4.40" "MAVProxy==1.8.17" ## Need to baseline on version
pip3 install --user --no-cache-dir pymavlink mavproxy
python3 -m pip install setuptools==65.5.1
pip install numpy==1.23.5
```

## GPU Tests
Generic testing
```
python3 -c "import torch; import torchvision; import ultralytics; print(f'\npytorch: {torch.__version__}'); print(f'torchvision: {torchvision.__version__}'); print(f'CUDA Available? {torch.cuda.is_available()}'); print(f'cuda devices: {torch.cuda.device_count()}' if torch.cuda.is_available() else ''); ultralytics.checks();"
```

Torch testing
```
python3 - <<'EOF'
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))

# quick GPU tensor test
x = torch.rand(3, 3).cuda()
y = torch.rand(3, 3).cuda()
z = torch.matmul(x, y)
print("Matrix multiply result (on GPU):\n", z)
EOF
```

Torchvision testing
```
python3 - <<'EOF'
import torch
import torchvision

print("Torch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("cuDNN enabled:", torch.backends.cudnn.enabled)

# Load a pretrained ResNet18 and push it to GPU
model = torchvision.models.resnet18(weights="DEFAULT").cuda().eval()

# Fake input image tensor
x = torch.rand(1, 3, 224, 224).cuda()

# Run inference
with torch.no_grad():
    y = model(x)

print("Output tensor shape:", y.shape)
print("Top-5 class scores:", torch.topk(y, 5).values.cpu().numpy())
EOF
rm -f /home/subuas/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
```

## Copy over SSH keys and grab nvac
```
cd ~
git clone --branch Naviator-4.2.3-sim-dev git@github.com:subuas-llc/fw-nv-ap-ardupilot-legacy.git ./nvac
cd nvac
```

At this point go ahead and figure out which commit is wanted and then perform `git reset --hard <commit here>`.
```
git submodule update --init --recursive
./waf configure --board sitl
./waf copter
```

## grab companion
```
cd ~
git clone --branch sands-b-drone-testing-merge_d5467f3-dev git@github.com:subuas-llc/internal-fw-companion-ros2.git ./internal-fw-companion-ros2
cd internal-fw-companion-ros2
```

At this point go ahead and figure out which commit is wanted and then perform `git reset --hard <commit here>`.
```
git submodule update --init --recursive
```

## Companion work for running SITL on the Jetson
This step is only needed if you intend to run SITL on the Jetson
Modify `internal-sim/gazebo_plugin/models/platform_with_ardupilot/model.sdf` and change with, then recompile:
```
      <fdm_addr>0.0.0.0</fdm_addr>
      <fdm_port_in>9002</fdm_port_in>
      <fdm_port_out>9003</fdm_port_out>
```

## Testing the ROS src
Depending on a system layout a virtual environment may be required.  `ros2_ws/postBuild.sh` was created to handle keeping the source code clean while allowing a user to quickly add, update or remove shebangs as needed for Python purposes.

If you are using this outside of a virtual environment you can disregard the postBuild.sh portion of these instructions:
```
cd ~/internal-fw-companion-ros2/ros2_ws
colcon build --symlink-install --event-handlers console_direct+ --parallel-workers $(( $(nproc) * 75 / 100 )) 2>&1 | tee build.log
bash ./postBuild.sh
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

# ROS by default
```
bash ~/companion/run_setup.sh
```

## ROS Env
Open a shell for Gazebo and do: ~~ LAPTOP
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

Open a shell for SITL: ~~ LAPTOP
```
export PATH=/home/subuas/nvac/Tools/autotest:$PATH
source ~/nvac/Tools/completion/completion.bash
export GZ_VERSION=harmonic
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/internal-sim/gazebo_plugin/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}
export GZ_SIM_RESOURCE_PATH=$HOME/internal-sim/gazebo_plugin/models:$HOME/internal-sim/gazebo_plugin/worlds:${GZ_SIM_RESOURCE_PATH}
cd ~/nvac

## Example where SITL runs on the Jetson
#Tools/autotest/sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --mavproxy-args="--source-system=101" --sitl-instance-args="--sim-address=192.168.100.101" --out 192.168.100.101:14550 --console

## Example where SITL on the local laptop reaches out to the Jetson
Tools/autotest/sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --out 192.168.168.32:14550 --console
```

Open a new shell for the 2nd mavproxy instance: ~~ LAPTOP
```
## Example where SITL runs on the Jetson
#mavproxy.py --master=tcp:127.0.0.1:5762 --out=udp:127.0.0.1:14555 --streamrate=-1 --source-system=102

## Example where SITL on the local laptop reaches out to the Jeson
mavproxy.py --master=tcp:127.0.0.1:5762 --out=udp:192.168.168.212:14555 --streamrate=-1
```

Run the companion for Arduino float: ~~ LAPTOP
This can be ran with SITL on the Jetson or SITL on the laptop
```
#pip install schedule cobs serial
cd ~/marcoDownloads
source pyEnv/bin/activate
python3 ./companion_mcu_driver.py
```

Run mavros: ~~ JETSON
```
#source ~/companion/ros2_ws/install/setup.bash
ros2 launch mavros node.launch \
    fcu_url:=udp://:14555@:14555 \
    gcs_url:=udp://@ \
    namespace:=/mavros/uas1 \
    tgt_system:=1 \
    tgt_component:=1 \
    respawn_mavros:=true \
    config_yaml:=/home/subuas/companion/ros2_ws/src/sands/config/naviator_config.yaml \
    pluginlists_yaml:=/home/subuas/companion/ros2_ws/src/sands/config/naviator_pluginlists.yaml
```

Open a new shell to interact with MAVROS topics/services ensuring the appropriate MAVLink rates: ~~ JETSON
```
#source ~/companion/ros2_ws/install/setup.bash
ros2 run sands control_status
```

Interact with MAVROS topics and output the cummulative data in a custom topic: ~~ JETSON
```
#source ~/companion/ros2_ws/install/setup.bash
ros2 run sands control_data
```

Set up the video_receiver: ~~ JETSON
For the RTSP we can point at wherever Gazebo or a real stream is running
```
#source ~/companion/ros2_ws/install/setup.bash
#export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
#ros2 run video_receiver receive_rtsp_gst --ros-args -p rtsp_url:=rtsp://127.0.0.1:8554/
#ros2 run video_receiver receive_rtsp_gst --ros-args -p rtsp_url:=rtsp://192.168.100.101:8554/down_camera -p output_topic:=image
ros2 run video_receiver receive_rtsp_gst --ros-args -p rtsp_url:=rtsp://root:root@192.168.168.222:554/cam0_0 -p output_topic:=image
```

Set up the detector: ~~ JETSON
```
#source ~/companion/ros2_ws/install/setup.bash
ros2 run detector detect --ros-args -p input_topic:=/video_receiver/image -p detection_type:=color
ros2 run detector detect --ros-args -p input_topic:=/video_receiver/image -p detection_type:=yolo
```

Control the node: ~~ JETSON
~ concept needed for `src/sands/sands/params.json`
~/companion ^ solves...
```
#source ~/companion/ros2_ws/install/setup.bash
ros2 run sands control_node
```

Control the vehicle ~~ JETSON
```
#source ~/companion/ros2_ws/install/setup.bash

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



## future work notes below  ~~ Not useful for directional purposes at this time
- Check for this from developer
                  libcusparselt0\ libcusparselt-dev 
- Think on copying profile and bashrc from subuas to root
- Chew on this for anything video_receiver related

## video_receiver tracing below
sudo apt -y install gstreamer1.0-tools

sudo apt -y install gir1.2-gst-rtsp-server-1.0

sudo apt -y install ros-humble-ros-gz-bridge

sudo apt -y install python3-rosbag

sudo apt -y install ros-humble-rosgraph-msgs

sudo apt -y install ros-humble-rqt-image-view

sudo apt -y install ros-humble-ros-gz


- When all hope is lost cam wise on the devboard you can do:
```
sudo apt install --reinstall nvidia-l4t-kernel nvidia-l4t-kernel-dtbs nvidia-l4t-kernel-headers
sudo apt install nvidia-l4t-jetson-io
```
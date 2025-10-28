#!/bin/bash

TMPDIR="/tmp/shell_tabs_example"
mkdir -p "$TMPDIR"

## Sands ENV
script1="$TMPDIR/shell1.sh"
cat > "$script1" <<'EOF'
#!/bin/bash
echo -ne "\033]0;Gazebo\007"
export PATH=/home/dev/nvac/Tools/autotest:$PATH
source ~/nvac/Tools/completion/completion.bash
export GZ_VERSION=harmonic
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/internal-sim/gazebo_plugin/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}
export GZ_SIM_RESOURCE_PATH=$HOME/internal-sim/gazebo_plugin/models:$HOME/internal-sim/gazebo_plugin/worlds:${GZ_SIM_RESOURCE_PATH}
cd ~
gz sim -v4 -r custom_world.sdf
exec bash
EOF

## SITL
script2="$TMPDIR/shell2.sh"
cat > "$script2" <<'EOF'
#!/bin/bash
echo -ne "\033]0;SITL\007"
export PATH=/home/dev/nvac/Tools/autotest:$PATH
source ~/nvac/Tools/completion/completion.bash
export GZ_VERSION=harmonic
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/internal-sim/gazebo_plugin/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}
export GZ_SIM_RESOURCE_PATH=$HOME/internal-sim/gazebo_plugin/models:$HOME/internal-sim/gazebo_plugin/worlds:${GZ_SIM_RESOURCE_PATH}
cd ~/nvac
Tools/autotest/sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --map --console --out 127.0.0.1:14550 --out 127.0.0.1:65535 --out 127.0.0.1:65534 --mavproxy-args="--source-system=101"
exec bash
EOF

## Mavproxy
script3="$TMPDIR/shell3.sh"
cat > "$script3" <<'EOF'
#!/bin/bash
echo -ne "\033]0;Mavproxy\007"
mavproxy.py --master=tcp:127.0.0.1:5762 --out=udp:127.0.0.1:14555 --streamrate=-1 --source-system=102
exec bash
EOF

## Arduino
script4="$TMPDIR/shell4.sh"
cat > "$script4" <<'EOF'
#!/bin/bash
echo -ne "\033]0;Arduino\007"
cd ~/Desktop/marcoDownloads
source pyEnv/bin/activate
python3 ./companion_mcu_driver.py
exec bash
EOF

## Mavros
script5="$TMPDIR/shell5.sh"
cat > "$script5" <<'EOF'
#!/bin/bash
echo -ne "\033]0;Mavros\007"
source ~/companion/ros2_ws/install/setup.bash
# source ~/testEnv/bin/activate
ros2 launch mavros node.launch fcu_url:=udp://:14555@:14555 gcs_url:="\"\"" namespace:=/mavros/uas1 tgt_system:=1 tgt_component:=1 respawn_mavros:=true config_yaml:=/home/dev/internal-fw-companion-ros2/ros2_ws/src/sands/config/naviator_config.yaml pluginlists_yaml:=/home/dev/internal-fw-companion-ros2/ros2_ws/src/sands/config/naviator_pluginlists.yaml
exec bash
EOF

## control_status
script6="$TMPDIR/shell6.sh"
cat > "$script6" <<'EOF'
#!/bin/bash
echo -ne "\033]0;control_status\007"
source ~/companion/ros2_ws/install/setup.bash
# source ~/testEnv/bin/activate
ros2 run sands control_status
exec bash
EOF

## control_data
script7="$TMPDIR/shell7.sh"
cat > "$script7" <<'EOF'
#!/bin/bash
echo -ne "\033]0;control_data\007"
source ~/companion/ros2_ws/install/setup.bash
# source ~/testEnv/bin/activate
ros2 run sands control_data
exec bash
EOF

## video_receiver
script8="$TMPDIR/shell8.sh"
cat > "$script8" <<'EOF'
#!/bin/bash
echo -ne "\033]0;video_receiver\007"
source ~/companion/ros2_ws/install/setup.bash
# source ~/testEnv/bin/activate
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 run video_receiver receive_rtsp_gst --ros-args -p rtsp_url:=rtsp://127.0.0.1:8554/down_camera -p output_topic:=image
exec bash
EOF

## detector
script9="$TMPDIR/shell9.sh"
cat > "$script9" <<'EOF'
#!/bin/bash
echo -ne "\033]0;detector\007"
source ~/companion/ros2_ws/install/setup.bash
# source ~/testEnv/bin/activate
ros2 run detector detect --ros-args -p input_topic:=/video_receiver/image -p detection_type:=color
exec bash
EOF

## control_node
script10="$TMPDIR/shell10.sh"
cat > "$script10" <<'EOF'
#!/bin/bash
echo -ne "\033]0;control_node\007"
source ~/companion/ros2_ws/install/setup.bash
# source ~/testEnv/bin/activate
ros2 run sands control_node
exec bash
EOF

## ROS2 CLI
script11="$TMPDIR/shell11.sh"
cat > "$script11" <<'EOF'
#!/bin/bash
echo -ne "\033]0;ROS2 CLI\007"
source ~/companion/ros2_ws/install/setup.bash
# source ~/testEnv/bin/activate
exec bash
EOF

# Make all scripts executable
chmod +x "$script1" "$script2" "$script3" "$script4" "$script5" "$script6" "$script7" "$script8" "$script9" "$script10" "$script11"

# I'll be back
terminator --new-tab -e "bash -c 'bash $script1; exec bash'" &
sleep 3
terminator --new-tab -e "bash -c 'bash $script2; exec bash'" &
sleep 3
terminator --new-tab -e "bash -c 'bash $script3; exec bash'" &
sleep 3
terminator --new-tab -e "bash -c 'bash $script4; exec bash'" &
sleep 3
terminator --new-tab -e "bash -c 'bash $script5; exec bash'" &
sleep 3
terminator --new-tab -e "bash -c 'bash $script6; exec bash'" &
sleep 3
terminator --new-tab -e "bash -c 'bash $script7; exec bash'" &
sleep 7
terminator --new-tab -e "bash -c 'bash $script8; exec bash'" &
sleep 3
terminator --new-tab -e "bash -c 'bash $script9; exec bash'" &
sleep 3
terminator --new-tab -e "bash -c 'bash $script10; exec bash'" &
sleep 3
terminator --new-tab -e "bash -c 'bash $script11; exec bash'"

## Converting Marco and Chris
awk '!/^#/ && NF>=4 { print $3, $4 }' <param file> > export.param

## nvac location for newer firmware
```
cd ~
git clone --recurse-submodules --branch ca_phase3 git@github.com:subuas-llc/fw-nv-ap-ardupilot.git ./nvac
cd nvac
cp ~/Downloads/gazebo-sitl-working-setup.patch .
git apply gazebo-sitl-working-setup.patch
./waf configure --board sitl
./waf copter
mkdir scripts
cd scripts
cp -aR ../libraries/AP_Scripting/Naviator/* .
mv Naviator_inactivity_alarm.lua inactivity_alarm.lua
cd ..
./waf configure --board sitl
./waf copter
```

## nvac location legacy
```
cd ~
git clone --recurse-submodules --branch Naviator-4.2.3-sim-dev git@github.com:subuas-llc/fw-nv-ap-ardupilot-legacy.git ./nvac
cd nvac
git reset --hard 5eec604   # currently working on desktop
git submodule update --init --recursive
./waf configure --board sitl
./waf copter
```

## Current commit
git rev-parse --short HEAD







Everything increases by 10 port wise for every vehicle addition
 1480  Tools/autotest/sim_vehicle.py -v ArduCopter -f toe --console -I2 --model JSON --map --console --out=tcpin:0.0.0.0:10002 --mavproxy-args="--source-system=103"
 1481  Tools/autotest/sim_vehicle.py -v ArduCopter -f toe --console -I2 --model JSON --out=tcpin:0.0.0.0:10020 --mavproxy-args="--source-system=103"
 1500  Tools/autotest/sim_vehicle.py -v ArduCopter -f tac --console -I1 --model JSON --map --console --out=tcpin:0.0.0.0:10001 --mavproxy-args="--source-system=102"
 1501  Tools/autotest/sim_vehicle.py -v ArduCopter -f tac --console -I1 --model JSON --out=tcpin:0.0.0.0:10010 --mavproxy-args="--source-system=102"
 1508  Tools/autotest/sim_vehicle.py -v ArduCopter -f tic --console -I0 --model JSON --map --console --out=tcpin:0.0.0.0:10000 --mavproxy-args="--source-system=101"
 1509  Tools/autotest/sim_vehicle.py -v ArduCopter -f tic --console -I0 --model JSON --out=tcpin:0.0.0.0:10000 --mavproxy-args="--source-system=101"
 1516  mavproxy.py --master=tcp:127.0.0.1:5760 --out=udp:127.0.0.1:14555 --streamrate=-1 --source-system=104
 1517  mavproxy.py --master=tcp:127.0.0.1:5770 --out=udp:127.0.0.1:14556 --streamrate=-1 --source-system=105
ros2 launch mavros node.launch fcu_url:=udp://:14555@:14555 gcs_url:="\"\"" namespace:=/mavros/uas1 tgt_system:=1 tgt_component:=1 respawn_mavros:=true config_yaml:=/home/dev/internal-fw-companion-ros2/ros2_ws/src/sands/config/naviator_config.yaml pluginlists_yaml:=/home/dev/internal-fw-companion-ros2/ros2_ws/src/sands/config/naviator_pluginlists.yaml


Gazebo on standalone .31:
```
<plugin name="ardupilot_plugin" filename="libArduPilotPlugin.so">
  <!-- Port settings for the machine running Gazebo -->
  <listen_addr>192.168.1.31</listen_addr>
  <fdm_port_in>9002</fdm_port_in>

  <!-- Port settings for the machine running SITL -->
  <fdm_addr>192.168.1.170</fdm_addr>
  <fdm_port_out>9003</fdm_port_out>

  <!-- control elements etc... -->

</plugin>
```


On the machine running sitl:
```
$ sim_vehicle.py \
  -v ArduCopter \
  -f gazebo-iris \
  --sitl-instance-args="--sim-address=192.168.1.31"
  --console \
  --map
```

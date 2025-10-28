Build the plugin:
```
cd ~/ardupilot_gazebo
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j4
```

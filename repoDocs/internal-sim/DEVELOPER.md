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
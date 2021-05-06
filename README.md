# About
- provides service node which return exploratory-search path 

# tested environtment
- ubuntu 18.04
- python 2.7
- ROS melodic
- gazebo 9
- keras 2.4.3
- tensorflow 2.3.0

# Node
## Service (MBM_server)
### input arguments
- navigation network (std_msgs/String)
- interest node (int32)

### return
- marginal probability of hypothetical node (std_msgs/String)

# How to run
- EST server \
``` rosrun topoexpsearch_MBM srv_MBM.py ```

- server call example \
``` call_MBM.py ```

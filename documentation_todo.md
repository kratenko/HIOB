# TODO: Documentation

## HIOB standalone
- Requirements (CUDA, python3, ..?)
- Installation -> Tobias
    - clone repository
    - setup venv
- Configuration
    - example/default configuration (with sample from releases)
        - setup environment.yaml
        - provide sensible default configuration files -> Tobias
    - custom use cases
        - add a data set
- Running/Usage
    - Command line parameters

## HIOB-ROS -> Tobias
- Additional requirements (ROS, ..?)
- Short explanation on setting up a ros workspace
- Installation with ros
    - setting up ros environment variables
    - clone packages
    - building/installing with catkin
- Running/Usage
    - starting roscore
    - additional command line parameters (--ros-subscribe and --ros-publish)
- Reference to example client
- Docker Container
    - Usage example
    ```sh
    docker run hiob-ros -e ROS_SUBSCRIBE /videoStream -e ROS_PUBLISH /hiob/object
    ```

docker run -itd \
    -v /mnt/hpfs_test/wlf/:/home/ \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /mnt/hpfs_test/:/mnt/hpfs_test/ \
    --ipc=host --privileged --network=host --shm-size=1500gb \
    --name verl_docker \
    swr.cn-north-4.myhuaweicloud.com/wlf_darkmaster/pytorch_ascend:pytorch_2.5.1-cann_8.2.rc1-py_3.10-aarch64_0827a \
    /bin/bash
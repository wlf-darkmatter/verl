# ssh root@localhost-
docker stop verl_container
docker rm verl_container

docker run -it -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /mnt:/mnt \
    -v /mnt/nv03:/tmp \
    -v /mnt/hpfs_test/wlf:/home/code \
    --privileged=true \
    --net=host \
    --shm-size=1000G \
    --name=verl_container \
    -e RANK=0 \
    -e WORLD_SIZE=16 \
    swr.cn-north-4.myhuaweicloud.com/wlf_darkmaster/x-contion-aarch64/rl_npu:verl-A3-cann82rc2-vllm0100-mcore012-torch27 bash

    # vllm 0110 的
    # swr.cn-north-4.myhuaweicloud.com/wlf_darkmaster/x-contion-aarch64/rl_npu:verl-A3-cann82rc2-vllm0110-mcore012-torch27 bash
    # swr.cn-north-4.myhuaweicloud.com/wlf_darkmaster/x-contion-aarch64/rl_npu:verl-A3-cann82rc2-vllm0110-mcore012-torch27

    # export https_proxy=172.16.2.65:3128
    # export http_proxy=172.16.2.65:3128
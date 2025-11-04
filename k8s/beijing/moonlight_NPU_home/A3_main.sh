####################################################
# 本脚本用于，在多机训练时：
# 1. 向其他节点，同步主节点修改后的数据
# 2. 多机启动docker、启动训练脚本
####################################################
# 本脚本需要提前配置好ssh免密登录，即将master的公钥传到目标服务器，指令示例如下：
# ssh-keygen -i /root/.ssh/id_rsa.pub
# ssh-copy-id -i /root/.ssh/id_rsa.pub root@90.90.122.182
# 配置完成后，即可通过  ssh root@90.90.122.182 实现免密登录
# 注意：
# 配置失败时，一般是ssh-copy-id拷贝的公钥和登录时使用的公钥不是同一个，导致免密配置失败
####################################################
# 脚本执行方式：
# chmod +x main.sh
# ./main.sh
####################################################

IPs=( # 除开master的其他节点IP
    # "90.90.122.117"
    # "90.90.122.181"
    # "90.90.122.120"
    # "90.90.122.182"
)

# 需要同步的目录。目标目录与源目录不相同的内容会被删除
DIRs=(
    "/home/q00887491/models/Moonlight-16B-A3B"
    "/home/q00887491/models/Moonlight-16B-A3B-dist"
    "/home/q00887491/datasets"
    "/home/q00887491/projects/wlf_darkmatter_verl"
    "/home/q00887491/logs/.cache/torch_extensions"
)

# 映射到docker内的路径
docker_v_dirs="-v ${DIRs[0]}:/data/models/Moonlight-16B-A3B:ro -v ${DIRs[1]}:/data/models/Moonlight-16B-A3B-dist:ro -v ${DIRs[2]}:/data/datasets:ro -v ${DIRs[3]}:/home/code/verl-gpu:ro -v ${DIRs[4]}:/root/.cache/torch_extensions -v /home/q00887491/logs:/data/logs -v /home/cann:/home/cann:ro"

# 启动脚本在docker内的路径
docker_cmd_file="/home/code/verl-gpu/k8s/beijing/moonlight_NPU_home/A3_start.sh"


echo "#################  数据同步中  ####################"
index=0
for ip in ${IPs[*]}; do
    echo -e "rsync to Node $index $ip"

    for diri in ${DIRs[*]}; do
        ssh root@$ip "mkdir -p $diri; exit" # 创建目录
        echo -e "rsync -av --delete $diri/ root@$ip:$diri/"
        rsync -av --delete $diri/ root@$ip:$diri/  # 同步数据，会删除目标目录中不一样的内容
        echo -e "\n"
    done

    ((index++))  # 索引自增
    echo -r "\n\n"
done
echo "#################  数据同步结束  ####################"

echo "#################  训练启动中  ####################"
# docker启动
DOCKER_IMAGES_ID=swr.cn-north-4.myhuaweicloud.com/wlf_darkmaster/pytorch_ascend:pytorch_2.7.1-cann_8.2.rc1-py_3.10-aarch64_0928
CONTAINER_NAME=test11

DOCKER_START_CMD="docker run --name ${CONTAINER_NAME} -itd --net=host --shm-size=500g \
    --privileged=true \
    -w /root \
   --device=/dev/davinci_manager \
   --device=/dev/hisi_hdc \
   --device /dev/devmm_svm \
   --entrypoint=bash \
   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
   -v /usr/local/dcmi:/usr/local/dcmi:ro \
   -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
   -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
   -v /usr/local/sbin:/usr/local/sbin:ro \
   -v /etc/hccn.conf:/etc/hccn.conf:ro \
   ${docker_v_dirs} \
   ${DOCKER_IMAGES_ID}"

DOCKER_RUN_CMD="docker exec ${CONTAINER_NAME} bash ${docker_cmd_file}"

echo -e "$DOCKER_START_CMD"
echo -e "$DOCKER_RUN_CMD"

echo -e "\nstart process on Master Node"
# eval "docker stop $CONTAINER_NAME; docker rm $CONTAINER_NAME"
# eval $DOCKER_START_CMD
eval $DOCKER_RUN_CMD &
sleep 10

index=0
for ip in ${IPs[*]}; do
    echo -e "\nstart process on Node $index $ip"
    # ssh root@$ip "docker stop $CONTAINER_NAME; docker rm $CONTAINER_NAME" # 停止+删除容器
    # ssh root@$ip "$DOCKER_START_CMD" # 创建容器
    ssh root@$ip "$DOCKER_RUN_CMD; exit" & # +“&”后，后台起另外一个线程运行
    ((index++))  # 索引自增
    echo "################################################"
done

# 等待后台任务完成
wait

echo "#################  训练结束  ####################"


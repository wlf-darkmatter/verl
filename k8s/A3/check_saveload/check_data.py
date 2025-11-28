import ray

"""
CURRENT_IP=172.16.2.73
export ServerPort=6666     # modify according to actual situation
export DashboardPort=8888  # modify according to actual situation
ray start --head --ray-debugger-external --port $ServerPort --dashboard-port=$DashboardPort --node-ip-address=$CURRENT_IP --dashboard-host=$CURRENT_IP --disable-usage-stats

"""
#



@ray.remote
def check():
    from pathlib import Path
    import torch
    import os

    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,8,9,10,11,12,13,14,15"


    path_dist_ckpt = Path("/mnt/hpfs_test/weight/CKPT/ckpt-DAPO-dpsk-671b-megatron-BASE-256die-npugpu-cpuoffload/global_step_1/actor/dist_ckpt/")
    list_before=[]
    list_after=[]
    for i in path_dist_ckpt.iterdir():
        print(i)
        if i.name.startswith("rank"):
            if "replace" in i.name:
                list_after.append(i)
            else:
                list_before.append(i)
    list_before.sort()
    list_after.sort()
    pass
    #breakpoint()
    for path_pt_i, path_pt_j in zip(list_before, list_after):
        pt_i = torch.load(path_pt_i, weights_only=False)
        pt_j = torch.load(path_pt_j, weights_only=False)

    #! 对比内容


if __name__ == "__main__":
    ray.init()

    ray.get(check.remote())
    pass
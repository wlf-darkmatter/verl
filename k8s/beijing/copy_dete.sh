target_name=moonlight-det-wlf
cwd=$(dirname $(realpath $0))
echo $cwd
cd $cwd

if [[ $1 != "" ]];then
    echo $1
    set -x

    new_target_dir=$target_name-$1
    rm -rf ${new_target_dir}
    \cp -r ${cwd}/$target_name ${new_target_dir}
    sed -i "s/exp_name=.*/exp_name=DAPO-MoonLight-16b-megatron-2NODES-det-$1/g" ${new_target_dir}/hw_run_dapo_deepseek_671b_megatron.sh


    sed -i "s/moonlight-16die-wlf-det/moonlight-16die-wlf-det-$1/g"  ${new_target_dir}/acjob_deepseek671b_megatron.yaml
    sed -i "s/moonlight-det-wlf/moonlight-det-wlf-$1/g"  ${new_target_dir}/acjob_deepseek671b_megatron.yaml
    diff ${new_target_dir}/acjob_deepseek671b_megatron.yaml $target_name/acjob_deepseek671b_megatron.yaml
    set +x
fi


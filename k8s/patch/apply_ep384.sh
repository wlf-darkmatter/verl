set -x

cwd=$(dirname $(realpath $0))
echo "dealing $cwd"
cd $cwd

\cp ./moe_distribute_dispatch/* /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/moe_distribute_dispatch/
echo -e "\033[32mApplied moe_distribute_dispatch done.\033[0m"
ls -lrt /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/moe_distribute_dispatch/
\cp ./moe_distribute_combine/* /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/moe_distribute_combine/
echo -e "\033[32mApplied moe_distribute_dispatch done.\033[0m"
ls -lrt /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/moe_distribute_combine/

set +x

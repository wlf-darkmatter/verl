

while true; do
    echo "=========================================="
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    free -h
    npu-smi info
    sleep 5
done

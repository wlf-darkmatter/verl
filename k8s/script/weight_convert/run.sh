cd $(dirname $0)
python hf_fp8_cast_bf16.py \
    --input-fp8-hf-path /mnt/hpfs_test/weights/dsv3-base-fp8-wlf \
    --output-bf16-hf-path /mnt/hpfs_test/weights/dsv3-base-fp8-wlf-bf16
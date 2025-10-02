## 运行方式

路径：/data01/huawei/wlf/verl/k8s/test

### 拉起
```bash
bash iter_run.sh start
```
### 查看

```bash
bash iter_run.sh check

```
### 停止
```bash
bash iter_run.sh stop

```

## 细节修改

1. 主要的文件是 `acjob_test_sleep.yaml`，运行时会自动的拷贝这个文件，并自动修改其中的 name
2. 【重要】目前默认双机打流



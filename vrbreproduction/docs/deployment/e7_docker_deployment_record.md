# E7 Docker 部署记录

## 本地关键路径

- 项目根目录：`/Users/huhu/Documents/Code/bishe/vrb`
- E7 主脚本：`/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/src/vrbreproduction/e7_object_tracking_projection.py`
- E7 Dockerfile：`/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/docker/Dockerfile.e7`
- E7 依赖清单：`/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/docker/e7-requirements.txt`
- 容器启动脚本：`/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/docker/run_e7.sh`
- annotations csv：`/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/data/annotations/epic-kitchens-100-annotations/EPIC_100_train.csv`
- HOA detections：`/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/data/P01_109.pkl`
- 正确连续帧压缩包：`/private/tmp/vrb_e7_frames_correct.tar.gz`
- SSH 私钥：`/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/docs/smoothcloud/key.pem`

## 云端关键路径

- 云端工作目录根：`/root/workspace/vrb`
- 云端项目目录：`/root/workspace/vrb/vrbreproduction`
- 云端日志目录：`/root/workspace/vrb/logs`
- 云端帧目录：`/root/workspace/vrb/vrbreproduction/data/P01_109_frames`
- 云端输出目录：`/root/workspace/vrb/vrbreproduction/outputs/100subaction-e7`
- 云端裸跑启动脚本：`/root/workspace/vrb/run_e7_bare.sh`
- 云端裸跑日志：`/root/workspace/vrb/logs/e7_bare_run.log`
- 云端裸跑 PID 文件：`/root/workspace/vrb/logs/e7_bare_run.pid`
- 云端帧修复日志：`/root/workspace/vrb/logs/e7_frames_extract.log`

## Docker 规划

- 镜像 tag：`vrb-e7:latest`
- 容器名：`vrb-e7-runner`
- 容器内工作目录：`/workspace`
- 容器内代码根：`/workspace/vrbreproduction`
- 容器内帧挂载点：`/workspace/vrbreproduction/data/P01_109_frames`
- 容器内日志目录：`/workspace/logs`
- 容器内输出目录：`/workspace/vrbreproduction/outputs/100subaction-e7`

## 数据挂载策略

- 将代码、`EPIC_100_train.csv`、`P01_109.pkl` bake 进镜像。
- 将大体积帧目录保留在云端持久盘，通过 volume 挂载到容器。
- 将日志目录和输出目录挂载到云端持久盘，避免容器删除后结果丢失。

## 当前实际执行路径

- 由于 SmoothCloud 实例内的 Docker daemon 受到宿主权限限制，`docker build` / `docker run` 无法可靠执行。
- 当前改为按平台手册支持的方式运行：
  - 数据与代码保存在 `/root/workspace/vrb`
  - 使用云端 venv：`/root/workspace/vrb/.venv`
  - 直接运行模块：`python -u -m vrbreproduction.e7_object_tracking_projection`
- 启动命令封装在：`/root/workspace/vrb/run_e7_bare.sh`
- 当前正式运行日志：`/root/workspace/vrb/logs/e7_bare_run.log`

## 帧修复确认

- 修复来源压缩包：`/root/workspace/vrb/vrb_e7_frames_correct.tar.gz`
- 修复后帧目录：`/root/workspace/vrb/vrbreproduction/data/P01_109_frames`
- 修复后关键检查：
  - `frame_0000000001.jpg` 存在
  - `frame_0000000083.jpg` 存在
  - `frame_0000016113.jpg` 存在
  - `1..16113` 连续编号缺失数：`0`

## 运行命令模板

```bash
docker run -d --name vrb-e7-runner \
  -v /root/workspace/vrb/vrbreproduction/data/P01_109_frames:/workspace/vrbreproduction/data/P01_109_frames:ro \
  -v /root/workspace/vrb/logs:/workspace/logs \
  -v /root/workspace/vrb/vrbreproduction/outputs:/workspace/vrbreproduction/outputs \
  -e E7_LOG_DIR=/workspace/logs \
  -e E7_LOG_PATH=/workspace/logs/e7_container.log \
  vrb-e7:latest
```

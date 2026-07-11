# E20 GPU job

```bash
cd /root/workspace/vrb/e20_gpu_job
python -m pip install -r requirements_server.txt
export HF_ENDPOINT=https://hf-mirror.com
wget -nc https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O /root/workspace/vrb/sam_vit_h_4b8939.pth
export SAM_CHECKPOINT=/root/workspace/vrb/sam_vit_h_4b8939.pth
python run_e20_server.py --job-dir . --out-dir /root/workspace/vrb/e20_gpu_results --device cuda --backends lama,sd --sdedit 1
tar czf /root/workspace/vrb/e20_gpu_results.tar.gz -C /root/workspace/vrb e20_gpu_results
```

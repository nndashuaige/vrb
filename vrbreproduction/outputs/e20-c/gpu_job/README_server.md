# E20-C GPU job

```bash
cd /root/workspace/vrb/e20c_gpu_job
export LD_LIBRARY_PATH=/opt/anaconda/lib
export SAM_CHECKPOINT=/root/workspace/vrb/models/sam_vit_h_4b8939.pth
export LAMA_MODEL=/root/workspace/vrb/models/big-lama.pt
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HOME=/root/workspace/vrb/cache/huggingface
export E20_SD_INPAINT_MODEL=runwayml/stable-diffusion-inpainting
export E20_SDEDIT_MODEL=runwayml/stable-diffusion-v1-5
export E20_DIFFUSERS_VARIANT=fp16
export E20_DINO_MODEL=/root/workspace/vrb/models/grounding-dino-tiny
/opt/anaconda/bin/python run_e20c_server.py --job-dir . --out-dir /root/workspace/vrb/e20c_gpu_results --device cuda --backends lama,sd --sdedit 1
tar czf /root/workspace/vrb/e20c_gpu_results.tar.gz -C /root/workspace/vrb e20c_gpu_results
```

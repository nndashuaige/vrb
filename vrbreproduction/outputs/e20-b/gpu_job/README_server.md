# E20-B GPU job

```bash
cd /root/workspace/vrb/e20b_gpu_job
export SAM_CHECKPOINT=/root/workspace/vrb/models/sam_vit_h_4b8939.pth
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HOME=/root/workspace/vrb/cache/huggingface
export E20_SD_INPAINT_MODEL=runwayml/stable-diffusion-inpainting
export E20_SDEDIT_MODEL=runwayml/stable-diffusion-v1-5
export E20_DIFFUSERS_VARIANT=fp16
python run_e20b_server.py --job-dir . --out-dir /root/workspace/vrb/e20b_gpu_results --device cuda --backends lama,sd --sdedit 1
tar czf /root/workspace/vrb/e20b_gpu_results.tar.gz -C /root/workspace/vrb e20b_gpu_results
```

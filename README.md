
**Subject:** Open-Source Options for Training Video Diffusion Models


---

### 1. **Wan 2.1 / 2.2 (Alibaba, Open Source)**

https://github.com/Wan-Video/Wan2.2


* **What it is:** State-of-the-art open video diffusion transformer with two sizes: 1.3B (runs on consumer GPUs) and 14B (requires cluster-scale compute).
* **Training:** Public repo available. LoRA fine-tuning possible on a single 24 GB GPU; full training needs large GPU clusters.
* **Strength:** Leading open performance; covers text-to-video, image-to-video, editing, and personalization tasks.

---

#### Video Diffusion Fine-Tuning (LoRA) — Wan2.2 + diffusion-pipe (Colab-ready)

> **Neden böyle?**

> * **Wan2.2**: Açık kaynak; inference odaklı resmi repo var.
> * **Eğitim (LoRA/transfer)**: **diffusion-pipe** ile video LoRA fine-tune yapılabilir.
---

##### 0) Hızlı Özet (ne yapıyoruz?)

1. Colab’de GPU’yu kontrol et
2. **Wan2.2** reposunu (inference için) ve **diffusion-pipe**’ı (eğitim için) kur
3. Küçük bir **demo dataset** oluştur (video klipler + caption .txt)
4. **LoRA config (TOML)** yaz
5. **Cache** + **Training** çalıştır
6. (Opsiyonel) LoRA’yla inference — not: LoRA’yı Wan2.2 inference’ına takmak araçtan araca değişir; burada temel akış gösterilmiştir

---

##### 1) Colab — Ortam hazırlığı

```bash
# GPU kontrolü
!nvidia-smi

# Temel paketler
!pip -q install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip -q install deepspeed accelerate datasets pillow imageio tensorboard

# diffusion-pipe (eğitim çerçevesi)
!git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe
%cd diffusion-pipe
!git submodule init && git submodule update
!pip -q install -r requirements.txt

# (Colab T4 için) flash-attn derlemeye çalışma; sürüm çakışırsa atla.
# Wan2.2 inference (ayrı klasörde)
%cd /content
!git clone https://github.com/Wan-Video/Wan2.2.git
%cd Wan2.2
!pip -q install -r requirements.txt
# TI2V-5B için ekstra gereksinimler gerekiyorsa requirements_s2v/animate'ı atlayabilirsin

# Model indir (Hugging Face CLI)
!pip -q install "huggingface_hub[cli]"
!huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
```

> **Not:** TI2V-5B, 720p@24fps’i 24 GB VRAM (4090) ile rahat çalıştırır. **T4 16 GB**’da inference için `--offload_model True --convert_model_dtype --t5_cpu` gibi bayraklarla denersin; eğitimde ise çözünürlük/FPS’i düşürüyoruz.

---

##### 2) Demo dataset (video + caption) hazırlığı

```bash
# Demo klasör
%cd /content
!mkdir -p data/my_videos

# Kendi .mp4 kliplerini /content/data/my_videos içine yükle (Colab sol panelden upload).
# Her video için aynı isimli .txt caption dosyası oluştur:
import os, textwrap, glob, pathlib
root = "/content/data/my_videos"
samples = [("clip0001.mp4","a brown dog running in a park, handheld, natural light"),
           ("clip0002.mp4","a close-up of coffee pouring into a cup, steam visible, cinematic")]

for name, cap in samples:
    p = pathlib.Path(root)/name
    # Sende video yoksa placeholder yarat (gerçekte kendi .mp4'ünü yüklemen gerekir)
    if not p.exists():
        open(p, "wb").close()
    with open(str(p).replace(".mp4",".txt"), "w") as f:
        f.write(cap)
print("Dataset files:", os.listdir(root))
```

**Gerçekte:** Kendi 1–3 saniyelik kliplerini yükle, her biri için **uyumlu** kısa bir caption (.txt) yaz.

---

##### 3) LoRA konfigürasyonu (diffusion-pipe, düşük VRAM ayarları)

```bash
%%bash
cat > /content/diffusion-pipe/examples/wan22_video_lora_colab.toml << 'EOF'
[run]
output_dir = "/content/outputs/wan22_run1"
pipeline_stages = 1

[model]
# diffusion-pipe içindeki Wan alias'ı; projede Wan2.2 desteği mevcut.
# TI2V-5B ile LoRA eğitimi hedefliyoruz (video t2v/i2v unified).
name = "wan2_2_ti2v"
pretrained_path = "/content/Wan2.2/Wan2.2-TI2V-5B"
train_mode = "lora"
lora_rank = 16
lora_alpha = 16

[data]
paths = ["/content/data/my_videos"]
type = "video"
resolution = 384    # T4 için düşük çözünürlük
fps = 12            # T4 için düşük fps
caption_ext = ".txt"
bucket_aspect = true

[train]
epochs = 4
batch_size = 1
grad_accum = 16     # efektif batch'i büyütür, VRAM tasarrufu
lr = 5e-5
optimizer = "AdamW8bitKahan"

[schedule]
warmup_steps = 200
lr_scheduler = "cosine"

[checkpointing]
save_every_n_epochs = 1
resume = false

[optimizer]
type = "AdamW8bitKahan"
lr = 5e-5
betas = [0.9, 0.99]
weight_decay = 0.01

[memory]
activation_checkpointing = "unsloth"
blocks_to_swap = 32
EOF
echo "Config written."
```

---

##### 4) Cache + Training (T4 16 GB ayarları)

```bash
%cd /content/diffusion-pipe

# Cache only (latents + text embeddings)
!deepspeed --num_gpus=1 train.py --deepspeed \
  --config examples/wan22_video_lora_colab.toml --cache_only

# Training (NCCL ayarları tek GPU'da da stabilite sağlar)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"

!deepspeed --num_gpus=1 train.py --deepspeed \
  --config examples/wan22_video_lora_colab.toml
```

> **İpucu:** OOM olursa `resolution=320` / `fps=8` / `grad_accum=32` deneyin.
> **Colab Pro/A100** varsa `resolution=512` / `fps=16` / `lora_rank=32`’ye çıkabilirsin.

---

##### 5) (Opsiyonel) LoRA ile inference

LoRA adaptörü **/content/outputs/wan22_run1/epochX** altında oluşur (`.safetensors` + config).
LoRA’yı inference tarafında nasıl “takacağın” kullandığın araca göre değişir:

* **diffusion-pipe inference**: Aynı framework içinde LoRA’yı load edip örnek üretebilirsin.
* **Wan2.2 generate.py**: Resmî script *doğrudan* LoRA adapter yüklemeyi desteklemeyebilir; topluluktaki LoRA-loader örneklerine göre bir “merge/apply” adımı gerekir (araç bağımlı). Pratik yol: **LoRA’yı diffusion-pipe ile sample alarak** kalite kontrol etmek.

Basit bir örnek (framework içi inference fikri):

```bash
# Not: diffusion-pipe içinde model+LoRA ile sample alan yardımcı script/örnekler değişebilir.
# Burada fikir vermek için temsili bir komut gösteriliyor.
# Projeye göre "inference.py" benzeri bir yardımcı kullanın ya da training script'inde eval/sample hooks açın.
```

> **Kolay demo** istersen: LoRA sonrası kaliteye bakmak için **eğitimde kullandığın klip/caption’lardan kısa örnekler** çıkar. Büyük ölçekli inference için, LoRA desteği olan bir inference pipeline (ComfyUI/Diffusers entegrasyonları vs.) tercih edebilirsin.

---

##### 6) “Ne beklemeliyim?”

* **Çıktılar:** `/content/outputs/wan22_run1/` → epoch klasörleri (LoRA ağırlıkları), TensorBoard logları.
* **Kalite:** Küçük veri + düşük çözünürlükte **stil/konsept aktarımı** bekle; uzun sekans & yüksek çözünürlük için daha fazla veri ve/veya daha büyük GPU gerekir.
* **Hız:** T4’te eğitim/inference yavaştır; Colab Pro(+)/A100 veya Runpod/Lambda gibi kiralık GPU’lar çok daha verimli.

---

##### 7) SSS (kısa)

* **Custom training var mı?** Resmî Wan2.2 reposu inference odaklı; *sıfırdan* eğitim yok. **Transfer learning/LoRA** ile özelleştiriyoruz.
* **Seedance?** Kapalı; yerelde eğitimi yok.
* **Dataset yapısı?** `video.mp4 + video.txt` (caption).
* **Windows/Mac?** NVIDIA CUDA gerekir. Mac’te GPU eğitimi pratik değil; bulut GPU önerilir.

---

##### 8) Hızlı karar tablosu

| Senaryo                     | Donanım  | Ayar                                            |
| --------------------------- | -------- | ----------------------------------------------- |
| Colab (T4 16 GB) ile deneme | T4 16 GB | 384p, 8–12 fps, LoRA rank 16, grad_accum 16–32  |
| 4090 (24 GB)                | 24 GB    | 512–576p, 12–16 fps, LoRA rank 32               |
| A100 80 GB / H100           | 80 GB+   | 720p@24 fps (TI2V-5B) ve üstü, daha büyük batch |

---

> **We cannot train Seedance locally (closed weights). We *can* fine-tune open models like Wan2.2 using diffusion-pipe on our GPUs/Colab to deliver a domain-adapted video generator via LoRA.**

---

### 2. **LTX-Video (Lightricks)**

* **What it is:** Optimized video diffusion model with an official training framework.
* **Training:** “LTX-Video-Trainer” supports both full fine-tuning and LoRA. Requires short video + caption datasets.
* **Strength:** Easy entry point; stable training pipeline; good for rapid prototyping on limited hardware.
---

#### LTX-Video — Fine-Tuning (LoRA / Full) + Inference Pipeline

**What is it?**
LTX-Video is a DiT-based, real-time video diffusion model with distilled and FP8 variants, long-shot support (up to ~60s), and first-class ComfyUI/Diffusers integrations. ([Hugging Face][1])

**Official resources**

* GitHub (model & inference): [https://github.com/Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video) ([GitHub][2])
* Hugging Face models (13B/2B, distilled & FP8): [https://huggingface.co/Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video) ([Hugging Face][1])
* Diffusers pipeline docs: [https://huggingface.co/docs/diffusers/en/api/pipelines/ltx_video](https://huggingface.co/docs/diffusers/en/api/pipelines/ltx_video) ([Hugging Face][3])
* ComfyUI integration: [https://github.com/Lightricks/ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) ([GitHub][4])
* Community trainer (fine-tune / LoRA / full FT): [https://github.com/Lightricks/LTX-Video-Trainer](https://github.com/Lightricks/LTX-Video-Trainer) ([GitHub][5])

---

##### 0) TL;DR

* **Inference:** Use the official LTX-Video repo or ComfyUI workflow. ([Hugging Face][1])
* **Fine-tuning:** Use **LTX-Video-Trainer** (supports LoRA, full FT, control/effect LoRAs). ([GitHub][5])
* **Data format:** short clips (2–5s) + matching `.txt` captions per clip.
* **Colab note:** On a T4 (16GB), stick to lower res/FPS and LoRA; scale up on A100/H100.

---

##### 1) Colab: environment setup

```bash
# Check GPU
!nvidia-smi

# Core deps (CUDA wheels on Colab)
!pip -q install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip -q install accelerate deepspeed datasets pillow imageio opencv-python tensorboard

# LTX-Video (inference code)
%cd /content
!git clone https://github.com/Lightricks/LTX-Video.git
%cd LTX-Video
!python -m pip -q install -e .[inference]
```

> LTX-Video was tested with PyTorch ≥2.1.2; FP8 kernels are optional and target Ada+ GPUs. ([Hugging Face][1])

---

##### 2) Download model weights (Hugging Face)

```bash
!pip -q install "huggingface_hub[cli]"

# Distilled 13B (fast/less VRAM)
!huggingface-cli download Lightricks/LTX-Video-0.9.8-13B-distilled --local-dir /content/ltxv-13b-0.9.8-distilled

# (Optional) Smaller 2B distilled
!huggingface-cli download Lightricks/LTX-Video-0.9.8-2b-distilled --local-dir /content/ltxv-2b-0.9.8-distilled
```

> Distilled & FP8 variants enable faster, lower-VRAM generation; 0.9.8 adds long-shot improvements. ([Hugging Face][1])

---

##### 3) Quick inference (Image-to-Video)

```bash
%cd /content/LTX-Video
!python inference.py \
  --prompt "A cinematic dolly shot of a skateboarder at golden hour" \
  --conditioning_media_paths /content/sample.jpg \
  --conditioning_start_frames 0 \
  --height 704 --width 1216 \
  --num_frames 121 \
  --seed 42 \
  --pipeline_config configs/ltxv-13b-0.9.8-distilled.yaml
```

> For best quality, use the provided **ComfyUI** workflows; `inference.py` is improving and may not match ComfyUI parity yet. Frame count should be **(8×k)+1**; invalid sizes are padded/cropped. ([Hugging Face][1])

---

##### 4) Fine-tuning (LoRA) with **LTX-Video-Trainer**

```bash
%cd /content
!git clone https://github.com/Lightricks/LTX-Video-Trainer.git
%cd LTX-Video-Trainer
!pip -q install -e .
```

**Dataset layout**

```
/content/data/my_videos/
  clip0001.mp4
  clip0001.txt   # short, accurate description
  clip0002.mp4
  clip0002.txt
```

**Minimal LoRA run (distilled 13B, low-VRAM PoC)**

```bash
# Example flags; adjust to match the trainer's README for your version.
!python train_lora.py \
  --base_model_dir /content/ltxv-13b-0.9.8-distilled \
  --data_root /content/data/my_videos \
  --output_dir /content/outputs/ltxv_lora_run1 \
  --resolution 576 --fps 16 \
  --lora_rank 32 --lora_alpha 32 \
  --batch_size 1 --grad_accum 16 \
  --lr 5e-5 --epochs 6 \
  --use_fp8 --enable_offload
```

> The trainer supports **LoRA** and **full fine-tuning** (multi-GPU), plus **Control LoRAs** (pose/depth/canny) and **Effect LoRAs**. For Colab T4, drop to **384–512p, 8–16 FPS, rank 16–32**; scale up on A100/H100. ([GitHub][5])

---

##### 5) Using the LoRA for inference

* Apply/merge the produced LoRA adapter with your inference pipeline.
* The ecosystem also provides **IC-LoRA control** (pose/depth/canny) and a **detailer** model; easiest path is via **ComfyUI** workflows. ([GitHub][4])

---

##### 6) Speed & long-shot tips

* **0.9.8**: long-shot (up to ~60s) and multi-scale rendering; distilled & FP8 variants reduce steps/VRAM. ([Hugging Face][1])
* **Diffusers** integration exposes the **Video-VAE** (≈1:192 compression) for efficient generation. ([Hugging Face][3])

---

##### 7) Quality checklist

* **Prompting:** One paragraph with action → motion → appearance → environment → camera → lighting/color; keep it literal and ≤200 words. ([Hugging Face][1])
* **Validation:** Hold-out 50–100 clips; check temporal consistency, flicker, and prompt adherence after each epoch.

---

##### 8) Quick decision table

| Scenario                  | Hardware           | Recommendation                                 |
| ------------------------- | ------------------ | ---------------------------------------------- |
| PoC / demo                | 1× RTX 4090 (24GB) | **LoRA** on distilled 13B @ 576p/16fps         |
| Budget / low VRAM         | 4060-class         | FP8 or **2B distilled**, 480–576p, fewer steps |
| High quality / long shots | H100/A100 (80GB)   | Full FT or distilled 13B + multi-scale         |

---

###### One-liner for stakeholders

[1]: https://huggingface.co/Lightricks/LTX-Video?utm_source=chatgpt.com "Lightricks/LTX-Video"
[2]: https://github.com/Lightricks/LTX-Video?utm_source=chatgpt.com "Lightricks/LTX-Video: Official repository for LTX-Video"
[3]: https://huggingface.co/docs/diffusers/en/api/pipelines/ltx_video?utm_source=chatgpt.com "LTX-Video"
[4]: https://github.com/Lightricks/ComfyUI-LTXVideo?utm_source=chatgpt.com "Lightricks/ComfyUI-LTXVideo: LTX-Video Support for ..."
[5]: https://github.com/Lightricks/LTX-Video-Trainer?utm_source=chatgpt.com "Community trainer for Lightricks' LTX Video model 🎬 ⚡️"


---

### 3. **HunyuanVideo (Tencent)**

* **What it is:** Multimodal text-to-video model with strong prompt alignment.
* **Training:** Fine-tuning supported (LoRA recommended for smaller datasets). Public training code available.
* **Strength:** Strong controllability and camera motion handling.
---

#### HunyuanVideo — Inference + LoRA Fine-Tuning

**Overview**
HunyuanVideo is Tencent’s open-sourced large-scale video generation model (≈13B params), supporting text-to-video, image-to-video, and unified multimodal conditioning. It uses a **3D VAE** backbone and spatio-temporal attention, achieving high-fidelity results.

**Official resources**

* GitHub (main repo, inference code): [Tencent/HunyuanVideo](https://github.com/Tencent/HunyuanVideo)
* Hugging Face model hub (weights): [tencent/HunyuanVideo](https://huggingface.co/Tencent/HunyuanVideo)
* Research paper: [arXiv:2412.03603](https://arxiv.org/abs/2412.03603)
* Community repo (LoRA training for I2V): [Tencent-Hunyuan/HunyuanVideo-I2V](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V)
* GitHub discussion on LoRA training: [Issue #187](https://github.com/Tencent/HunyuanVideo/issues/187)

⚠️ **Note**: Full training from scratch is **not** open. You can run inference directly and experiment with **LoRA fine-tuning** using the I2V repo or adapted community scripts.

---

##### 0) TL;DR

*  **Inference** supported officially.
*  **LoRA fine-tuning** possible via [HunyuanVideo-I2V](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V).
*  **Full custom training** is not available.
*  **Hardware needs**: up to ~60 GB VRAM for 720p inference on the 13B model.

---

##### 1) Colab setup (inference)

```bash
# Check GPU
!nvidia-smi

# Install deps
!pip -q install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip -q install accelerate datasets pillow imageio opencv-python

# Clone repo
%cd /content
!git clone https://github.com/Tencent/HunyuanVideo.git
%cd HunyuanVideo
!pip -q install -r requirements.txt
```

---

##### 2) Download model weights

```bash
!pip -q install "huggingface_hub[cli]"
!huggingface-cli download Tencent/HunyuanVideo --local-dir /content/hunyuanvideo_ckpts
```

(Weights are large, so Colab T4 isn’t sufficient; use A100/H100 for practical runs.)

---

##### 3) Example inference

```bash
%cd /content/HunyuanVideo
!python sample.py \
  --prompt "A futuristic city skyline at dusk, with a dynamic drone shot" \
  --output_path out.mp4 \
  --height 720 --width 1280 \
  --num_frames 64
```

---

##### 4) LoRA fine-tuning (community path)

Clone the [HunyuanVideo-I2V](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V) repo:

```bash
%cd /content
!git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V.git
%cd HunyuanVideo-I2V
!pip -q install -r requirements.txt

# Example LoRA training (pseudo; check repo README for exact args)
!python train_lora.py \
  --data_root /content/data/my_images \
  --output_dir /content/outputs/hyv_i2v_lora \
  --lora_rank 16 --lr 5e-5 --epochs 4
```

You then load the LoRA adapter into the inference pipeline.

---

##### 5) Practical constraints

* **VRAM**: ≈60 GB for 720p inference.
* **Colab**: Only tiny demo runs possible; real training/inference requires A100/H100.
* **LoRA scope**: Image-to-Video (I2V) fine-tuning is feasible. Full text-to-video fine-tuning isn’t public yet.

---

###### One-liner for stakeholders

> **HunyuanVideo** enables high-fidelity video generation with open weights. We can run inference directly, and apply domain adaptation via LoRA (community repos). Full training isn’t open-sourced.
---

### 4. **Mochi (Genmo)**

* **What it is:** Open-source diffusion transformer for video generation.
* **Training:** Supports LoRA and large-scale training via frameworks like Deepspeed or diffusion-pipe.
* **Strength:** Research-friendly; competitive benchmarks in the open community.

---

### 5. **diffusion-pipe (Framework, not a model)**

* **What it is:** A unified training infrastructure for large diffusion models (supports Wan, Mochi, SDXL-Video, etc.).
* **Training:** Enables pipeline-parallel, multi-GPU training, caching, and efficient LoRA workflows.
* **Strength:** Practical for scaling open models across GPU clusters.

---

**Summary:**

* For **small-scale fine-tuning (single GPU)**: Wan-1.3B or LTX-Video are practical.
* For **medium scale (multi-GPU clusters)**: HunyuanVideo or Mochi with diffusion-pipe.
* For **enterprise-scale R&D**: Wan-14B requires 100+ GPUs and large-scale curated datasets.

##  Fiyatlandırma

| Platform        | Ücretsiz Plan | Aylık Fiyatlar | Yıllık İndirimli | Notlar |
|-----------------|---------------|----------------|------------------|--------|
| **Seedance AI** | Yok        | $29.9 – $89.9  | $19.9 – $62.9    | Ticari lisans dahil |
| **Wan Video**   |  Var        | $5 – $20       | Yıllık ödeme ile  | Ek kredi paketleri mevcut |
| **LTX Studio**  |  Yok        | $15 – $125     | $12 – $100       | “Compute seconds” bazlı |
| **Hunyuan Video** | Var      | Ücretsiz       | –                | Tencent destekli, 4K üretim |

###  Özet
- **En ucuz başlangıç:** Wan Video ($5/ay)  
- **Daha profesyonel özellikler:** LTX Studio ($35–125/ay)  
- **Ticari lisanslı SaaS:** Seedance ($30–90/ay)  
- **Deneme / araştırma için:** Hunyuan (şimdilik ücretsiz)

  
https://github.com/tdrussell/diffusion-pipe?tab=readme-ov-file
https://github.com/tdrussell/diffusion-pipe/blob/main/docs/supported_models.md
https://huggingface.co/Lightricks/LTX-Video?utm_source=chatgpt.com
https://huggingface.co/docs/diffusers/en/api/pipelines/wan
https://huggingface.co/docs/diffusers/index
https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B?utm_source=chatgpt.com
https://vast.ai/?utm_target=&utm_group=&placement=&device=c&adposition=&utm_source=google&utm_medium=cpc&utm_campaign=20740487654_&utm_content=&utm_term=&hsa_acc=7028527117&hsa_cam=20740487654&hsa_grp=&hsa_ad=&hsa_src=x&hsa_tgt=&hsa_kw=&hsa_mt=&hsa_net=adwords&hsa_ver=3&gad_source=1&gad_campaignid=22457588611&gbraid=0AAAAAC66i41TXjKRbSmgGjmhiMff2LzFL&gclid=CjwKCAjwuePGBhBZEiwAIGCVS2rnw9R2SBFsRdq-tbQjhbRy3Yk7_zWF8bhA5w7WD6EmUG_6dfY5DBoC9rAQAvD_BwE
https://www.runpod.io/

Wan 1.3B → 8–16 GB VRAM yeter (Colab T4’de çalışır).
LTX 5B → 24+ GB VRAM gerekir (4090 sınırda, A100 daha iyi).
Hunyuan 13B → 60+ GB VRAM gerekir (tek GPU’da imkânsız, cluster lazım).

---

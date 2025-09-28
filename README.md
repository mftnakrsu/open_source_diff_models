
**Subject:** Open-Source Options for Training Video Diffusion Models


---

### 1. **Wan 2.1 / 2.2 (Alibaba, Open Source)**

* **What it is:** State-of-the-art open video diffusion transformer with two sizes: 1.3B (runs on consumer GPUs) and 14B (requires cluster-scale compute).
* **Training:** Public repo available. LoRA fine-tuning possible on a single 24 GB GPU; full training needs large GPU clusters.
* **Strength:** Leading open performance; covers text-to-video, image-to-video, editing, and personalization tasks.

---

### 2. **LTX-Video (Lightricks)**

* **What it is:** Optimized video diffusion model with an official training framework.
* **Training:** “LTX-Video-Trainer” supports both full fine-tuning and LoRA. Requires short video + caption datasets.
* **Strength:** Easy entry point; stable training pipeline; good for rapid prototyping on limited hardware.

---

### 3. **HunyuanVideo (Tencent)**

* **What it is:** Multimodal text-to-video model with strong prompt alignment.
* **Training:** Fine-tuning supported (LoRA recommended for smaller datasets). Public training code available.
* **Strength:** Strong controllability and camera motion handling.

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



---

Do you want me to also prepare a **one-slide visual summary** (like a comparison table with model/logo, training cost, and best use case) so you can show it directly in a meeting?

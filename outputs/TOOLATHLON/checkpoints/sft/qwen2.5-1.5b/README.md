---
library_name: peft
model_name: qwen2.5-1.5b
tags:
- base_model:adapter:/seu_share2/home/fenglei/sharedata/Qwen2.5-1.5B-Instruct
- lora
- sft
- transformers
- trl
licence: license
pipeline_tag: text-generation
base_model: /seu_share2/home/fenglei/sharedata/Qwen2.5-1.5B-Instruct
---

# Model Card for qwen2.5-1.5b

This model is a fine-tuned version of [None](https://huggingface.co/None).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- PEFT 0.18.0
- TRL: 0.28.0
- Transformers: 4.57.3
- Pytorch: 2.9.1
- Datasets: 4.5.0
- Tokenizers: 0.22.1

## Citations



Cite TRL as:
    
```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```
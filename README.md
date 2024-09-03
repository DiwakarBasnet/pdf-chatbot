## YaRN - Yet another RoPE extensioN
YaRN is a compute-efficient method, it requires 10 times fewer tokens anda 2.5 times fewer training steps than former methods.
## 128K
128K context window
## GGUF - GPT-Generated Unified Format


## Things to do for running application
```
mkdir data
```
```
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download TheBloke/Yarn-Mistral-7B-128k-GGUF yarn-mistral-7b-128k.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
```
```
pip install -r requirements.txt
```

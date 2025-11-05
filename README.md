# Agent-Omni

The official implementation for the paper “Agent-Omni: Test-Time Multimodal Reasoning via Model Coordination for Understanding Anything.”

Agent-Omni is a modular framework that enables test-time multimodal reasoning by coordinating existing foundation models through a master-agent system. Instead of end-to-end fine-tuning, the master agent interprets user intent, delegates subtasks to modality-specific agents (text, image, audio, video), and integrates their outputs into coherent responses. This design allows flexible, interpretable, and extensible omni-modal understanding across diverse input combinations.

## Installation

```
git clone https://github.com/huawei-lin/Agent-Omni.git
cd Agent-Omni

conda create -n agent-omni python=3.10 -y
conda activate agent-omni
pip install -r requirements.txt
```

## Quick Start

Here's an example using `Qwen/Qwen3-Omni-30B-A3B-Instruct` on (SiliconFlow)[https://www.siliconflow.cn/] (setup is similar to OpenAI):
```
export siliconflow_api_key=${your_api_key}
```

### Model Configuration

Model configurations are stored in `configs/model_configs/qwen3-omni-30b-a3b-instruct.yaml`.
```yaml
api_provider: openai # can be 'openai', 'bedrock', or 'local'
model_id: Qwen/Qwen3-Omni-30B-A3B-Instruct
base_url: https://api.siliconflow.cn/v1
api_key_name: siliconflow_api_key
params:
  max_tokens: 8192
  temperature: 0
```

### Run a Vanilla LLM
```
cd test
python vanilla_test.py --config_path ../configs/model_configs/qwen3-omni-30b-a3b-instruct.yaml
```

### Run Agent-Omni

Agent-Omni configurations are defined in `test/config.yaml`:
```
model:
  master_agent: ../configs/model_configs/qwen3-omni-30b-a3b-instruct.yaml
  text_agent: ../configs/model_configs/qwen3-omni-30b-a3b-instruct.yaml
  image_agent: ../configs/model_configs/qwen3-omni-30b-a3b-instruct.yaml
  video_agent: ../configs/model_configs/qwen3-omni-30b-a3b-instruct.yaml
  audio_agent: ../configs/model_configs/qwen3-omni-30b-a3b-instruct.yaml

agents:
  supported: ["text", "image", "video", "audio"]
  names: ["text_agent", "image_agent", "video_agent", "audio_agent"]

system:
  retry_times: 10
```
You can assign different models for each modality as needed.

Then simply run:
```
cd test
python agent_test.py
```

## Notes

- Agent-Omni supports most OpenAI-compatible APIs, including SiliconFlow, Bedrock, and vLLM backends.
- Each agent can be configured independently, enabling fine-grained multimodal orchestration.
- No fine-tuning is required — the system operates fully at test time.

## Citation
@article{agent-omni,
  author       = {Huawei Lin and
                  Yunzhi Shi and
                  Tong Geng and
                  Weijie Zhao and
                  Wei Wang and
                  Ravender Pal Singh},
  title        = {Agent-Omni: Test-Time Multimodal Reasoning via Model Coordination for Understanding Anything},
  journal      = {arXiv preprint arXiv:xxxx.xxxxx},
  year         = {2025}
}





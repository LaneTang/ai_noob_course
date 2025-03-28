import re
import torch
import pandas as pd
from datasets import load_dataset, Dataset
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

print("开始微调!")

SYSTEM_PROMPT = """
严格按以下格式回答问题：

<reasoning>
在此处详细写出你的推理步骤（必须换行）
</reasoning>

<answer>
在此处只输出最终答案的数字（必须换行）
</answer>

注意：
1. <reasoning> 和 <answer> 标签必须独立成行
2. 答案必须是纯数字，不能包含任何其他字符
3. 标签必须正确闭合
"""

data = load_dataset('json', data_files='../Datasets/GSM8K_zh.json', split='train')
data = pd.DataFrame(data)
data = data[['question_zh', 'answer_only']]
def convert_data(data,SYSTEM_PROMPT):
    new_data = []
    for x in data.values.tolist():
        # 构建训练数据格式
        item = {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x[0]}
            ],
            'answer':x[1]
        }
        
        new_data.append(item)
    return new_data

dataset = convert_data(data,SYSTEM_PROMPT)
print("数据加载成功!")


# 从文本中提取 <answer> 标签内的内容
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

class RewardLogger:
    __name__ = "correctness_reward_func"  # 添加这一行
    def __init__(self):
        self.step = 0
        self.rewards_history = []
    
    def __call__(self, prompts, completions, answer, **kwargs):
        responses = [completion[0]['content'] for completion in completions]
        extracted = [extract_xml_answer(r) for r in responses]
        rewards = [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]
        
        self.rewards_history.extend(rewards)
        self.step += 1
        
        if self.step % 100 == 0:
            avg = sum(self.rewards_history) / len(self.rewards_history) if self.rewards_history else 0
            print(f"Step {self.step} | Avg Reward: {avg:.2f}")
            self.rewards_history = []  # 可选：是否清空历史
        if self.step % 200 == 0:
            print('-'*20, f"Question:\n{prompts[0][-1]['content']}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted[0]}")
        return rewards

# 初始化
reward_logger = RewardLogger()

# 奖励函数：检查回答是否为整数
def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # 如果提取的回答是一个数字字符串，奖励 0.5，否则奖励 0.0
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """严格检查格式（允许标签前后有空格）"""
    pattern = r"^\s*<reasoning>\n+.+?\n+</reasoning>\s*<answer>\n+.+?\n+</answer>\s*$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if m else 0.0 for m in matches]

# 奖励函数：宽松检查完成结果的格式
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    # 定义宽松的格式匹配模式，不要求 <reasoning> 和 <answer> 标签内有换行
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    # 如果匹配成功，奖励 0.5，否则奖励 0.0
    return [0.5 if match else 0.0 for match in matches]

# 计算文本中特定 XML 标签的数量并给予相应奖励
def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

# 奖励函数：根据 XML 标签计数给予奖励
def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    # 从 completions 中提取每个完成结果的内容
    contents = [completion[0]["content"] for completion in completions]
    # 对每个内容调用 count_xml 函数计算奖励
    return [count_xml(c) for c in contents]

# 模型地址
model_name = "/root/Model/Qwen/Qwen2___5-0___5B-Instruct"
# 输出地址
output_dir="outputs/Qwen-0.5B-GRPO"
# 自定义任务名称
run_name="Qwen-0.5B-GRPO-gsm8k-zh"

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=10,
    bf16=True,
    per_device_train_batch_size=12,
    # 梯度累积步数，将多个小批次的梯度累积起来，等效于使用更大的批次大小进行训练
    gradient_accumulation_steps=1,
    # 每个样本生成的候选数量，例如在生成式任务中可能会生成多个候选输出
    num_generations=12,
    max_prompt_length=256,
    max_completion_length=512,
    num_train_epochs=1,
    save_steps=500,
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=False,
    vllm_gpu_memory_utilization=.3,
    vllm_device="cuda",
    report_to="none"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        reward_logger],
    args=training_args,
    train_dataset=dataset,
    #peft_config=peft_config
)
trainer.train()

trainer.save_model(output_dir)
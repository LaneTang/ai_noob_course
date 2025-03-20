## 1.安装LLama-factory

下载LLama-factory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
```

进入仓库目录

```bash
cd LLaMA-Factory
```

安装LLama-factory

```bash
pip install -e ".[torch,metrics]"
```

## 2.添加数据集

2.1 上传数据集ruozhiba_qa.json至LLaMA-Factory/data

2.2 修改LLaMA-Factory/data/dataset_info.json
添加下述信息进入dataset_info.json
```
"ruozhiba_qa": {
    "file_name": "ruozhiba_qa.json"
  }
```

## 3. 修改模板文件

修改 LLaMA-Factory/src/llamafactory/data/template.py
添加下述信息进入template.py
```python
register_template(
  name="qwen2",
  format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
  format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
  format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
  default_system="You are a helpful assistant.",
  stop_words=["<|im_end|>"],
  replace_eos=True,
)
```

## 4. 修改配置文件

### 4.1 修改训练文件:
修改:LLaMA-Factory/examples/train_lora/llama3_lora_sft.yaml

```
model_name_or_path: ../../Models/Qwen/Qwen2___5-0___5B-Instruct # 模型路径

dataset: ruozhiba_qa # 数据集名称

template: qwen2 # 模板名称

output_dir: saves/Qwen25 # 输出储存路径
```

### 4.2 修改测试文件:
修改:LLaMA-Factory/examples/inference/llama3_lora_sft.yaml
```
model_name_or_path: ../../Models/Qwen/Qwen2___5-0___5B-Instruct
adapter_name_or_path: saves/Qwen25
template: qwen2
```

## 5. 运行训练

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

## 测试运行结果

```bash
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
```

## 6. 设置测试集

修改:LLaMA-Factory/examples/train_lora/llama3_lora_sft.yaml

```
### eval
val_size: 0.1
```

再次运行, 测试运行结果

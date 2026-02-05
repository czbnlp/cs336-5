import json
import gzip
import torch
import random
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, seq_length: int, shuffle: bool):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        dataset_path_str = str(dataset_path)
        
        # 1. 加载数据
        docs = []
        open_fn = gzip.open if dataset_path_str.endswith(".gz") else open
        with open_fn(dataset_path_str, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))

        # 2. 打乱顺序
        if shuffle:
            random.seed(42)
            random.shuffle(docs)

        # 3. 格式化与拼接
        template = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{prompt}\n\n"
            "### Response:\n{response}"
        )
        
        all_token_ids = []
        eos_token_id = tokenizer.eos_token_id
        
        for doc in docs:
            p = doc.get("prompt", doc.get("instruction", ""))
            r = doc.get("response", doc.get("output", ""))
            
            formatted_text = template.format(prompt=p, response=r)
            
            # 保持默认行为：自动添加 BOS (128000)
            tokens = tokenizer.encode(formatted_text)
            
            all_token_ids.extend(tokens)
            all_token_ids.append(eos_token_id)

        # 4. 执行 Packing (关键修改)
        self.input_chunks = []
        self.label_chunks = []
        
        # 这里的 range 需要确保我们能取到 i+seq_length+1 的数据
        # 输入是 [i : i+seq_len]
        # 标签是 [i+1 : i+seq_len+1] (向后移一位)
        for i in range(0, len(all_token_ids) - seq_length, seq_length):
            input_chunk = all_token_ids[i : i + seq_length]
            label_chunk = all_token_ids[i + 1 : i + seq_length + 1]
            
            self.input_chunks.append(input_chunk)
            self.label_chunks.append(label_chunk)

    def __len__(self):
        return len(self.input_chunks)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.input_chunks[i], dtype=torch.long),
            "labels": torch.tensor(self.label_chunks[i], dtype=torch.long)
        }
import os
import sys
import yaml
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from utils.scrn_model import SCRNModel
from safetensors.torch import load_file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = {
    'model_name': 'klue/roberta-base',
    'model_dir': 'data_out/scrn_data/checkpoint-87816',
    'batch_size': 16,
    'max_length': 512
}
class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length
        )

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

def main():
    
    model_name = cfg['model_name']

    config = AutoConfig.from_pretrained(model_name)
    model = SCRNModel(model_name, config=config)
    
    # safetensors 파일 로드하는 올바른 방법
    safetensors_path = os.path.join(cfg['model_dir'], 'model.safetensors')
    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict)
        print(f"✅ Model loaded from safetensors: {safetensors_path}")
    else:
        # 백업으로 .bin 파일 시도
        bin_path = os.path.join(cfg['model_dir'], 'pytorch_model.bin')
        if os.path.exists(bin_path):
            model.load_state_dict(torch.load(bin_path))
            print(f"✅ Model loaded from .bin: {bin_path}")
        else:
            raise FileNotFoundError(f"Neither {safetensors_path} nor {bin_path} found")
    
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_path = "/home/jiseung/TCN/data/test.csv"
    test_df = pd.read_csv(test_path)

    texts = test_df['paragraph_text'].tolist()
    ids = test_df['ID'].tolist()



    test_dataset = InferenceDataset(texts, tokenizer, cfg['max_length'])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model.eval()
    scores = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="🚀 Running Inference"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # BCELoss에 맞는 sigmoid 활용 (main.py와 동일)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]  # class 1의 확률만 사용
            scores.extend(probs.cpu().numpy().flatten().tolist())  # or use list comprehension


    # ----------------------
    # 제출 파일 저장
    # ----------------------
    print(f"🔍 len(ids) = {len(ids)}, len(scores) = {len(scores)}")

    submission = pd.DataFrame({
        'ID': ids,
        'generated': scores
    })

    submission_path = '/home/jiseung/TCN/outputs/submissions/scrn_roberta_submission.csv'
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission.to_csv(submission_path, index=False)

    print(f"✅ Submission saved to {submission_path}")
    print(f"📊 Prediction statistics:")
    print(f"   - Mean score: {submission['generated'].mean():.4f}")
    print(f"   - Min score: {submission['generated'].min():.4f}")
    print(f"   - Max score: {submission['generated'].max():.4f}")
    print(f"   - Predictions > 0.5: {(submission['generated'] > 0.5).sum()}")

if __name__ == "__main__":
    main()
    
    
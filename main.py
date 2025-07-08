from utils.utils import set_logger, path_checker, metrics_fn, compute_metrics

import torch
import numpy as np
import pandas as pd
import random
import pickle
import datetime
import json
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import Dataset

from transformers import (AutoConfig, AutoModelForSequenceClassification, Trainer, HfArgumentParser, set_seed, 
AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForCausalLM)

from arguments import ModelArguments, DataTrainingArguments, TrainingArguments
from datasets import load_dataset
from utils.scrn_model import SCRNModel, SCRNTrainer
from utils.metric_based import get_ll, get_rank, get_entropy, get_rank_GLTR, run_threshold_experiment, run_GLTR_experiment
from utils.metric_utils import load_base_model_and_tokenizer
from utils.flooding_model import FloodingTrainer
from utils.rdrop import RDropTrainer
from utils.ranmask_model import RanMaskModel
from utils.utils import mask_tokens

import wandb
import os
from functools import lru_cache
from typing import Dict, Any
import threading

# os.environ["WANDB_MODE"] = "offline"  # 원격 업로드를 위해 주석 처리
os.environ["WANDB__SERVICE_WAIT"] = "300"
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        # Cross entropy용으로 long 타입으로 변환 (SCRN 모델은 cross entropy 사용)
        self.labels = torch.tensor(labels, dtype=torch.long)

        
    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": self.labels[idx],  # 이미 float32 tensor로 되어 있음
        }

        # Optional: token_type_ids가 있는 경우 처리
        if "token_type_ids" in self.encodings:
            item["token_type_ids"] = torch.tensor(self.encodings["token_type_ids"][idx])

        return item
    def __len__(self):
        return len(self.labels)
class ParagraphDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, cache_size=1000):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_size = cache_size
        self._cache = {}
        self._lock = threading.Lock()  # 멀티프로세싱 안전성
        
        # 캐시 히트/미스 통계 (선택사항)
        self.cache_hits = 0
        self.cache_misses = 0

    def __len__(self):
        return len(self.texts)

    def _tokenize_text(self, idx: int) -> Dict[str, torch.Tensor]:
        """텍스트를 토크나이징하는 내부 함수"""
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        result = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }
        
        if "token_type_ids" in encoding:
            result["token_type_ids"] = encoding["token_type_ids"].squeeze()
            
        return result

    def __getitem__(self, idx):
        # 캐시에서 먼저 확인
        with self._lock:
            if idx in self._cache:
                self.cache_hits += 1
                cached_item = self._cache[idx].copy()
                cached_item["labels"] = self.labels[idx]
                return cached_item
            else:
                self.cache_misses += 1

        # 캐시에 없으면 토크나이징
        tokenized = self._tokenize_text(idx)
        
        # 캐시 사이즈 관리 (LRU 방식)
        with self._lock:
            if len(self._cache) >= self.cache_size:
                # 가장 오래된 항목 제거 (간단한 FIFO, 더 정교한 LRU도 가능)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[idx] = tokenized

        # 레이블 추가
        item = tokenized.copy()
        item["labels"] = self.labels[idx]
        
        return item
    
    def get_cache_stats(self):
        """캐시 통계 반환"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache)
        }


class CustomDataCollatorForSeqCLS(DataCollatorForSeq2Seq):    
    def __call__(self, features, return_tensors=None): 
        if return_tensors is None:
            return_tensors = self.return_tensors

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        return features


def metrics_fn(outputs):
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(-1)
    y_score = torch.tensor(outputs.predictions).softmax(-1).numpy()[:, 1]
    return compute_metrics(y_true, y_pred, y_score)    
    
def compute_metrics(pred):
    labels = pred.label_ids
    # SCRN 모델은 2-class classification이므로 softmax 사용
    logits = torch.tensor(pred.predictions)
    probs = torch.softmax(logits, dim=-1)[:, 1].numpy()  # class 1의 확률만 사용
    preds_binary = (probs > 0.5).astype(int)
    
    # 혼동행렬 계산
    tn, fp, fn, tp = confusion_matrix(labels, preds_binary).ravel()
    
    return {
        "eval_accuracy": accuracy_score(labels, preds_binary),
        "eval_f1": f1_score(labels, preds_binary),
        "eval_auc_roc": roc_auc_score(labels, probs),
        "eval_auc_pr": average_precision_score(labels, probs),
        "eval_true_positive": tp,
        "eval_false_positive": fp,
        "eval_true_negative": tn,
        "eval_false_negative": fn,
        "eval_precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "eval_recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "eval_specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "eval_false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "eval_false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0.0
    }
class HybridDataset(Dataset):
    """
    메모리 효율성과 성능을 균형있게 맞춘 하이브리드 데이터셋
    - 초기에 일정 비율만 미리 토크나이징 (빠른 접근)
    - 나머지는 lazy loading + 캐시 (메모리 절약)
    """
    def __init__(self, texts, labels, tokenizer, max_length, 
                 preprocess_ratio=0.3, cache_size=500):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_size = cache_size
        
        # 미리 처리할 데이터 개수 계산
        self.preprocess_count = int(len(texts) * preprocess_ratio)
        
        print(f"🔄 Pre-processing {self.preprocess_count}/{len(texts)} samples...")
        
        # 초기 데이터 미리 토크나이징
        self.preprocessed = {}
        for i in tqdm(range(self.preprocess_count), desc="Pre-tokenizing"):
            self.preprocessed[i] = self._tokenize_text(i)
        
        # 나머지 데이터용 캐시
        self._cache = {}
        self._lock = threading.Lock()
        
        print(f"✅ Pre-processing complete. Cache size limit: {cache_size}")

    def __len__(self):
        return len(self.texts)

    def _tokenize_text(self, idx: int) -> Dict[str, torch.Tensor]:
        """텍스트를 토크나이징하는 내부 함수"""
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        result = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }
        
        if "token_type_ids" in encoding:
            result["token_type_ids"] = encoding["token_type_ids"].squeeze()
            
        return result

    def __getitem__(self, idx):
        # 1. 미리 처리된 데이터에서 확인
        if idx < self.preprocess_count:
            item = self.preprocessed[idx].copy()
            item["labels"] = self.labels[idx]
            return item
        
        # 2. 캐시에서 확인
        with self._lock:
            if idx in self._cache:
                cached_item = self._cache[idx].copy()
                cached_item["labels"] = self.labels[idx]
                return cached_item

        # 3. 새로 토크나이징
        tokenized = self._tokenize_text(idx)
        
        # 4. 캐시에 저장 (크기 제한)
        with self._lock:
            if len(self._cache) >= self.cache_size:
                # 랜덤하게 일부 제거 (더 공정한 분산)
                keys_to_remove = list(self._cache.keys())[:len(self._cache)//4]
                for key in keys_to_remove:
                    del self._cache[key]
            
            self._cache[idx] = tokenized

        # 5. 레이블 추가 후 반환
        item = tokenized.copy()
        item["labels"] = self.labels[idx]
        return item
    
    def get_stats(self):
        """데이터셋 통계 반환"""
        return {
            "total_samples": len(self.texts),
            "preprocessed_samples": self.preprocess_count,
            "cached_samples": len(self._cache),
            "preprocessing_ratio": self.preprocess_count / len(self.texts),
            "cache_usage": len(self._cache) / self.cache_size if self.cache_size > 0 else 0
        }


class CustomTrainer(Trainer):
    """데이터셋 캐시 성능 모니터링이 가능한 커스텀 트레이너"""
    
    def log(self, logs):
        """로깅 시 데이터셋 통계도 함께 기록"""
        super().log(logs)
        
        # 데이터셋 통계 수집
        if hasattr(self.train_dataset, 'get_cache_stats'):
            stats = self.train_dataset.get_cache_stats()
            for key, value in stats.items():
                logs[f"train_dataset_{key}"] = value
                
        elif hasattr(self.train_dataset, 'get_stats'):
            stats = self.train_dataset.get_stats()
            for key, value in stats.items():
                logs[f"train_dataset_{key}"] = value
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """평가 시작 전 캐시 통계 출력"""
        if hasattr(self.eval_dataset, 'get_cache_stats'):
            print(f"📊 Eval dataset cache stats: {self.eval_dataset.get_cache_stats()}")
        elif hasattr(self.eval_dataset, 'get_stats'):
            print(f"📊 Eval dataset stats: {self.eval_dataset.get_stats()}")
            
        return super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)


def main():
    supervised_model_list = ['bert-base', 'roberta-base', 'deberta-base', 'ChatGPT-Detector', 'flooding', 'rdrop', 'ranmask', 'scrn']
    metric_based_model_list = ["Log-Likelihood", "Rank", "Log-Rank", "Entropy", "GLTR"]
    wandb.login()
    wandb.init(project="TCN",entity="ruka030809-soongsil-university", name="scrn")
    
    # Get arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_abbr = training_args.output_dir.split('/')[-1]
    dataset_abbr = data_args.data_files.split('/')[-1]
    training_args.output_dir = training_args.output_dir + '_' + dataset_abbr
    # Path check and set logger
    # path_checker(training_args)
    os.makedirs(training_args.output_dir, exist_ok=True)
    print('Output directory created or already exists: %s'%training_args.output_dir)
    logger = set_logger(training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load dataset
    #####################################################################################
    # 데이터 로드 및 분할
    train_file_path = "/home/jiseung/TCN/data/pre_train.csv"
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    data = pd.read_csv(train_file_path)
    human = data[data['generated'] == 0].copy()
    ai = data[data['generated'] == 1].copy()
    del data
    human = human.sample(n=200000, random_state=42)
    data = pd.concat([human, ai], ignore_index=True)
    del human, ai
    
    #texts = data['full_text'].tolist()
    texts = data['text'].tolist()
    labels = data['generated'].tolist()
    texts = [text[:1000] for text in texts]
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    del data
    # Tokenizer & Encodings with tqdm
    # Dataset 생성
    split=True
    if split:
        print("🔄 Creating datasets...")
        
        # 데이터셋 방식 선택 (3가지 옵션)
        dataset_mode = "hybrid"  # "basic", "cached", "hybrid" 중 선택
        
        if dataset_mode == "basic":
            # 기본 방식: 매번 토크나이징 (느리지만 메모리 적게 사용)
            print("Using ParagraphDataset (dynamic tokenization)")
            train_dataset = ParagraphDataset(train_texts, train_labels, tokenizer, 512)
            eval_dataset = ParagraphDataset(val_texts, val_labels, tokenizer, 512)
            
        elif dataset_mode == "cached":
            # 캐시 방식: LRU 캐시 사용 (균형잡힌 접근)
            print("Using ParagraphDataset with LRU cache (1000 samples)")
            train_dataset = ParagraphDataset(train_texts, train_labels, tokenizer, 512, cache_size=1000)
            eval_dataset = ParagraphDataset(val_texts, val_labels, tokenizer, 512, cache_size=200)
            
        elif dataset_mode == "hybrid":
            # 하이브리드 방식: 일부 미리 처리 + 캐시 (최적의 성능)
            print("Using HybridDataset (30% pre-processed + 500 cache)")
            train_dataset = HybridDataset(
                train_texts, train_labels, tokenizer, 512, 
                preprocess_ratio=0.3, cache_size=500
            )
            eval_dataset = HybridDataset(
                val_texts, val_labels, tokenizer, 512, 
                preprocess_ratio=0.5, cache_size=200  # eval은 더 많이 미리 처리
            )
        
        print(f"✅ Datasets created using {dataset_mode} mode")
    else:
        print("🔄 Tokenizing training data...")
        train_encodings = tokenizer(
            list(tqdm(train_texts, desc="Tokenizing train")),
            truncation=True,
            padding=True,
            max_length=512
        )
        train_dataset = TextDataset(train_encodings, train_labels)
        print("Train tokenizing done!")
        print("🔄 Tokenizing validation data...")
        val_encodings = tokenizer(
            list(tqdm(val_texts, desc="Tokenizing val")),
            truncation=True,
            padding=True,
            max_length=512
        )
        eval_dataset = TextDataset(val_encodings, val_labels)


    
    #################################################################################################
    if model_abbr in supervised_model_list:
        # Load model
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=data_args.max_seq_length,
            padding_side="right",
            use_fast=False,
        )
        if model_abbr == 'scrn':
            model = SCRNModel(model_args.model_name_or_path, config)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)
        
        
        def preprocess_function_for_ranmask(examples):
            examples["text"] = mask_tokens(examples["text"], mask_token=tokenizer.mask_token)
            inputs = tokenizer(examples["text"], truncation=True)
            model_inputs = inputs
            return model_inputs
        
        def preprocess_function_for_seq_cls(examples):
            inputs = tokenizer(examples["text"], truncation=True)
            model_inputs = inputs
            return model_inputs
        
        if model_abbr == 'ranmask':
            train_data_preprocess_fn = preprocess_function_for_ranmask
            infer_data_preprocess_fn = preprocess_function_for_seq_cls
        else:
            train_data_preprocess_fn = preprocess_function_for_seq_cls
            infer_data_preprocess_fn = preprocess_function_for_seq_cls


        
        # Preprocess dataset
        #train_dataset, eval_dataset = raw_dataset["train"], raw_dataset["test"]

        # with training_args.main_process_first(desc="train dataset map pre-processing"):
        #     train_dataset = train_dataset.map(
        #         train_data_preprocess_fn,
        #         batched=True,
        #         num_proc=data_args.preprocessing_num_workers,
        #         load_from_cache_file=not data_args.overwrite_cache,
        #         desc="Running tokenizer on train dataset",
        #     )
        #     test_dataset = test_dataset.map(
        #         infer_data_preprocess_fn,
        #         batched=True,
        #         num_proc=data_args.preprocessing_num_workers,
        #         load_from_cache_file=not data_args.overwrite_cache,
        #         desc="Running tokenizer on test dataset",
        #     )
        
        # data_collator = CustomDataCollatorForSeqCLS(tokenizer, model=model, pad_to_multiple_of=8 if training_args.fp16 else None,)


        # Set trainer
        if model_abbr == 'scrn':
            trainer_fn = SCRNTrainer
        elif model_abbr == 'flooding':
            trainer_fn = FloodingTrainer
        elif model_abbr == 'rdrop':
            trainer_fn = RDropTrainer
        else:
            trainer_fn = Trainer
        trainer = trainer_fn(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            #data_collator=data_collator,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            
        )

        # Training
        if training_args.do_train:
            train_result = trainer.train()
            # trainer.save_state()
            trainer.save_model()

        # Predict
        if training_args.do_predict:
            
            if model_abbr == 'ranmask':
                config = AutoConfig.from_pretrained(training_args.output_dir)
                model = RanMaskModel.from_pretrained(training_args.output_dir)
                # set params for ensemble inference
                model.tokenizer = tokenizer
                model.mask_percentage = model_args.infer_mask_percentage
                model.ensemble_num = model_args.ensemble_num
                model.ensemble_method = model_args.ensemble_method
            elif model_abbr == 'scrn':
                config = AutoConfig.from_pretrained(training_args.output_dir)
                model = SCRNModel(model_args.model_name_or_path, config=config)
                model.load_state_dict(torch.load(os.path.join(training_args.output_dir,'pytorch_model.bin')))
            else:
                config = AutoConfig.from_pretrained(training_args.output_dir)
                model = AutoModelForSequenceClassification.from_pretrained(training_args.output_dir)
            trainer = trainer_fn(
                model=model,
                args=training_args,
                tokenizer=tokenizer,
                #data_collator=data_collator,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
            )
            predict_results = trainer.evaluate()
            trainer.save_metrics("predict", predict_results)

    elif model_abbr in metric_based_model_list:
        DEVICE = 'cuda'
        START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
        START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

        # get generative model and set device
        # gpt-2
        base_model, base_tokenizer = load_base_model_and_tokenizer(model_args.metric_base_model_name_or_path)
        base_model.to(DEVICE)

        # build features

        def ll_criterion(text): return get_ll(text, base_model, base_tokenizer, DEVICE)

        def rank_criterion(text): return -get_rank(text, base_model, base_tokenizer, DEVICE, log=False)

        def logrank_criterion(text): return -get_rank(text, base_model, base_tokenizer, DEVICE, log=True)

        def entropy_criterion(text): return get_entropy(text, base_model, base_tokenizer, DEVICE)

        def GLTR_criterion(text): return get_rank_GLTR(text, base_model, base_tokenizer, DEVICE)
    
        outputs = []
        data = train_dataset
        if model_abbr == "Log-Likelihood":
            outputs.append(run_threshold_experiment(data, ll_criterion, "likelihood", logger=logger))
        elif model_abbr == "Rank":
            outputs.append(run_threshold_experiment(data, rank_criterion, "rank", logger=logger))
        elif model_abbr == "Log-Rank":
            outputs.append(run_threshold_experiment(data, logrank_criterion, "log_rank", logger=logger))
        elif model_abbr == "Entropy":
            outputs.append(run_threshold_experiment(data, entropy_criterion, "entropy", logger=logger))
        elif model_abbr == "GLTR":
            outputs.append(run_GLTR_experiment(data, GLTR_criterion, "rank_GLTR", logger=logger))
        clf = outputs[0]['clf']
        filename = training_args.output_dir + '/classifier.bin'
        pickle.dump(clf, open(filename, 'wb'))
        # save metrics
        test_metrics = {'eval_%s'%k:v for k, v in outputs[0]['general_test'].items()}
        file_name = training_args.output_dir + '/predict_results.json'
        json.dump(test_metrics, open(file_name, 'w'))
    
    
    
    
    else:
        raise ValueError("Invalid model abbreviation")


if __name__ == "__main__":
    main()

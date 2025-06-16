"""
Claude-to-CodeLlama için Bilgi Damıtımı Eğitim Sistemi

Bu modül, gelişmiş teknikler kullanarak Claude Opus 4 (öğretmen) modelinden
Code Llama (öğrenci) modeline bilgi damıtan temel eğitim sistemini uygular.
"""

import os
import json
import logging
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig, get_scheduler
)
from peft import (
    LoraConfig, get_peft_model, TaskType, 
    prepare_model_for_kbit_training, PeftModel
)
from datasets import Dataset as HFDataset, load_from_disk, DatasetDict
import wandb
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Bilgi damıtımı eğitimi için yapılandırma."""
    
    # Model yapılandırması
    student_model_name: str = "codellama/CodeLlama-7b-hf"
    max_length: int = 2048
    
    # LoRA yapılandırması
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Kuantizasyon yapılandırması
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # Eğitim yapılandırması
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Damıtım yapılandırması
    distillation_weight: float = 0.7
    task_weight: float = 0.3
    temperature: float = 4.0
    
    # Optimizasyon
    optimizer_type: str = "adamw"
    lr_scheduler_type: str = "cosine"
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    
    # İzleme
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Yollar
    output_dir: str = "./models/distilled_codellama"
    dataset_path: str = "./data/generated"
    cache_dir: str = "./cache"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'DistillationConfig':
        """YAML dosyasından yapılandırmayı yükler."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        distillation_config = config_dict.get('distillation', {})
        student_config = config_dict.get('student_model', {})
        
        # Yapılandırmaları birleştir
        merged_config = {**distillation_config, **student_config}
        
        return cls(**merged_config)


class CodeDataset(Dataset):
    """Kod üretimi eğitimi için veri seti."""
    
    def __init__(
        self, 
        dataset: HFDataset, 
        tokenizer: AutoTokenizer, 
        max_length: int = 2048
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Gerekirse özel token'ları ekle
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.dataset[idx]
        
        # Girdiyi formatla
        instruction = example['instruction']
        output = example['output']
        
        # Tam metni oluştur
        full_text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}{self.tokenizer.eos_token}"
        
        # Token'lara ayır
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # Etiketleri oluştur (nedensel DM için input_ids ile aynı)
        labels = input_ids.clone()
        
        # Etiketlerde talimat kısmını maskele (sadece yanıt üzerinde eğit)
        instruction_text = f"### Instruction:\n{instruction}\n\n### Response:\n"
        instruction_encoding = self.tokenizer(
            instruction_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        instruction_length = instruction_encoding["input_ids"].shape[1]
        labels[:instruction_length] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class DistillationLoss(nn.Module):
    """Bilgi damıtımı için özel kayıp fonksiyonu."""
    
    def __init__(
        self, 
        temperature: float = 4.0,
        distillation_weight: float = 0.7,
        task_weight: float = 0.3
    ):
        super().__init__()
        self.temperature = temperature
        self.distillation_weight = distillation_weight
        self.task_weight = task_weight
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(
        self, 
        student_logits: torch.Tensor,
        teacher_logits: Optional[torch.Tensor],
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Damıtım kaybını hesaplar.
        
        Args:
            student_logits: Öğrenci modelden logitler
            teacher_logits: Öğretmen modelden logitler (isteğe bağlı)
            labels: Gerçek etiketler
            
        Returns:
            Toplam kayıp ve kayıp bileşenleri
        """
        # Görev kaybı (standart çapraz entropi)
        task_loss = self.ce_loss(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )
        
        loss_dict = {"task_loss": task_loss}
        
        if teacher_logits is not None:
            # Damıtım kaybı (KL ıraksaması)
            student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
            
            distillation_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
            loss_dict["distillation_loss"] = distillation_loss
            
            # Birleşik kayıp
            total_loss = (
                self.task_weight * task_loss + 
                self.distillation_weight * distillation_loss
            )
        else:
            # Öğretmen logitleri yoksa sadece görev kaybı
            total_loss = task_loss
            loss_dict["distillation_loss"] = torch.tensor(0.0)
        
        loss_dict["total_loss"] = total_loss
        
        return total_loss, loss_dict


class DistillationTrainer(Trainer):
    """Bilgi damıtımı için özel eğitici."""
    
    def __init__(self, distillation_loss_fn: DistillationLoss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distillation_loss_fn = distillation_loss_fn
        self.loss_history = []
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Özel damıtım kaybını hesaplar."""
        labels = inputs.get("labels")
        
        # İleri geçiş
        outputs = model(**inputs)
        student_logits = outputs.get("logits")
        
        # Şimdilik eğitim sırasında öğretmen logitlerimiz yok
        # Tam bir uygulamada burada öğretmen tahminlerini alırdınız
        teacher_logits = None
        
        # Kaybı hesapla
        loss, loss_dict = self.distillation_loss_fn(
            student_logits, teacher_logits, labels
        )
        
        # Kayıp bileşenlerini logla
        self.loss_history.append(loss_dict)
        
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: Dict[str, float]) -> None:
        """Kayıp bileşenleri ile gelişmiş loglama."""
        if self.loss_history:
            # Son kayıp bileşenlerinin ortalaması
            recent_losses = self.loss_history[-10:]  # Son 10 grup
            
            avg_losses = {}
            for key in recent_losses[0].keys():
                avg_losses[f"avg_{key}"] = torch.stack([
                    loss[key] for loss in recent_losses
                ]).mean().item()
            
            logs.update(avg_losses)
        
        super().log(logs)


class KnowledgeDistillationSystem:
    """Bilgi damıtımı eğitimi için ana sistem."""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Bileşenleri başlat
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Çıktı dizinini oluştur
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Bilgi damıtımı sistemi başlatıldı")
        logger.info(f"Cihaz: {self.device}")
        logger.info(f"Öğrenci model: {config.student_model_name}")
    
    def _get_target_modules(self, model_name: str) -> List[str]:
        """Model tipine göre uygun LoRA target modüllerini döndürür."""
        model_name_lower = model_name.lower()
        
        if "codellama" in model_name_lower or "llama" in model_name_lower:
            # Llama-based models
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "dialogpt" in model_name_lower or "gpt" in model_name_lower:
            # GPT-based models  
            return ["c_attn", "c_proj", "c_fc"]
        elif "bert" in model_name_lower:
            # BERT-based models
            return ["query", "key", "value", "dense"]
        elif "t5" in model_name_lower:
            # T5-based models
            return ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]
        else:
            # Default: try common patterns
            logger.warning(f"Unknown model type for {model_name}, using default target modules")
            return ["q_proj", "v_proj", "k_proj", "o_proj"]

    def setup_model_and_tokenizer(self):
        """Öğrenci model ve tokenizer'ı kurar."""
        logger.info("Tokenizer ve model yükleniyor...")
        
        # Tokenizer'ı yükle
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.student_model_name,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True
        )
        
        # Özel token'ları ekle
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Kuantizasyon yapılandırması
        if self.config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            )
        else:
            quantization_config = None
        
        # Modeli yükle
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.student_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            cache_dir=self.config.cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.config.use_mixed_precision else torch.float32
        )
        
        # Model tipine göre target modülleri belirle
        target_modules = self._get_target_modules(self.config.student_model_name)
        logger.info(f"Model {self.config.student_model_name} için target modüller: {target_modules}")
        
        # Modeli eğitime hazırla
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA'yı kur
        try:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            self.model = get_peft_model(self.model, lora_config)
            logger.info("LoRA başarıyla uygulandı")
            
        except Exception as e:
            logger.warning(f"LoRA uygulanırken hata: {e}")
            logger.info("Model layer isimlerini kontrol ediyoruz...")
            
            # Model layer isimlerini kontrol et
            available_modules = []
            for name, module in self.model.named_modules():
                if hasattr(module, 'weight') and len(module.weight.shape) >= 2:
                    available_modules.append(name)
            
            logger.info(f"Mevcut linear layer'lar: {available_modules[:10]}...")  # İlk 10'unu göster
            
            # Basit fallback: sadece attention layer'ları
            fallback_targets = [name for name in available_modules if any(
                pattern in name.lower() for pattern in ['attn', 'attention', 'query', 'key', 'value', 'proj']
            )][:4]  # En fazla 4 tane
            
            if fallback_targets:
                logger.info(f"Fallback target modüller kullanılıyor: {fallback_targets}")
                lora_config = LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    target_modules=fallback_targets,
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                self.model = get_peft_model(self.model, lora_config)
                logger.info("Fallback LoRA başarıyla uygulandı")
            else:
                logger.warning("LoRA uygulanamadı, tam model eğitimi kullanılacak")
        
        # Gradyan kontrol noktalarını etkinleştir
        if self.config.use_gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable()
            except Exception as e:
                logger.warning(f"Gradient checkpointing etkinleştirilemedi: {e}")
        
        # Eğitilebilir parametreleri yazdır
        if hasattr(self.model, 'print_trainable_parameters'):
            self.model.print_trainable_parameters()
        else:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Toplam parametreler: {total_params:,}")
            logger.info(f"Eğitilebilir parametreler: {trainable_params:,}")
            logger.info(f"Eğitilebilir oran: %{100 * trainable_params / total_params:.2f}")
        
        logger.info("Model ve tokenizer kurulumu tamamlandı")
    
    def _load_jsonl_dataset(self, dataset_path: str) -> DatasetDict:
        """JSONL dosyalarından veri seti yükler."""
        dataset_dict = {}
        
        # Her split için JSONL dosyalarını yükle
        splits = ["train", "validation", "test"]
        for split in splits:
            jsonl_path = Path(dataset_path) / f"{split}.jsonl"
            if jsonl_path.exists():
                logger.info(f"{split} split yükleniyor: {jsonl_path}")
                data = []
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                
                if data:
                    dataset_dict[split] = HFDataset.from_list(data)
                    logger.info(f"{len(data)} örnek yüklendi ({split})")
                else:
                    logger.warning(f"{split} split boş")
            else:
                logger.warning(f"{split} split dosyası bulunamadı: {jsonl_path}")
        
        return DatasetDict(dataset_dict)

    def load_dataset(self) -> Tuple[CodeDataset, CodeDataset]:
        """Veri setlerini yükler ve hazırlar."""
        logger.info(f"{self.config.dataset_path} konumundan veri seti yükleniyor")
        
        # Önce HuggingFace dataset dizini olup olmadığını kontrol et
        hf_dataset_path = Path(self.config.dataset_path)
        if (hf_dataset_path / "dataset_info.json").exists():
            # HuggingFace dataset formatı
            dataset_dict = load_from_disk(self.config.dataset_path)
        else:
            # JSONL dosyalarından yükle
            dataset_dict = self._load_jsonl_dataset(self.config.dataset_path)
        
        # Create datasets
        train_dataset = CodeDataset(
            dataset_dict["train"], 
            self.tokenizer, 
            self.config.max_length
        )
        
        eval_dataset = CodeDataset(
            dataset_dict["validation"], 
            self.tokenizer, 
            self.config.max_length
        )
        
        logger.info(f"{len(train_dataset)} eğitim örneği yüklendi")
        logger.info(f"{len(eval_dataset)} doğrulama örneği yüklendi")
        
        return train_dataset, eval_dataset
    
    def setup_trainer(self, train_dataset: CodeDataset, eval_dataset: CodeDataset):
        """Eğiticiyi kurar."""
        logger.info("Eğitici kuruluyor...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.config.use_mixed_precision,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="wandb" if wandb.run else None,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Distillation loss
        distillation_loss = DistillationLoss(
            temperature=self.config.temperature,
            distillation_weight=self.config.distillation_weight,
            task_weight=self.config.task_weight
        )
        
        # Create trainer
        self.trainer = DistillationTrainer(
            distillation_loss_fn=distillation_loss,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        logger.info("Eğitici kurulumu tamamlandı")
    
    def train(self):
        """Eğitim sürecini çalıştırır."""
        logger.info("Eğitim başlatılıyor...")
        
        # Modeli eğit
        train_result = self.trainer.train()
        
        # Son modeli kaydet
        self.trainer.save_model()
        self.trainer.save_state()
        
        # Eğitim sonuçlarını logla
        logger.info("Eğitim tamamlandı!")
        logger.info(f"Eğitim kaybı: {train_result.training_loss:.4f}")
        
        return train_result
    
    def evaluate(self) -> Dict[str, float]:
        """Modeli değerlendirir."""
        logger.info("Değerlendirme çalıştırılıyor...")
        
        eval_results = self.trainer.evaluate()
        
        logger.info("Değerlendirme tamamlandı!")
        for key, value in eval_results.items():
            logger.info(f"{key}: {value:.4f}")
        
        return eval_results
    
    def save_model(self, save_path: Optional[str] = None):
        """Eğitilmiş modeli kaydeder."""
        if save_path is None:
            save_path = self.config.output_dir
        
        # Modeli kaydet
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Yapılandırmayı kaydet
        config_path = Path(save_path) / "distillation_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Model {save_path} konumuna kaydedildi")
    
    def run_full_training(self):
        """Tam eğitim pipeline'ını çalıştırır."""
        logger.info("Tam eğitim pipeline'ı başlatılıyor...")
        
        # Model ve tokenizer'ı kur
        self.setup_model_and_tokenizer()
        
        # Veri setlerini yükle
        train_dataset, eval_dataset = self.load_dataset()
        
        # Eğiticiyi kur
        self.setup_trainer(train_dataset, eval_dataset)
        
        # Eğit
        train_result = self.train()
        
        # Değerlendir
        eval_results = self.evaluate()
        
        # Modeli kaydet
        self.save_model()
        
        logger.info("Tam eğitim pipeline'ı tamamlandı!")
        
        return {
            "train_result": train_result,
            "eval_results": eval_results
        }


# Örnek kullanım
def main():
    """Örnek eğitim çalışması."""
    # Yapılandırmayı yükle
    config = DistillationConfig.from_yaml("configs/config.yml")
    
    # wandb'yi başlat (isteğe bağlı)
    if os.getenv("WANDB_API_KEY"):
        wandb.init(
            project="claude-to-codellama-distillation",
            config=config.__dict__
        )
    
    # Eğitim sistemini oluştur
    system = KnowledgeDistillationSystem(config)
    
    # Eğitimi çalıştır
    results = system.run_full_training()
    
    print("Eğitim tamamlandı!")
    print(f"Son eğitim kaybı: {results['train_result'].training_loss:.4f}")
    print(f"Son değerlendirme kaybı: {results['eval_results']['eval_loss']:.4f}")


if __name__ == "__main__":
    main()


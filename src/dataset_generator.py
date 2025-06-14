"""
Claude-to-CodeLlama Bilgi Damıtımı için Veri Seti Üretim Pipeline'ı

Bu modül, Code Llama'yı eğitmek için kod örnekleri üretmek üzere Claude Opus 4'ü
öğretmen model olarak kullanarak yüksek kaliteli talimat-yanıt çiftleri oluşturur.
"""

import os
import json
import asyncio
import logging
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import yaml
from datetime import datetime

import pandas as pd
from tqdm.asyncio import tqdm
from datasets import Dataset, DatasetDict
import numpy as np

from claude_client import ClaudeAPIClient, ClaudeConfig, CodeGenerationRequest, CodeGenerationResponse

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Veri seti üretimi için yapılandırma."""
    target_size: int = 25000
    languages: List[str] = None
    language_distribution: Dict[str, int] = None
    difficulty_distribution: Dict[str, int] = None
    style_distribution: Dict[str, int] = None
    output_dir: str = "./data/generated"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["python", "javascript", "java", "cpp", "go", "rust"]
        if self.language_distribution is None:
            self.language_distribution = {
                "python": 40, "javascript": 25, "java": 15,
                "cpp": 10, "go": 5, "rust": 5
            }
        if self.difficulty_distribution is None:
            self.difficulty_distribution = {"easy": 30, "medium": 50, "hard": 20}
        if self.style_distribution is None:
            self.style_distribution = {"clean": 50, "documented": 30, "optimized": 20}


class InstructionGenerator:
    """Kod üretimi için çeşitli programlama talimatları üretir."""
    
    def __init__(self, config_path: str = "configs/config.yml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.instruction_templates = self.config.get('instruction_templates', {})
        self.task_categories = self.config.get('task_categories', {})
        
    def _get_random_task(self, category: str) -> str:
        """Bir kategoriden rastgele bir görev alır."""
        tasks = self.task_categories.get(category, [])
        return random.choice(tasks) if tasks else "bir çözüm uygula"
    
    def _fill_template(self, template: str, language: str, difficulty: str) -> str:
        """Talimat şablonunu uygun içerikle doldurur."""
        # Zorluk seviyesine göre rastgele görev al
        if difficulty == "easy":
            categories = ["utilities", "mathematical"]
        elif difficulty == "medium":
            categories = ["data_structures", "algorithms", "web_development"]
        else:  # hard
            categories = ["algorithms", "web_development", "data_structures"]
        
        category = random.choice(categories)
        task = self._get_random_task(category)
        
        # Şablonu doldur
        filled = template.format(
            task=task,
            problem=task,
            concept=task.split()[-1] if task.split() else "solution",
            algorithm=task.split()[0] if task.split() else "algorithm",
            features=f"advanced {task}"
        )
        
        return filled
    
    def generate_instruction(self, language: str, difficulty: str) -> str:
        """Bir programlama talimatı üretir."""
        templates = self.instruction_templates.get(language, {}).get(difficulty, [])
        
        if not templates:
            # Yedek şablonlar
            fallback_templates = [
                f"Write a {language} function to {{task}}",
                f"Implement {{concept}} in {language}",
                f"Create a {language} solution for {{problem}}"
            ]
            templates = fallback_templates
        
        template = random.choice(templates)
        return self._fill_template(template, language, difficulty)


class DatasetQualityController:
    """Veri seti kalitesini kontrol eder ve doğrular."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.seen_instructions = set()
        self.quality_stats = {
            "total_generated": 0,
            "duplicates_removed": 0,
            "quality_filtered": 0,
            "syntax_errors": 0,
            "accepted": 0
        }
    
    def is_duplicate(self, instruction: str) -> bool:
        """Talimatın tekrar olup olmadığını kontrol eder."""
        instruction_hash = hash(instruction.lower().strip())
        if instruction_hash in self.seen_instructions:
            self.quality_stats["duplicates_removed"] += 1
            return True
        self.seen_instructions.add(instruction_hash)
        return False
    
    def validate_code_quality(self, response: CodeGenerationResponse) -> bool:
        """Üretilen kod kalitesini doğrular."""
        if not response.success:
            return False
        
        code = response.generated_code.strip()
        
        # Temel kalite kontrolleri
        if len(code) < 50:  # Çok kısa
            self.quality_stats["quality_filtered"] += 1
            return False
        
        if len(code) > 2048:  # Çok uzun
            self.quality_stats["quality_filtered"] += 1
            return False
        
        # Dile özel kontroller
        if response.language == "python":
            if not self._validate_python_code(code):
                self.quality_stats["syntax_errors"] += 1
                return False
        
        self.quality_stats["accepted"] += 1
        return True
    
    def _validate_python_code(self, code: str) -> bool:
        """Python kod sözdizimini doğrular."""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Kalite kontrol istatistiklerini alır."""
        total = self.quality_stats["total_generated"]
        if total == 0:
            return self.quality_stats
        
        return {
            **self.quality_stats,
            "acceptance_rate": self.quality_stats["accepted"] / total,
            "duplicate_rate": self.quality_stats["duplicates_removed"] / total,
            "quality_filter_rate": self.quality_stats["quality_filtered"] / total,
            "syntax_error_rate": self.quality_stats["syntax_errors"] / total
        }


class DatasetGenerator:
    """Ana veri seti üretim pipeline'ı."""
    
    def __init__(self, dataset_config: DatasetConfig, claude_config: ClaudeConfig):
        self.dataset_config = dataset_config
        self.claude_client = ClaudeAPIClient(claude_config)
        self.instruction_generator = InstructionGenerator()
        self.quality_controller = DatasetQualityController(dataset_config)
        
        # Çıktı dizinini oluştur
        Path(dataset_config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"{dataset_config.target_size} örnek için veri seti üretici başlatıldı")
    
    def _calculate_distribution(self) -> Dict[str, Dict[str, int]]:
        """Her kategori için kaç örnek üretileceğini hesaplar."""
        distribution = {}
        
        for language in self.dataset_config.languages:
            lang_count = int(
                self.dataset_config.target_size * 
                self.dataset_config.language_distribution[language] / 100
            )
            
            distribution[language] = {}
            for difficulty in ["easy", "medium", "hard"]:
                diff_count = int(
                    lang_count * 
                    self.dataset_config.difficulty_distribution[difficulty] / 100
                )
                distribution[language][difficulty] = diff_count
        
        return distribution
    
    def _generate_requests(self) -> List[CodeGenerationRequest]:
        """Tüm kod üretim isteklerini oluşturur."""
        distribution = self._calculate_distribution()
        requests = []
        
        for language, difficulties in distribution.items():
            for difficulty, count in difficulties.items():
                for _ in range(count):
                    # Talimat üret
                    instruction = self.instruction_generator.generate_instruction(language, difficulty)
                    
                    # Tekrarları atla
                    if self.quality_controller.is_duplicate(instruction):
                        continue
                    
                    # Stil seç
                    style = np.random.choice(
                        list(self.dataset_config.style_distribution.keys()),
                        p=[v/100 for v in self.dataset_config.style_distribution.values()]
                    )
                    
                    request = CodeGenerationRequest(
                        instruction=instruction,
                        language=language,
                        difficulty=difficulty,
                        style=style,
                        max_tokens=2048,
                        temperature=0.1
                    )
                    requests.append(request)
        
        # Daha iyi dağılım için istekleri karıştır
        random.shuffle(requests)
        logger.info(f"{len(requests)} kod üretim isteği oluşturuldu")
        
        return requests
    
    async def generate_dataset(
        self, 
        max_concurrent: int = 5,
        save_intermediate: bool = True
    ) -> Dataset:
        """Tam veri setini üretir."""
        logger.info("Veri seti üretimi başlatılıyor...")
        
        # İstekleri üret
        requests = self._generate_requests()
        
        # İlerlemeyi takip et
        successful_examples = []
        failed_count = 0
        
        def progress_callback(response: CodeGenerationResponse):
            self.quality_controller.quality_stats["total_generated"] += 1
            
            if self.quality_controller.validate_code_quality(response):
                example = {
                    "instruction": response.instruction,
                    "input": "",  # Kod üretim görevleri için boş
                    "output": response.generated_code,
                    "language": response.language,
                    "tokens_used": response.tokens_used,
                    "generation_time": response.generation_time,
                    "metadata": response.metadata
                }
                successful_examples.append(example)
                
                # Ara sonuçları kaydet
                if save_intermediate and len(successful_examples) % 1000 == 0:
                    self._save_intermediate(successful_examples)
            else:
                nonlocal failed_count
                failed_count += 1
        
        # Kod örneklerini gruplar halinde üret
        batch_size = 100
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            logger.info(f"Grup işleniyor {i//batch_size + 1}/{(len(requests)-1)//batch_size + 1}")
            
            responses = await self.claude_client.generate_batch(
                batch, 
                max_concurrent=max_concurrent,
                progress_callback=progress_callback
            )
            
            # İlerlemeyi logla
            logger.info(f"Grup tamamlandı. Başarılı: {len(successful_examples)}, Başarısız: {failed_count}")
        
        # Son veri setini oluştur
        if successful_examples:
            dataset = Dataset.from_list(successful_examples)
            logger.info(f"Veri seti üretimi tamamlandı: {len(dataset)} örnek")
        else:
            logger.error("Başarılı örnek üretilemedi!")
            dataset = Dataset.from_list([])
        
        return dataset
    
    def _save_intermediate(self, examples: List[Dict[str, Any]]):
        """Ara sonuçları kaydeder."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = Path(self.dataset_config.output_dir) / f"intermediate_{timestamp}.jsonl"
        
        with open(filepath, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"{len(examples)} ara örnek {filepath} konumuna kaydedildi")
    
    def split_dataset(self, dataset: Dataset) -> DatasetDict:
        """Veri setini eğitim/doğrulama/test setlerine böler."""
        # Veri setini karıştır
        dataset = dataset.shuffle(seed=42)
        
        # Bölünme boyutlarını hesapla
        total_size = len(dataset)
        train_size = int(total_size * self.dataset_config.train_split)
        val_size = int(total_size * self.dataset_config.val_split)
        
        # Bölünmeleri oluştur
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, total_size))
        
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })
        
        logger.info(f"Veri seti bölündü: Eğitim={len(train_dataset)}, "
                   f"Doğrulama={len(val_dataset)}, Test={len(test_dataset)}")
        
        return dataset_dict
    
    def save_dataset(self, dataset_dict: DatasetDict, format: str = "jsonl"):
        """Veri setini diske kaydeder."""
        output_dir = Path(self.dataset_config.output_dir)
        
        for split_name, split_dataset in dataset_dict.items():
            if format == "jsonl":
                filepath = output_dir / f"{split_name}.jsonl"
                with open(filepath, 'w') as f:
                    for example in split_dataset:
                        f.write(json.dumps(example) + '\n')
            
            elif format == "parquet":
                filepath = output_dir / f"{split_name}.parquet"
                split_dataset.to_parquet(str(filepath))
            
            elif format == "huggingface":
                filepath = output_dir / split_name
                split_dataset.save_to_disk(str(filepath))
            
            logger.info(f"{split_name} bölümü {filepath} konumuna kaydedildi")
        
        # Meta verileri kaydet
        metadata = {
            "dataset_config": self.dataset_config.__dict__,
            "generation_stats": self.claude_client.get_statistics(),
            "quality_stats": self.quality_controller.get_quality_report(),
            "timestamp": datetime.now().isoformat()
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Meta veriler {metadata_path} konumuna kaydedildi")
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Kapsamlı kalite raporu üretir."""
        claude_stats = self.claude_client.get_statistics()
        quality_stats = self.quality_controller.get_quality_report()
        
        report = {
            "generation_summary": {
                "target_size": self.dataset_config.target_size,
                "actual_size": quality_stats["accepted"],
                "completion_rate": quality_stats["accepted"] / self.dataset_config.target_size,
                "total_api_calls": claude_stats["total_requests"],
                "total_cost": claude_stats["total_cost"],
                "average_cost_per_example": claude_stats["total_cost"] / max(quality_stats["accepted"], 1)
            },
            "quality_metrics": quality_stats,
            "api_statistics": claude_stats,
            "language_distribution": self.dataset_config.language_distribution,
            "difficulty_distribution": self.dataset_config.difficulty_distribution
        }
        
        return report


# Örnek kullanım ve test
async def main():
    """Örnek veri seti üretimi."""
    # Konfigürasyonları yükle
    claude_config = ClaudeConfig(
        api_key=os.getenv("ANTHROPIC_API_KEY", "your-api-key-here"),
        model="claude-3-opus-20240229",
        max_tokens=2048,
        temperature=0.1,
        rate_limit_rpm=50
    )
    
    dataset_config = DatasetConfig(
        target_size=100,  # Küçük test
        languages=["python", "javascript"],
        output_dir="./data/test_generation"
    )
    
    # Üreticiyi başlat
    generator = DatasetGenerator(dataset_config, claude_config)
    
    # Veri setini üret
    print("Veri seti üretiliyor...")
    dataset = await generator.generate_dataset(max_concurrent=2)
    
    if len(dataset) > 0:
        # Veri setini böl
        dataset_dict = generator.split_dataset(dataset)
        
        # Veri setini kaydet
        generator.save_dataset(dataset_dict, format="jsonl")
        
        # Rapor üret
        report = generator.generate_quality_report()
        print("\n--- ÜRETİM RAPORU ---")
        for section, data in report.items():
            print(f"\n{section.upper()}:")
            for key, value in data.items():
                print(f"  {key}: {value}")
    else:
        print("Hiç örnek üretilemedi!")


if __name__ == "__main__":
    asyncio.run(main())


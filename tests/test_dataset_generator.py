"""
Veri seti üreticisi modülü için testler.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from src.dataset_generator import (
    DatasetGenerator, DatasetConfig, InstructionGenerator, DatasetQualityController
)
from src.claude_client import ClaudeConfig, CodeGenerationResponse


class TestDatasetConfig:
    """Veri seti yapılandırma sınıfını test eder."""
    
    def test_default_config(self):
        """Varsayılan yapılandırma değerlerini test eder."""
        config = DatasetConfig()
        assert config.target_size == 25000
        assert "python" in config.languages
        assert "javascript" in config.languages
        assert config.train_split == 0.8
        assert config.val_split == 0.1
        assert config.test_split == 0.1
    
    def test_custom_config(self):
        """Özel yapılandırma değerlerini test eder."""
        config = DatasetConfig(
            target_size=1000,
            languages=["python", "java"],
            train_split=0.7,
            val_split=0.2,
            test_split=0.1
        )
        assert config.target_size == 1000
        assert len(config.languages) == 2
        assert config.train_split == 0.7
    
    def test_post_init_defaults(self):
        """post_init metodunun varsayılanları doğru ayarladığını test eder."""
        config = DatasetConfig()
        assert config.language_distribution is not None
        assert config.difficulty_distribution is not None
        assert config.style_distribution is not None
        assert sum(config.language_distribution.values()) == 100
        assert sum(config.difficulty_distribution.values()) == 100


class TestInstructionGenerator:
    """Talimat üreticisi fonksiyonalitesini test eder."""
    
    def setup_method(self):
        """Setup test environment."""
        # Geçici bir yapılandırma dosyası oluştur
        self.temp_config = {
            "instruction_templates": {
                "python": {
                    "easy": ["Write a Python function to {task}"],
                    "medium": ["Implement {algorithm} in Python"],
                    "hard": ["Design a Python system that {task}"]
                }
            },
            "task_categories": {
                "utilities": ["calculate sum", "format string"],
                "algorithms": ["binary search", "merge sort"]
            }
        }
        
        with patch('yaml.safe_load', return_value=self.temp_config):
            with patch('builtins.open'):
                self.generator = InstructionGenerator()
    
    def test_initialization(self):
        """Talimat üreticisi başlatılmasını test eder."""
        assert self.generator.instruction_templates is not None
        assert self.generator.task_categories is not None
    
    def test_get_random_task(self):
        """Kategoriden rastgele görev alma işlemini test eder."""
        task = self.generator._get_random_task("utilities")
        assert task in ["calculate sum", "format string"]
    
    def test_get_random_task_missing_category(self):
        """Eksik kategoriden görev alma işlemini test eder."""
        task = self.generator._get_random_task("nonexistent")
        assert task == "implement a solution"
    
    def test_generate_instruction(self):
        """Talimat üretimini test eder."""
        instruction = self.generator.generate_instruction("python", "easy")
        assert "Python function" in instruction
        assert len(instruction) > 0
    
    def test_generate_instruction_fallback(self):
        """Yedek seçenek ile talimat üretimini test eder."""
        # Şablonlarda olmayan dil ile test et
        instruction = self.generator.generate_instruction("rust", "medium")
        assert "rust" in instruction.lower()
        assert len(instruction) > 0


class TestDatasetQualityController:
    """Test dataset quality controller."""
    
    def setup_method(self):
        """Setup test environment."""
        config = DatasetConfig(target_size=100)
        self.controller = DatasetQualityController(config)
    
    def test_initialization(self):
        """Kalite kontrolörü başlatılmasını test eder."""
        assert len(self.controller.seen_instructions) == 0
        assert self.controller.quality_stats["total_generated"] == 0
        assert self.controller.quality_stats["accepted"] == 0
    
    def test_duplicate_detection(self):
        """Yinelenen talimat tespitini test eder."""
        instruction = "Write a Python function to calculate sum"
        
        # İlk kez yinelenen olmamalıdır
        assert not self.controller.is_duplicate(instruction)
        
        # İkinci kez yinelenen olmalıdır
        assert self.controller.is_duplicate(instruction)
        assert self.controller.quality_stats["duplicates_removed"] == 1
    
    def test_duplicate_case_insensitive(self):
        """Yinelenen tespitinin büyük-küçük harf duyarlı olmadığını test eder."""
        instruction1 = "Write a Python function"
        instruction2 = "WRITE A PYTHON FUNCTION"
        
        assert not self.controller.is_duplicate(instruction1)
        assert self.controller.is_duplicate(instruction2)
    
    def test_code_quality_validation_success(self):
        """Başarılı kod kalite doğrulamasını test eder."""
        response = CodeGenerationResponse(
            instruction="Write a function",
            generated_code="def hello_world():\n    return 'Hello, World!'",
            language="python",
            tokens_used=100,
            generation_time=1.0,
            success=True
        )
        
        assert self.controller.validate_code_quality(response)
        assert self.controller.quality_stats["accepted"] == 1
    
    def test_code_quality_validation_failed_response(self):
        """Başarısız yanıt ile kod kalite doğrulamasını test eder."""
        response = CodeGenerationResponse(
            instruction="Write a function",
            generated_code="",
            language="python",
            tokens_used=0,
            generation_time=1.0,
            success=False,
            error="API Error"
        )
        
        assert not self.controller.validate_code_quality(response)
        assert self.controller.quality_stats["accepted"] == 0
    
    def test_code_quality_validation_too_short(self):
        """Çok kısa kod ile kalite doğrulamasını test eder."""
        response = CodeGenerationResponse(
            instruction="Write a function",
            generated_code="x=1",  # Too short
            language="python",
            tokens_used=10,
            generation_time=1.0,
            success=True
        )
        
        assert not self.controller.validate_code_quality(response)
        assert self.controller.quality_stats["quality_filtered"] == 1
    
    def test_code_quality_validation_too_long(self):
        """Çok uzun kod ile kalite doğrulamasını test eder."""
        response = CodeGenerationResponse(
            instruction="Write a function",
            generated_code="x = 1\n" * 500,  # Too long
            language="python",
            tokens_used=1000,
            generation_time=1.0,
            success=True
        )
        
        assert not self.controller.validate_code_quality(response)
        assert self.controller.quality_stats["quality_filtered"] == 1
    
    def test_python_syntax_validation(self):
        """Python sözdizimi doğrulamasını test eder."""
        # Geçerli Python kodu
        valid_code = "def hello():\n    return 'Hello'"
        assert self.controller._validate_python_code(valid_code)
        
        # Geçersiz Python kodu
        invalid_code = "def hello(\n    return 'Hello'"
        assert not self.controller._validate_python_code(invalid_code)
    
    def test_quality_report(self):
        """Test quality report generation."""
        # Simulate some statistics
        self.controller.quality_stats.update({
            "total_generated": 100,
            "accepted": 80,
            "duplicates_removed": 10,
            "quality_filtered": 8,
            "syntax_errors": 2
        })
        
        report = self.controller.get_quality_report()
        assert report["acceptance_rate"] == 0.8
        assert report["duplicate_rate"] == 0.1
        assert report["quality_filter_rate"] == 0.08
        assert report["syntax_error_rate"] == 0.02


class TestDatasetGenerator:
    """Test main dataset generator functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.claude_config = ClaudeConfig(api_key="test-key")
        self.dataset_config = DatasetConfig(
            target_size=10,
            languages=["python"],
            output_dir="/tmp/test_dataset"
        )
        
        # Talimat üreticisini taklit et
        with patch('src.dataset_generator.InstructionGenerator'):
            self.generator = DatasetGenerator(self.dataset_config, self.claude_config)
    
    def test_initialization(self):
        """Veri seti üreticisi başlatılmasını test eder."""
        assert self.generator.dataset_config == self.dataset_config
        assert self.generator.claude_client is not None
        assert self.generator.instruction_generator is not None
        assert self.generator.quality_controller is not None
    
    def test_distribution_calculation(self):
        """Dağılım hesaplamasını test eder."""
        distribution = self.generator._calculate_distribution()
        
        assert "python" in distribution
        assert "easy" in distribution["python"]
        assert "medium" in distribution["python"]
        assert "hard" in distribution["python"]
        
        # Toplamın doğru çıktığını kontrol et
        total = sum(
            sum(difficulties.values()) 
            for difficulties in distribution.values()
        )
        assert total <= self.dataset_config.target_size
    
    def test_generate_requests(self):
        """İstek üretimini test eder."""
        # Talimat üretimini taklit et
        with patch.object(self.generator.instruction_generator, 'generate_instruction', 
                         return_value="Write a Python function"):
            requests = self.generator._generate_requests()
            
            assert len(requests) > 0
            assert all(req.language == "python" for req in requests)
            assert all(req.instruction == "Write a Python function" for req in requests)
    
    @pytest.mark.asyncio
    async def test_generate_dataset_success(self):
        """Başarılı veri seti üretimini test eder."""
        # Başarılı yanıtları taklit et
        mock_responses = [
            CodeGenerationResponse(
                instruction=f"Write function {i}",
                generated_code=f"def func_{i}(): pass",
                language="python",
                tokens_used=100,
                generation_time=1.0,
                success=True
            ) for i in range(5)
        ]
        
        with patch.object(self.generator.claude_client, 'generate_batch', 
                         new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = mock_responses
            
            with patch.object(self.generator, '_generate_requests') as mock_requests:
                mock_requests.return_value = [Mock() for _ in range(5)]
                
                dataset = await self.generator.generate_dataset()
                
                assert len(dataset) == 5
                mock_batch.assert_called()
    
    @pytest.mark.asyncio
    async def test_generate_dataset_empty(self):
        """Başarılı yanıt olmadan veri seti üretimini test eder."""
        # Başarısız yanıtları taklit et
        mock_responses = [
            CodeGenerationResponse(
                instruction="Write function",
                generated_code="",
                language="python",
                tokens_used=0,
                generation_time=1.0,
                success=False,
                error="API Error"
            )
        ]
        
        with patch.object(self.generator.claude_client, 'generate_batch', 
                         new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = mock_responses
            
            with patch.object(self.generator, '_generate_requests') as mock_requests:
                mock_requests.return_value = [Mock()]
                
                dataset = await self.generator.generate_dataset()
                
                assert len(dataset) == 0
    
    def test_split_dataset(self):
        """Veri seti bölümlemesini test eder."""
        from datasets import Dataset
        
        # Sahte veri seti oluştur
        examples = [{"instruction": f"Task {i}", "output": f"Code {i}"} for i in range(100)]
        dataset = Dataset.from_list(examples)
        
        dataset_dict = self.generator.split_dataset(dataset)
        
        assert "train" in dataset_dict
        assert "validation" in dataset_dict
        assert "test" in dataset_dict
        
        train_size = len(dataset_dict["train"])
        val_size = len(dataset_dict["validation"])
        test_size = len(dataset_dict["test"])
        
        # Yaklaşık bölümleme oranlarını kontrol et
        total_size = train_size + val_size + test_size
        assert abs(train_size / total_size - 0.8) < 0.1
        assert abs(val_size / total_size - 0.1) < 0.1
        assert abs(test_size / total_size - 0.1) < 0.1
    
    def test_save_dataset_jsonl(self):
        """Veri setini JSONL formatında kaydetmeyi test eder."""
        from datasets import Dataset, DatasetDict
        
        with tempfile.TemporaryDirectory() as temp_dir:
            self.generator.dataset_config.output_dir = temp_dir
            
            # Sahte veri seti oluştur
            examples = [{"instruction": "Task", "output": "Code"}]
            dataset = Dataset.from_list(examples)
            dataset_dict = DatasetDict({"train": dataset})
            
            self.generator.save_dataset(dataset_dict, format="jsonl")
            
            # Dosyanın oluşturulduğunu kontrol et
            train_file = Path(temp_dir) / "train.jsonl"
            assert train_file.exists()
            
            # İçeriği kontrol et
            with open(train_file, 'r') as f:
                line = f.readline()
                data = json.loads(line)
                assert data["instruction"] == "Task"
                assert data["output"] == "Code"
    
    def test_generate_quality_report(self):
        """Test quality report generation."""
        # İstatistikleri taklit et
        with patch.object(self.generator.claude_client, 'get_statistics') as mock_claude_stats:
            mock_claude_stats.return_value = {
                "total_requests": 100,
                "total_cost": 10.0,
                "total_tokens": 50000
            }
            
            with patch.object(self.generator.quality_controller, 'get_quality_report') as mock_quality_stats:
                mock_quality_stats.return_value = {
                    "accepted": 80,
                    "total_generated": 100
                }
                
                report = self.generator.generate_quality_report()
                
                assert "generation_summary" in report
                assert "quality_metrics" in report
                assert "api_statistics" in report
                assert report["generation_summary"]["target_size"] == self.dataset_config.target_size
                assert report["generation_summary"]["actual_size"] == 80


if __name__ == "__main__":
    pytest.main([__file__])
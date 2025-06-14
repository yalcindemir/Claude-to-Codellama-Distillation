"""
Bilgi damıtma eğitim sistemi için testler.
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datasets import Dataset, DatasetDict

from src.distillation_trainer import (
    DistillationConfig, KnowledgeDistillationSystem, 
    CodeDataset, DistillationLoss, DistillationTrainer
)


class TestDistillationConfig:
    """Damıtma yapılandırma sınıfını test eder."""
    
    def test_default_config(self):
        """Varsayılan yapılandırma değerlerini test eder."""
        config = DistillationConfig()
        assert config.student_model_name == "codellama/CodeLlama-7b-hf"
        assert config.max_length == 2048
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.num_epochs == 3
        assert config.batch_size == 4
    
    def test_custom_config(self):
        """Özel yapılandırma değerlerini test eder."""
        config = DistillationConfig(
            student_model_name="codellama/CodeLlama-13b-hf",
            max_length=1024,
            lora_r=32,
            num_epochs=5,
            batch_size=2
        )
        assert config.student_model_name == "codellama/CodeLlama-13b-hf"
        assert config.max_length == 1024
        assert config.lora_r == 32
        assert config.num_epochs == 5
        assert config.batch_size == 2
    
    def test_from_yaml(self):
        """YAML'dan yapılandırma yüklemeyi test eder."""
        yaml_content = {
            'distillation': {
                'num_epochs': 2,
                'batch_size': 8,
                'learning_rate': 1e-4
            },
            'student_model': {
                'lora_r': 8,
                'lora_alpha': 16
            }
        }
        
        with patch('yaml.safe_load', return_value=yaml_content):
            with patch('builtins.open'):
                config = DistillationConfig.from_yaml('test.yml')
                
                assert config.num_epochs == 2
                assert config.batch_size == 8
                assert config.learning_rate == 1e-4
                assert config.lora_r == 8
                assert config.lora_alpha == 16


class TestCodeDataset:
    """Kod veri seti fonksiyonalitesini test eder."""
    
    def setup_method(self):
        """Test ortamını kurar."""
        # Sahte tokenizer oluştur
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.pad_token = None
        self.mock_tokenizer.eos_token = "</s>"
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
        }
        
        # Örnek veri seti oluştur
        examples = [
            {
                "instruction": "Write a Python function to add numbers",
                "output": "def add(a, b):\n    return a + b"
            },
            {
                "instruction": "Create a JavaScript function",
                "output": "function test() { return true; }"
            }
        ]
        self.hf_dataset = Dataset.from_list(examples)
    
    def test_dataset_initialization(self):
        """Veri seti başlatılmasını test eder."""
        dataset = CodeDataset(self.hf_dataset, self.mock_tokenizer, max_length=512)
        
        assert len(dataset) == 2
        assert dataset.max_length == 512
        assert dataset.tokenizer == self.mock_tokenizer
    
    def test_dataset_getitem(self):
        """Veri seti öğe alma işlemini test eder."""
        dataset = CodeDataset(self.hf_dataset, self.mock_tokenizer)
        
        item = dataset[0]
        
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)
    
    def test_tokenizer_pad_token_setup(self):
        """Pad tokenin doğru ayarlandığını test eder."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        
        dataset = CodeDataset(self.hf_dataset, mock_tokenizer)
        
        # pad_token'in ayarlandığını kontrol et
        assert mock_tokenizer.pad_token == mock_tokenizer.eos_token


class TestDistillationLoss:
    """Damıtma kayıp fonksiyonunu test eder."""
    
    def setup_method(self):
        """Test ortamını kurar."""
        self.vocab_size = 1000
        self.loss_fn = DistillationLoss(
            temperature=4.0,
            distillation_weight=0.7,
            task_weight=0.3
        )
    
    def test_loss_initialization(self):
        """Kayıp fonksiyonu başlatılmasını test eder."""
        assert self.loss_fn.temperature == 4.0
        assert self.loss_fn.distillation_weight == 0.7
        assert self.loss_fn.task_weight == 0.3
    
    def test_loss_with_teacher_logits(self):
        """Öğretmen logitleri ile kayıp hesaplamasını test eder."""
        batch_size, seq_len, vocab_size = 2, 10, self.vocab_size
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        total_loss, loss_dict = self.loss_fn(student_logits, teacher_logits, labels)
        
        assert isinstance(total_loss, torch.Tensor)
        assert "task_loss" in loss_dict
        assert "distillation_loss" in loss_dict
        assert "total_loss" in loss_dict
        assert loss_dict["distillation_loss"] > 0
    
    def test_loss_without_teacher_logits(self):
        """Öğretmen logitleri olmadan kayıp hesaplamasını test eder."""
        batch_size, seq_len, vocab_size = 2, 10, self.vocab_size
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        total_loss, loss_dict = self.loss_fn(student_logits, None, labels)
        
        assert isinstance(total_loss, torch.Tensor)
        assert "task_loss" in loss_dict
        assert "distillation_loss" in loss_dict
        assert loss_dict["distillation_loss"] == 0.0
    
    def test_loss_with_ignored_tokens(self):
        """Göz ardı edilen tokenlar (-100) ile kayıp hesaplamasını test eder."""
        batch_size, seq_len, vocab_size = 2, 10, self.vocab_size
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Bazı etiketleri -100 olarak ayarla (göz ardı edilir)
        labels[:, :3] = -100
        
        total_loss, loss_dict = self.loss_fn(student_logits, None, labels)
        
        assert isinstance(total_loss, torch.Tensor)
        assert torch.isfinite(total_loss)


class TestDistillationTrainer:
    """Özel damıtma eğiticisini test eder."""
    
    def setup_method(self):
        """Test ortamını kurar."""
        self.loss_fn = DistillationLoss()
        
        # Eğitim argümanlarını taklit et
        mock_args = Mock()
        mock_args.output_dir = "/tmp/test"
        mock_args.logging_steps = 10
        
        # Modeli taklit et
        mock_model = Mock()
        
        # Veri setini taklit et
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        
        self.trainer = DistillationTrainer(
            distillation_loss_fn=self.loss_fn,
            model=mock_model,
            args=mock_args,
            train_dataset=mock_dataset
        )
    
    def test_trainer_initialization(self):
        """Eğitici başlatılmasını test eder."""
        assert self.trainer.distillation_loss_fn == self.loss_fn
        assert len(self.trainer.loss_history) == 0
    
    def test_compute_loss(self):
        """Özel kayıp hesaplamasını test eder."""
        # Modeli taklit et output
        mock_outputs = Mock()
        mock_outputs.get.return_value = torch.randn(2, 10, 1000)
        
        # Girdileri taklit et
        mock_inputs = {
            "labels": torch.randint(0, 1000, (2, 10)),
            "input_ids": torch.randint(0, 1000, (2, 10))
        }
        
        # Modeli taklit et call
        mock_model = Mock()
        mock_model.return_value = mock_outputs
        
        loss = self.trainer.compute_loss(mock_model, mock_inputs)
        
        assert isinstance(loss, torch.Tensor)
        assert len(self.trainer.loss_history) == 1


class TestKnowledgeDistillationSystem:
    """Ana bilgi damıtma sistemini test eder."""
    
    def setup_method(self):
        """Test ortamını kurar."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = temp_dir
            self.config = DistillationConfig(
                student_model_name="microsoft/DialoGPT-small",  # Test için küçük model
                output_dir=f"{temp_dir}/models",
                dataset_path=f"{temp_dir}/data",
                cache_dir=f"{temp_dir}/cache",
                num_epochs=1,
                batch_size=1,
                eval_steps=10,
                save_steps=10,
                logging_steps=1,
                use_4bit=False  # Test için devre dışı bırak
            )
            
            # Sahte veri seti oluştur
            self.create_mock_dataset()
    
    def create_mock_dataset(self):
        """Test için sahte veri seti oluşturur."""
        os.makedirs(self.config.dataset_path, exist_ok=True)
        
        # Eğitim verisi oluştur
        train_data = [
            {
                "instruction": "Write a function",
                "input": "",
                "output": "def test(): pass"
            }
        ]
        
        # Doğrulama verisi oluştur
        val_data = [
            {
                "instruction": "Write another function", 
                "input": "",
                "output": "def test2(): pass"
            }
        ]
        
        # DatasetDict olarak kaydet
        dataset_dict = DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data)
        })
        dataset_dict.save_to_disk(self.config.dataset_path)
    
    def test_system_initialization(self):
        """Sistem başlatılmasını test eder."""
        system = KnowledgeDistillationSystem(self.config)
        
        assert system.config == self.config
        assert system.tokenizer is None
        assert system.model is None
        assert system.trainer is None
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_system_with_cpu(self, mock_cuda):
        """CPU ile sistem başlatılmasını test eder."""
        system = KnowledgeDistillationSystem(self.config)
        assert system.device.type == "cpu"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_system_with_gpu(self):
        """GPU ile sistem başlatılmasını test eder."""
        system = KnowledgeDistillationSystem(self.config)
        if torch.cuda.is_available():
            assert system.device.type == "cuda"
    
    def test_load_dataset_missing_path(self):
        """Eksik yol ile veri seti yüklemeyi test eder."""
        config = DistillationConfig(dataset_path="/nonexistent/path")
        system = KnowledgeDistillationSystem(config)
        
        with pytest.raises((FileNotFoundError, ValueError)):
            system.load_dataset()
    
    def test_save_model(self):
        """Model kaydetmeyi test eder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = DistillationConfig(output_dir=temp_dir)
            system = KnowledgeDistillationSystem(config)
            
            # Modeli taklit et and tokenizer
            system.model = Mock()
            system.tokenizer = Mock()
            
            system.save_model()
            
            # Kaydetme metodlarının çağrıldığını kontrol et
            system.model.save_pretrained.assert_called_once()
            system.tokenizer.save_pretrained.assert_called_once()


class TestIntegration:
    """Tam sistem için entegrasyon testleri."""
    
    @pytest.mark.slow
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained') 
    def test_minimal_training_run(self, mock_tokenizer, mock_model):
        """Taklit edilmiş bileşenlerle minimal eğitim çalıştırmasını test eder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Yapılandırmayı ayarla
            config = DistillationConfig(
                student_model_name="microsoft/DialoGPT-small",
                output_dir=f"{temp_dir}/models",
                dataset_path=f"{temp_dir}/data",
                num_epochs=1,
                batch_size=1,
                eval_steps=1,
                save_steps=1,
                logging_steps=1,
                use_4bit=False,
                max_length=64
            )
            
            # Minimal veri seti oluştur
            os.makedirs(config.dataset_path, exist_ok=True)
            
            dataset_dict = DatasetDict({
                "train": Dataset.from_list([
                    {"instruction": "test", "input": "", "output": "def test(): pass"}
                ]),
                "validation": Dataset.from_list([
                    {"instruction": "test2", "input": "", "output": "def test2(): pass"}
                ])
            })
            dataset_dict.save_to_disk(config.dataset_path)
            
            # Tokenizer'ı taklit et
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.pad_token = None
            mock_tokenizer_instance.eos_token = "</s>"
            mock_tokenizer_instance.return_value = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]])
            }
            mock_tokenizer_instance.decode.return_value = "test"
            mock_tokenizer.return_value = mock_tokenizer_instance
            
            # Modeli taklit et
            mock_model_instance = Mock()
            mock_model_instance.config.vocab_size = 1000
            mock_model.return_value = mock_model_instance
            
            # Sistemi başlat
            system = KnowledgeDistillationSystem(config)
            
            # Bireysel bileşenleri test et
            system.setup_model_and_tokenizer()
            assert system.tokenizer is not None
            assert system.model is not None
            
            train_dataset, eval_dataset = system.load_dataset()
            assert len(train_dataset) == 1
            assert len(eval_dataset) == 1


class TestPerformance:
    """Performans ve bellek testleri."""
    
    def test_memory_usage_tracking(self):
        """Eğitim sırasında bellek kullanımı takibini test eder."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory testing")
        
        config = DistillationConfig(use_4bit=True)
        system = KnowledgeDistillationSystem(config)
        
        # Başlangıç belleğini kontrol et
        initial_memory = torch.cuda.memory_allocated()
        
        # Bu normalde model yükler ve bellek kullanımını kontrol eder
        # Test için sadece takibin çalıştığını doğrular
        assert isinstance(initial_memory, int)
        assert initial_memory >= 0
    
    def test_quantization_config(self):
        """Kuantizasyon yapılandırmasını test eder."""
        config = DistillationConfig(use_4bit=True)
        system = KnowledgeDistillationSystem(config)
        
        assert config.use_4bit is True
        assert config.bnb_4bit_compute_dtype == "float16"
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_use_double_quant is True


if __name__ == "__main__":
    pytest.main([__file__])
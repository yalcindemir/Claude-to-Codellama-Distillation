"""
Claude API istemci modülü için testler.
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch
from src.claude_client import (
    ClaudeAPIClient, ClaudeConfig, CodeGenerationRequest, 
    CodeGenerationResponse, RateLimiter
)


class TestClaudeConfig:
    """Claude yapılandırma sınıfını test eder."""
    
    def test_default_config(self):
        """Varsayılan yapılandırma değerlerini test eder."""
        config = ClaudeConfig(api_key="test-key")
        assert config.api_key == "test-key"
        assert config.model == "claude-3-opus-20240229"
        assert config.max_tokens == 4096
        assert config.temperature == 0.1
        assert config.rate_limit_rpm == 50
    
    def test_custom_config(self):
        """Özel yapılandırma değerlerini test eder."""
        config = ClaudeConfig(
            api_key="test-key",
            model="claude-3-sonnet-20240229",
            max_tokens=2048,
            temperature=0.5
        )
        assert config.model == "claude-3-sonnet-20240229"
        assert config.max_tokens == 2048
        assert config.temperature == 0.5


class TestRateLimiter:
    """Hız sınırlayıcı fonksiyonalitesini test eder."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_initialization(self):
        """Hız sınırlayıcı başlatılmasını test eder."""
        limiter = RateLimiter(rpm=10, tpm=1000)
        assert limiter.rpm == 10
        assert limiter.tpm == 1000
        assert len(limiter.request_times) == 0
        assert len(limiter.token_usage) == 0
    
    @pytest.mark.asyncio
    async def test_rate_limiter_no_wait(self):
        """Hız sınırlayıcının sınırlar altında beklemediğini test eder."""
        limiter = RateLimiter(rpm=100, tpm=10000)
        
        # Küçük kullanım için beklememelidir
        await limiter.wait_if_needed(100)
        
        # Kullanımın kayıt edildiğini kontrol et
        assert len(limiter.request_times) == 1
        assert len(limiter.token_usage) == 1


class TestCodeGenerationRequest:
    """Kod üretim isteği yapısını test eder."""
    
    def test_basic_request(self):
        """Temel istek oluşturmayı test eder."""
        request = CodeGenerationRequest(
            instruction="Write a Python function",
            language="python"
        )
        assert request.instruction == "Write a Python function"
        assert request.language == "python"
        assert request.difficulty == "medium"
        assert request.style == "clean"
    
    def test_custom_request(self):
        """Özel istek parametrelerini test eder."""
        request = CodeGenerationRequest(
            instruction="Write a JavaScript function",
            language="javascript",
            difficulty="hard",
            style="optimized",
            max_tokens=1024,
            temperature=0.2
        )
        assert request.difficulty == "hard"
        assert request.style == "optimized"
        assert request.max_tokens == 1024
        assert request.temperature == 0.2


class TestClaudeAPIClient:
    """Claude API istemci fonksiyonalitesini test eder."""
    
    def setup_method(self):
        """Test ortamını kurar."""
        self.config = ClaudeConfig(
            api_key="test-api-key",
            model="claude-3-opus-20240229",
            max_tokens=1024,
            temperature=0.1
        )
    
    def test_client_initialization(self):
        """İstemci başlatılmasını test eder."""
        client = ClaudeAPIClient(self.config)
        assert client.config == self.config
        assert client.total_requests == 0
        assert client.total_tokens == 0
        assert client.total_cost == 0.0
        assert client.failed_requests == 0
    
    def test_token_estimation(self):
        """Token sayısı tahminini test eder."""
        client = ClaudeAPIClient(self.config)
        text = "Hello world"
        estimated = client._estimate_tokens(text)
        assert estimated == len(text) // 4
    
    def test_cost_calculation(self):
        """Maliyet hesaplamasını test eder."""
        client = ClaudeAPIClient(self.config)
        cost = client._calculate_cost(1000, 500)
        expected_cost = (1000 * 15 + 500 * 75) / 1_000_000
        assert cost == expected_cost
    
    def test_prompt_creation(self):
        """Kod üretim prompt'u oluşturmayı test eder."""
        client = ClaudeAPIClient(self.config)
        request = CodeGenerationRequest(
            instruction="Write a Python function to sort a list",
            language="python",
            difficulty="medium",
            style="clean"
        )
        
        prompt = client._create_code_generation_prompt(request)
        assert "Write a Python function to sort a list" in prompt
        assert "python" in prompt.lower()
        assert "medium" in prompt.lower()
        assert "clean" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_generate_code_success(self):
        """Başarılı kod üretimini test eder."""
        client = ClaudeAPIClient(self.config)
        
        # API yanıtını taklit et
        mock_response = Mock()
        mock_response.content = [Mock(text="def sort_list(lst):\n    return sorted(lst)")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            request = CodeGenerationRequest(
                instruction="Write a Python function to sort a list",
                language="python"
            )
            
            response = await client.generate_code(request)
            
            assert response.success is True
            assert "def sort_list" in response.generated_code
            assert response.language == "python"
            assert response.tokens_used == 150
            assert response.error is None
    
    @pytest.mark.asyncio
    async def test_generate_code_failure(self):
        """Başarısız kod üretimini test eder."""
        client = ClaudeAPIClient(self.config)
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("API Error")
            
            request = CodeGenerationRequest(
                instruction="Write a Python function",
                language="python"
            )
            
            response = await client.generate_code(request)
            
            assert response.success is False
            assert response.generated_code == ""
            assert "API Error" in response.error
            assert client.failed_requests == 1
    
    @pytest.mark.asyncio
    async def test_generate_batch(self):
        """Toplu üretimi test eder."""
        client = ClaudeAPIClient(self.config)
        
        requests = [
            CodeGenerationRequest(
                instruction="Write a Python function to add numbers",
                language="python"
            ),
            CodeGenerationRequest(
                instruction="Write a JavaScript function to multiply numbers",
                language="javascript"
            )
        ]
        
        # Başarılı yanıtları taklit et
        mock_response1 = Mock()
        mock_response1.content = [Mock(text="def add(a, b):\n    return a + b")]
        mock_response1.usage.input_tokens = 50
        mock_response1.usage.output_tokens = 25
        
        mock_response2 = Mock()
        mock_response2.content = [Mock(text="function multiply(a, b) {\n    return a * b;\n}")]
        mock_response2.usage.input_tokens = 60
        mock_response2.usage.output_tokens = 30
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = [mock_response1, mock_response2]
            
            responses = await client.generate_batch(requests, max_concurrent=2)
            
            assert len(responses) == 2
            assert all(r.success for r in responses)
            assert "def add" in responses[0].generated_code
            assert "function multiply" in responses[1].generated_code
    
    def test_statistics(self):
        """İstatistik toplama işlemini test eder."""
        client = ClaudeAPIClient(self.config)
        
        # Biraz kullanım simülasyonu yap
        client.total_requests = 10
        client.failed_requests = 2
        client.total_tokens = 5000
        client.total_cost = 0.50
        
        stats = client.get_statistics()
        
        assert stats["total_requests"] == 10
        assert stats["failed_requests"] == 2
        assert stats["success_rate"] == 0.8
        assert stats["total_tokens"] == 5000
        assert stats["total_cost"] == 0.50
        assert stats["average_tokens_per_request"] == 500
        assert stats["average_cost_per_request"] == 0.05


class TestCodeGenerationDataset:
    """Kod üretim veri seti fonksiyonalitesini test eder."""
    
    def test_dataset_initialization(self):
        """Veri seti başlatılmasını test eder."""
        from src.claude_client import CodeGenerationDataset
        dataset = CodeGenerationDataset()
        assert len(dataset.examples) == 0
    
    def test_add_successful_response(self):
        """Başarılı yanıtın veri setine eklenmesini test eder."""
        from src.claude_client import CodeGenerationDataset
        dataset = CodeGenerationDataset()
        
        response = CodeGenerationResponse(
            instruction="Write a function",
            generated_code="def test(): pass",
            language="python",
            tokens_used=100,
            generation_time=1.5,
            success=True,
            metadata={"cost": 0.01}
        )
        
        dataset.add_response(response)
        assert len(dataset.examples) == 1
        assert dataset.examples[0]["instruction"] == "Write a function"
        assert dataset.examples[0]["code"] == "def test(): pass"
    
    def test_add_failed_response(self):
        """Başarısız yanıtların eklenmediğini test eder."""
        from src.claude_client import CodeGenerationDataset
        dataset = CodeGenerationDataset()
        
        response = CodeGenerationResponse(
            instruction="Write a function",
            generated_code="",
            language="python",
            tokens_used=0,
            generation_time=0.5,
            success=False,
            error="API Error"
        )
        
        dataset.add_response(response)
        assert len(dataset.examples) == 0
    
    def test_dataset_statistics(self):
        """Veri seti istatistiklerini test eder."""
        from src.claude_client import CodeGenerationDataset
        dataset = CodeGenerationDataset()
        
        # Bazı örnekler ekle
        for i in range(3):
            response = CodeGenerationResponse(
                instruction=f"Write function {i}",
                generated_code=f"def func_{i}(): pass",
                language="python",
                tokens_used=100 + i * 10,
                generation_time=1.0 + i * 0.5,
                success=True
            )
            dataset.add_response(response)
        
        stats = dataset.get_statistics()
        assert stats["total_examples"] == 3
        assert "python" in stats["languages"]
        assert stats["languages"]["python"] == 3
        assert stats["average_tokens"] == 110  # (100 + 110 + 120) / 3


if __name__ == "__main__":
    pytest.main([__file__])
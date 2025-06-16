"""
Bilgi Damıtma için Claude API İstemcisi

Bu modül, öğrenci modelini eğitmek için yüksek kaliteli kod örnekleri 
üretmek amacıyla Claude Opus 4 ile etkileşim kurmak için kapsamlı bir 
arayüz sağlar.
"""

import os
import time
import json
import asyncio
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

import anthropic
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message
import pandas as pd
from tqdm import tqdm
import backoff

logger = logging.getLogger(__name__)


@dataclass
class ClaudeConfig:
    """Claude API istemcisi için yapılandırma."""
    api_key: str
    model: str = "claude-opus-4-20250514"  # claude-4-opus mevcut olduğunda güncellenecek
    max_tokens: int = 4096
    temperature: float = 0.1
    top_p: float = 0.95
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_rpm: int = 50  # Dakika başına istek sayısı
    rate_limit_tpm: int = 40000  # Dakika başına token sayısı
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ClaudeConfig':
        """YAML dosyasından yapılandırmayı yükle."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict.get('claude', {}))


@dataclass
class CodeGenerationRequest:
    """Kod üretimi için istek."""
    instruction: str
    language: str
    context: Optional[str] = None
    difficulty: str = "medium"  # easy, medium, hard
    style: str = "clean"  # clean, documented, optimized
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


@dataclass
class CodeGenerationResponse:
    """Kod üretiminden gelen yanıt."""
    instruction: str
    generated_code: str
    language: str
    tokens_used: int
    generation_time: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RateLimiter:
    """API istekleri için hız sınırlayıcı."""
    
    def __init__(self, rpm: int = 50, tpm: int = 40000):
        self.rpm = rpm
        self.tpm = tpm
        self.request_times = []
        self.token_usage = []
        
    async def wait_if_needed(self, estimated_tokens: int = 1000):
        """Hız limitleri aşılacaksa bekle."""
        current_time = time.time()
        
        # Eski kayıtları temizle (1 dakikadan eski)
        cutoff_time = current_time - 60
        self.request_times = [t for t in self.request_times if t > cutoff_time]
        self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > cutoff_time]
        
        # İstek hız limitini kontrol et
        if len(self.request_times) >= self.rpm:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                logger.info(f"Hız limiti ulaşıldı, {wait_time:.2f} saniye bekleniyor")
                await asyncio.sleep(wait_time)
        
        # Token hız limitini kontrol et
        total_tokens = sum(tokens for _, tokens in self.token_usage)
        if total_tokens + estimated_tokens > self.tpm:
            wait_time = 60 - (current_time - self.token_usage[0][0])
            if wait_time > 0:
                logger.info(f"Token limiti ulaşıldı, {wait_time:.2f} saniye bekleniyor")
                await asyncio.sleep(wait_time)
        
        # Bu isteği kaydet
        self.request_times.append(current_time)
        self.token_usage.append((current_time, estimated_tokens))


class ClaudeAPIClient:
    """Claude API ile etkileşim için istemci."""
    
    def __init__(self, config: ClaudeConfig):
        self.config = config
        self.client = Anthropic(api_key=config.api_key)
        self.async_client = AsyncAnthropic(api_key=config.api_key)
        self.rate_limiter = RateLimiter(config.rate_limit_rpm, config.rate_limit_tpm)
        
        # İstatistikler
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.failed_requests = 0
        
        logger.info(f"Claude API istemcisi model ile başlatıldı: {config.model}")
    
    def _estimate_tokens(self, text: str) -> int:
        """Metin için token sayısını tahmin et."""
        # Yaklaşık tahmin: 1 token ≈ 4 karakter
        return len(text) // 4
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Token kullanımına dayalı maliyeti hesapla."""
        # Claude Opus 4 fiyatlandırması (2025 itibariyle)
        input_cost = input_tokens * 15 / 1_000_000  # 1M girdi tokeni için $15
        output_cost = output_tokens * 75 / 1_000_000  # 1M çıktı tokeni için $75
        return input_cost + output_cost
    
    @backoff.on_exception(
        backoff.expo,
        (anthropic.RateLimitError, anthropic.APITimeoutError),
        max_tries=3,
        max_time=300
    )
    async def _make_request(self, messages: List[Dict[str, str]], **kwargs) -> Message:
        """Yeniden deneme mantığı ile API isteği yap."""
        try:
            response = await self.async_client.messages.create(
                model=self.config.model,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                top_p=self.config.top_p,
                timeout=self.config.timeout
            )
            return response
        except Exception as e:
            logger.error(f"API isteği başarısız: {e}")
            raise
    
    def _create_code_generation_prompt(self, request: CodeGenerationRequest) -> str:
        """Kod üretimi için optimize edilmiş prompt oluştur."""
        base_prompt = f"""Sen uzman bir {request.language} programcısısın. Aşağıdaki talimata göre yüksek kaliteli, üretime hazır kod üret.

Talimat: {request.instruction}

Gereksinimler:
- Dil: {request.language}
- Zorluk: {request.difficulty}
- Stil: {request.style}
- Temiz, okunabilir ve iyi yapılandırılmış kod yaz
- Uygun yorumlar ve dokümantasyon ekle
- {request.language} için en iyi uygulamaları ve konvansiyonları takip et
- Kodun işlevsel ve çalıştırılabilir olmasını sağla
"""

        if request.context:
            base_prompt += f"\nBağlam: {request.context}"
        
        if request.style == "documented":
            base_prompt += "\n- Kapsamlı dokümantasyon ve yorumlar ekle"
        elif request.style == "optimized":
            base_prompt += "\n- Performans ve verimlilik odaklı ol"
        
        base_prompt += f"\n\nSadece {request.language} kodunu üret, ek açıklama ekleme:"
        
        return base_prompt
    
    async def generate_code(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """Claude API'sini kullanarak kod üret."""
        start_time = time.time()
        
        try:
            # Token'ları tahmin et ve hız sınırlaması uygula
            prompt = self._create_code_generation_prompt(request)
            estimated_tokens = self._estimate_tokens(prompt) + (request.max_tokens or self.config.max_tokens)
            
            await self.rate_limiter.wait_if_needed(estimated_tokens)
            
            # Mesajları hazırla
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # API isteği yap
            response = await self._make_request(
                messages=messages,
                max_tokens=request.max_tokens or self.config.max_tokens,
                temperature=request.temperature or self.config.temperature
            )
            
            # Üretilen kodu çıkar
            generated_code = response.content[0].text.strip()
            
            # Metrikleri hesapla
            generation_time = time.time() - start_time
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens
            cost = self._calculate_cost(input_tokens, output_tokens)
            
            # İstatistikleri güncelle
            self.total_requests += 1
            self.total_tokens += total_tokens
            self.total_cost += cost
            
            logger.info(f"{request.language} için kod üretildi {generation_time:.2f}s, "
                       f"tokenlar: {total_tokens}, maliyet: ${cost:.4f}")
            
            return CodeGenerationResponse(
                instruction=request.instruction,
                generated_code=generated_code,
                language=request.language,
                tokens_used=total_tokens,
                generation_time=generation_time,
                success=True,
                metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": cost,
                    "model": self.config.model
                }
            )
            
        except Exception as e:
            self.failed_requests += 1
            generation_time = time.time() - start_time
            
            logger.error(f"Kod üretimi başarısız: {e}")
            
            return CodeGenerationResponse(
                instruction=request.instruction,
                generated_code="",
                language=request.language,
                tokens_used=0,
                generation_time=generation_time,
                success=False,
                error=str(e)
            )
    
    async def generate_batch(
        self, 
        requests: List[CodeGenerationRequest],
        max_concurrent: int = 5,
        progress_callback: Optional[callable] = None
    ) -> List[CodeGenerationResponse]:
        """Birden fazla istek için eş zamanlı kod üret."""
        logger.info(f"{len(requests)} istek için toplu üretim başlatılıyor")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        responses = []
        
        async def generate_with_semaphore(request: CodeGenerationRequest) -> CodeGenerationResponse:
            async with semaphore:
                response = await self.generate_code(request)
                if progress_callback:
                    progress_callback(response)
                return response
        
        # İstekleri eş zamanlı çalıştır
        tasks = [generate_with_semaphore(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # İstisna durumları ele al
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"İstek {i} başarısız: {response}")
                valid_responses.append(CodeGenerationResponse(
                    instruction=requests[i].instruction,
                    generated_code="",
                    language=requests[i].language,
                    tokens_used=0,
                    generation_time=0,
                    success=False,
                    error=str(response)
                ))
            else:
                valid_responses.append(response)
        
        logger.info(f"Toplu üretim tamamlandı. Başarı oranı: "
                   f"{sum(1 for r in valid_responses if r.success)}/{len(valid_responses)}")
        
        return valid_responses
    
    def get_statistics(self) -> Dict[str, Any]:
        """Kullanım istatistiklerini al."""
        success_rate = (self.total_requests - self.failed_requests) / max(self.total_requests, 1)
        
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "average_tokens_per_request": self.total_tokens / max(self.total_requests, 1),
            "average_cost_per_request": self.total_cost / max(self.total_requests, 1)
        }
    
    def save_statistics(self, filepath: str):
        """İstatistikleri dosyaya kaydet."""
        stats = self.get_statistics()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"İstatistikler {filepath} dosyasına kaydedildi")


class CodeGenerationDataset:
    """Oluşturulan kod örneklerini saklamak için veri kümesi."""
    
    def __init__(self):
        self.examples = []
    
    def add_response(self, response: CodeGenerationResponse):
        """Veri kümesine bir yanıt ekle."""
        if response.success:
            self.examples.append({
                "instruction": response.instruction,
                "code": response.generated_code,
                "language": response.language,
                "tokens_used": response.tokens_used,
                "generation_time": response.generation_time,
                "metadata": response.metadata
            })
    
    def save_to_jsonl(self, filepath: str):
        """Veri kümesini JSONL formatında kaydet."""
        with open(filepath, 'w') as f:
            for example in self.examples:
                f.write(json.dumps(example) + '\n')
        logger.info(f"Veri kümesi {len(self.examples)} örnekle {filepath} dosyasına kaydedildi")
    
    def save_to_csv(self, filepath: str):
        """Veri kümesini CSV formatında kaydet."""
        df = pd.DataFrame(self.examples)
        df.to_csv(filepath, index=False)
        logger.info(f"Veri kümesi {len(self.examples)} örnekle {filepath} dosyasına kaydedildi")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Veri kümesi istatistiklerini al."""
        if not self.examples:
            return {}
        
        df = pd.DataFrame(self.examples)
        
        return {
            "total_examples": len(self.examples),
            "languages": df['language'].value_counts().to_dict(),
            "average_tokens": df['tokens_used'].mean(),
            "total_tokens": df['tokens_used'].sum(),
            "average_generation_time": df['generation_time'].mean(),
            "total_generation_time": df['generation_time'].sum()
        }


# Example usage and testing
async def main():
    """Claude API istemcisinin örnek kullanımı."""
    # Yapılandırmayı yükle
    config = ClaudeConfig(
        api_key=os.getenv("ANTHROPIC_API_KEY", "your-api-key-here"),
        model="claude-3-opus-20240229",
        max_tokens=2048,
        temperature=0.1
    )
    
    # İstemciyi başlat
    client = ClaudeAPIClient(config)
    
    # Örnek istekler oluştur
    requests = [
        CodeGenerationRequest(
            instruction="Bir sayının faktöriyelini hesaplayan bir Python fonksiyonu yaz",
            language="python",
            difficulty="easy",
            style="documented"
        ),
        CodeGenerationRequest(
            instruction="Regex kullanarak e-posta adreslerini doğrulayan bir JavaScript fonksiyonu oluştur",
            language="javascript",
            difficulty="medium",
            style="clean"
        ),
        CodeGenerationRequest(
            instruction="Java'da ikili arama algoritmasını uygula",
            language="java",
            difficulty="medium",
            style="optimized"
        )
    ]
    
    # Kod üret
    dataset = CodeGenerationDataset()
    
    print("Kod örnekleri üretiliyor...")
    responses = await client.generate_batch(requests, max_concurrent=2)
    
    for response in responses:
        if response.success:
            print(f"\n--- {response.language.upper()} ---")
            print(f"Talimat: {response.instruction}")
            print(f"Üretilen Kod:\n{response.generated_code}")
            print(f"Tokenlar: {response.tokens_used}, Süre: {response.generation_time:.2f}s")
            dataset.add_response(response)
        else:
            print(f"Başarısız: {response.error}")
    
    # İstatistikleri yazdır
    print("\n--- İSTEMCİ İSTATİSTİKLERİ ---")
    stats = client.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n--- VERİ KÜMESİ İSTATİSTİKLERİ ---")
    dataset_stats = dataset.get_statistics()
    for key, value in dataset_stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())


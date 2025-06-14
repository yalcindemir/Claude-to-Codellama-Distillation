"""
Bilgi Damıtımı için Kapsamlı Değerlendirme Sistemi

Bu modül, damıtılmış öğrenci modelini öğretmen model ve temel
modellerle karşılaştırmak için kapsamlı değerlendirme yetenekleri sağlar.
"""

import os
import json
import logging
import subprocess
import tempfile
import ast
import sys
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wandb

# Evaluation metrics
from evaluate import load
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


@dataclass
class EvaluationConfig:
    """Değerlendirme için yapılandırma."""
    
    # Değerlendirilecek modeller
    student_model_path: str = "./models/distilled_codellama"
    teacher_model_name: str = "claude-3-opus-20240229"  # API aracılığıyla
    baseline_models: List[str] = field(default_factory=lambda: [
        "codellama/CodeLlama-7b-hf",
        "codellama/CodeLlama-7b-Instruct-hf"
    ])
    
    # Değerlendirme veri setleri
    test_datasets: List[str] = field(default_factory=lambda: [
        "humaneval",
        "mbpp", 
        "apps"
    ])
    custom_test_path: Optional[str] = "./data/generated/test.jsonl"
    
    # Üretim parametreleri
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95
    do_sample: bool = True
    num_return_sequences: int = 1
    
    # Değerlendirme parametreleri
    timeout_seconds: int = 10
    max_workers: int = 4
    batch_size: int = 8
    
    # Hesaplanacak metrikler
    compute_pass_at_k: bool = True
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10])
    compute_bleu: bool = True
    compute_rouge: bool = True
    compute_code_quality: bool = True
    compute_execution_accuracy: bool = True
    
    # Çıktı
    output_dir: str = "./evaluation_results"
    save_predictions: bool = True
    generate_report: bool = True


class CodeExecutor:
    """Fonksiyonel doğruluk değerlendirmesi için güvenli kod çalıştırma."""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
    
    def execute_python_code(self, code: str, test_cases: List[Dict]) -> Dict[str, Any]:
        """Test durumlarıyla Python kodunu çalıştırır."""
        results = {
            "syntax_valid": False,
            "execution_success": False,
            "test_results": [],
            "error_message": None,
            "execution_time": 0.0
        }
        
        try:
            # Sözdizimini kontrol et
            ast.parse(code)
            results["syntax_valid"] = True
        except SyntaxError as e:
            results["error_message"] = f"Syntax Error: {str(e)}"
            return results
        
        # Test durumlarıyla çalıştır
        for i, test_case in enumerate(test_cases):
            test_result = self._run_single_test(code, test_case)
            results["test_results"].append(test_result)
        
        # Genel çalışma başarısı
        results["execution_success"] = all(
            test["passed"] for test in results["test_results"]
        )
        
        return results
    
    def _run_single_test(self, code: str, test_case: Dict) -> Dict[str, Any]:
        """Tek bir test durumunu çalıştırır."""
        test_result = {
            "passed": False,
            "expected": test_case.get("expected"),
            "actual": None,
            "error": None,
            "execution_time": 0.0
        }
        
        try:
            # Geçici dosya oluştur
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Kod ve testi yaz
                f.write(code + "\n\n")
                f.write(f"# Test durumu\n")
                f.write(f"result = {test_case['input']}\n")
                f.write(f"print(repr(result))\n")
                temp_file = f.name
            
            # Zaman aşımıyla çalıştır
            start_time = time.time()
            process = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            execution_time = time.time() - start_time
            
            if process.returncode == 0:
                # Çıktıyı ayrıştır
                output = process.stdout.strip()
                try:
                    actual_result = eval(output)
                    test_result["actual"] = actual_result
                    test_result["passed"] = actual_result == test_case["expected"]
                except:
                    test_result["error"] = f"Failed to parse output: {output}"
            else:
                test_result["error"] = process.stderr
            
            test_result["execution_time"] = execution_time
            
        except subprocess.TimeoutExpired:
            test_result["error"] = f"Execution timeout ({self.timeout}s)"
        except Exception as e:
            test_result["error"] = str(e)
        finally:
            # Temizle
            try:
                os.unlink(temp_file)
            except:
                pass
        
        return test_result


class BenchmarkEvaluator:
    """Standart kodlama kıyaslama testleri için değerlendirici."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.code_executor = CodeExecutor(config.timeout_seconds)
        
        # Değerlendirme metriklerini yükle
        self.bleu_scorer = None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def load_humaneval(self) -> Dataset:
        """HumanEval veri setini yükler."""
        try:
            dataset = load_dataset("openai_humaneval")["test"]
            return dataset
        except Exception as e:
            logger.error(f"Failed to load HumanEval: {e}")
            return None
    
    def load_mbpp(self) -> Dataset:
        """MBPP veri setini yükler."""
        try:
            dataset = load_dataset("mbpp", "sanitized")["test"]
            return dataset
        except Exception as e:
            logger.error(f"Failed to load MBPP: {e}")
            return None
    
    def load_apps(self) -> Dataset:
        """APPS veri setini yükler."""
        try:
            dataset = load_dataset("codeparrot/apps")["test"]
            # Daha hızlı değerlendirme için bir alt küme örnekle
            dataset = dataset.select(range(min(1000, len(dataset))))
            return dataset
        except Exception as e:
            logger.error(f"Failed to load APPS: {e}")
            return None
    
    def evaluate_humaneval(self, model, tokenizer, dataset: Dataset) -> Dict[str, float]:
        """HumanEval veri seti üzerinde değerlendirme yapar."""
        logger.info("HumanEval üzerinde değerlendirme yapılıyor...")
        
        results = []
        for example in tqdm(dataset, desc="HumanEval"):
            prompt = example["prompt"]
            canonical_solution = example["canonical_solution"]
            test = example["test"]
            
            # Kod üret
            generated_code = self._generate_code(model, tokenizer, prompt)
            
            # Test durumlarını hazırla
            test_cases = self._parse_humaneval_test(test, canonical_solution)
            
            # Çalıştır ve değerlendir
            execution_result = self.code_executor.execute_python_code(
                generated_code, test_cases
            )
            
            results.append({
                "task_id": example["task_id"],
                "prompt": prompt,
                "generated_code": generated_code,
                "execution_result": execution_result,
                "passed": execution_result["execution_success"]
            })
        
        # Metrikleri hesapla
        pass_at_1 = sum(r["passed"] for r in results) / len(results)
        
        return {
            "pass_at_1": pass_at_1,
            "total_problems": len(results),
            "solved_problems": sum(r["passed"] for r in results),
            "results": results
        }
    
    def evaluate_mbpp(self, model, tokenizer, dataset: Dataset) -> Dict[str, float]:
        """MBPP veri seti üzerinde değerlendirme yapar."""
        logger.info("MBPP üzerinde değerlendirme yapılıyor...")
        
        results = []
        for example in tqdm(dataset, desc="MBPP"):
            prompt = f"# {example['text']}\n# Bu problemi çözmek için bir Python fonksiyonu yazın.\n"
            
            # Kod üret
            generated_code = self._generate_code(model, tokenizer, prompt)
            
            # Test durumlarını hazırla
            test_cases = []
            for test_case in example.get("test_list", []):
                try:
                    # Test durumunu ayrıştır (basitleştirilmiş)
                    if "assert" in test_case:
                        # Fonksiyon çağrısı ve beklenen sonucu çıkar
                        # Bu basitleştirilmiş bir ayrıştırıcı
                        parts = test_case.replace("assert ", "").split(" == ")
                        if len(parts) == 2:
                            input_call = parts[0].strip()
                            expected = eval(parts[1].strip())
                            test_cases.append({
                                "input": input_call,
                                "expected": expected
                            })
                except:
                    continue
            
            # Çalıştır ve değerlendir
            execution_result = self.code_executor.execute_python_code(
                generated_code, test_cases
            )
            
            results.append({
                "task_id": example.get("task_id", len(results)),
                "prompt": prompt,
                "generated_code": generated_code,
                "execution_result": execution_result,
                "passed": execution_result["execution_success"]
            })
        
        # Metrikleri hesapla
        pass_at_1 = sum(r["passed"] for r in results) / len(results) if results else 0
        
        return {
            "pass_at_1": pass_at_1,
            "total_problems": len(results),
            "solved_problems": sum(r["passed"] for r in results),
            "results": results
        }
    
    def _generate_code(self, model, tokenizer, prompt: str) -> str:
        """Model kullanarak kod üretir."""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                num_return_sequences=self.config.num_return_sequences,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Üretilen metni çöz
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Sadece üretilen kısmı çıkar
        generated_code = generated_text[len(prompt):].strip()
        
        return generated_code
    
    def _parse_humaneval_test(self, test: str, canonical_solution: str) -> List[Dict]:
        """HumanEval test durumlarını ayrıştırır."""
        # Bu basitleştirilmiş bir ayrıştırıcı
        # Pratikte daha sağlam bir ayrıştırıcıya ihtiyacınız olur
        test_cases = []
        
        try:
            # Test durumlarını çıkarmak için testi çalıştır
            # Bu basitleştirilmiş bir yaklaşım
            exec_globals = {}
            exec(canonical_solution, exec_globals)
            
            # Fonksiyon adını çıkar
            func_name = None
            for name, obj in exec_globals.items():
                if callable(obj) and not name.startswith('_'):
                    func_name = name
                    break
            
            if func_name:
                # Bazı temel test durumları oluştur
                test_cases = [
                    {"input": f"{func_name}()", "expected": None}  # Yer tutucu
                ]
        except:
            pass
        
        return test_cases
    
    def compute_code_quality_metrics(self, code: str) -> Dict[str, float]:
        """Kod kalite metriklerini hesaplar."""
        metrics = {
            "length": len(code),
            "lines": len(code.split('\n')),
            "complexity": 1.0,  # Yer tutucu
            "readability": 1.0   # Yer tutucu
        }
        
        try:
            # Karmaşıklık analizi için AST'yi ayrıştır
            tree = ast.parse(code)
            
            # Farklı düğüm tiplerini say
            node_counts = {}
            for node in ast.walk(tree):
                node_type = type(node).__name__
                node_counts[node_type] = node_counts.get(node_type, 0) + 1
            
            # Basit karmaşıklık metriği
            complexity_nodes = ['If', 'For', 'While', 'Try', 'With']
            complexity = sum(node_counts.get(node, 0) for node in complexity_nodes)
            metrics["complexity"] = complexity
            
            # Basit okunabilirlik metriği (yorumlar, doküman string'leri)
            comment_lines = len([line for line in code.split('\n') if line.strip().startswith('#')])
            docstring_count = node_counts.get('Str', 0)  # Basitleştirilmiş
            metrics["readability"] = (comment_lines + docstring_count) / max(metrics["lines"], 1)
            
        except:
            pass
        
        return metrics


class ModelComparator:
    """Birden fazla modeli çeşitli metrikler üzerinde karşılaştırır."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.benchmark_evaluator = BenchmarkEvaluator(config)
        
        # Çıktı dizinini oluştur
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_model(self, model_path: str) -> Tuple[Any, Any]:
        """Model ve tokenizer'ı yükler."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return None, None
    
    def evaluate_model(self, model_name: str, model_path: str) -> Dict[str, Any]:
        """Tek bir modeli değerlendirir."""
        logger.info(f"Model değerlendiriliyor: {model_name}")
        
        # Modeli yükle
        model, tokenizer = self.load_model(model_path)
        if model is None:
            return {"error": f"Failed to load model {model_path}"}
        
        results = {"model_name": model_name, "model_path": model_path}
        
        # Kıyaslamalarda değerlendir
        for dataset_name in self.config.test_datasets:
            try:
                if dataset_name == "humaneval":
                    dataset = self.benchmark_evaluator.load_humaneval()
                    if dataset:
                        eval_result = self.benchmark_evaluator.evaluate_humaneval(
                            model, tokenizer, dataset
                        )
                        results[f"{dataset_name}_results"] = eval_result
                
                elif dataset_name == "mbpp":
                    dataset = self.benchmark_evaluator.load_mbpp()
                    if dataset:
                        eval_result = self.benchmark_evaluator.evaluate_mbpp(
                            model, tokenizer, dataset
                        )
                        results[f"{dataset_name}_results"] = eval_result
                
                # Gerektiğinde daha fazla veri seti ekle
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name} on {dataset_name}: {e}")
                results[f"{dataset_name}_error"] = str(e)
        
        return results
    
    def compare_models(self) -> Dict[str, Any]:
        """Tüm modelleri karşılaştırır."""
        logger.info("Model karşılaştırması başlatılıyor...")
        
        all_results = []
        
        # Öğrenci modelini değerlendir
        if os.path.exists(self.config.student_model_path):
            student_results = self.evaluate_model("Öğrenci (Damıtılmış)", self.config.student_model_path)
            all_results.append(student_results)
        
        # Temel modelleri değerlendir
        for baseline_model in self.config.baseline_models:
            baseline_results = self.evaluate_model(f"Temel ({baseline_model})", baseline_model)
            all_results.append(baseline_results)
        
        # Karşılaştırma özetini oluştur
        comparison_summary = self._create_comparison_summary(all_results)
        
        # Sonuçları kaydet
        self._save_results(all_results, comparison_summary)
        
        return {
            "individual_results": all_results,
            "comparison_summary": comparison_summary
        }
    
    def _create_comparison_summary(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Karşılaştırma özetini oluşturur."""
        summary = {
            "models": [],
            "metrics": {},
            "rankings": {}
        }
        
        for result in all_results:
            if "error" in result:
                continue
            
            model_summary = {
                "name": result["model_name"],
                "path": result["model_path"]
            }
            
            # Anahtar metrikleri çıkar
            for dataset_name in self.config.test_datasets:
                dataset_key = f"{dataset_name}_results"
                if dataset_key in result:
                    dataset_result = result[dataset_key]
                    metric_key = f"{dataset_name}_pass_at_1"
                    model_summary[metric_key] = dataset_result.get("pass_at_1", 0.0)
                    
                    # Global metriklere ekle
                    if metric_key not in summary["metrics"]:
                        summary["metrics"][metric_key] = []
                    summary["metrics"][metric_key].append({
                        "model": result["model_name"],
                        "value": dataset_result.get("pass_at_1", 0.0)
                    })
            
            summary["models"].append(model_summary)
        
        # Sıralamaları oluştur
        for metric_name, metric_values in summary["metrics"].items():
            sorted_values = sorted(metric_values, key=lambda x: x["value"], reverse=True)
            summary["rankings"][metric_name] = sorted_values
        
        return summary
    
    def _save_results(self, all_results: List[Dict], comparison_summary: Dict):
        """Değerlendirme sonuçlarını kaydeder."""
        output_dir = Path(self.config.output_dir)
        
        # Bireysel sonuçları kaydet
        for result in all_results:
            model_name = result["model_name"].replace(" ", "_").replace("(", "").replace(")", "")
            result_file = output_dir / f"{model_name}_results.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        
        # Karşılaştırma özetini kaydet
        summary_file = output_dir / "comparison_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(comparison_summary, f, indent=2, default=str)
        
        # Görselleştirme oluştur
        self._create_visualizations(comparison_summary)
        
        logger.info(f"Sonuçlar {output_dir} konumuna kaydedildi")
    
    def _create_visualizations(self, comparison_summary: Dict):
        """Görselleştirme grafikleri oluşturur."""
        output_dir = Path(self.config.output_dir)
        
        # Performans karşılaştırma grafiği
        fig, axes = plt.subplots(1, len(self.config.test_datasets), figsize=(15, 5))
        if len(self.config.test_datasets) == 1:
            axes = [axes]
        
        for i, dataset_name in enumerate(self.config.test_datasets):
            metric_key = f"{dataset_name}_pass_at_1"
            if metric_key in comparison_summary["rankings"]:
                rankings = comparison_summary["rankings"][metric_key]
                
                models = [r["model"] for r in rankings]
                values = [r["value"] for r in rankings]
                
                axes[i].bar(models, values)
                axes[i].set_title(f"{dataset_name.upper()} Pass@1")
                axes[i].set_ylabel("Pass@1 Score")
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Görselleştirmeler oluşturuldu")
    
    def generate_report(self, results: Dict) -> str:
        """Değerlendirme raporu üretir."""
        report_lines = []
        report_lines.append("# Bilgi Damıtımı Değerlendirme Raporu")
        report_lines.append("")
        report_lines.append(f"Oluşturulma tarihi: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Özet
        report_lines.append("## Özet")
        comparison_summary = results["comparison_summary"]
        
        for metric_name, rankings in comparison_summary["rankings"].items():
            report_lines.append(f"\n### {metric_name}")
            for i, ranking in enumerate(rankings):
                report_lines.append(f"{i+1}. {ranking['model']}: {ranking['value']:.3f}")
        
        # Detaylı sonuçlar
        report_lines.append("\n## Detaylı Sonuçlar")
        for result in results["individual_results"]:
            if "error" in result:
                continue
            
            report_lines.append(f"\n### {result['model_name']}")
            
            for dataset_name in self.config.test_datasets:
                dataset_key = f"{dataset_name}_results"
                if dataset_key in result:
                    dataset_result = result[dataset_key]
                    report_lines.append(f"\n#### {dataset_name.upper()}")
                    report_lines.append(f"- Geçme@1: {dataset_result.get('pass_at_1', 0):.3f}")
                    report_lines.append(f"- Toplam Problem: {dataset_result.get('total_problems', 0)}")
                    report_lines.append(f"- Çözülen Problem: {dataset_result.get('solved_problems', 0)}")
        
        report_content = "\n".join(report_lines)
        
        # Raporu kaydet
        report_file = Path(self.config.output_dir) / "evaluation_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Rapor {report_file} konumuna kaydedildi")
        return report_content


# Örnek kullanım
def main():
    """Örnek değerlendirme çalışması."""
    config = EvaluationConfig(
        student_model_path="./models/distilled_codellama",
        test_datasets=["humaneval"],  # Bir veri seti ile başla
        output_dir="./evaluation_results"
    )
    
    comparator = ModelComparator(config)
    results = comparator.compare_models()
    
    if config.generate_report:
        report = comparator.generate_report(results)
        print("Değerlendirme tamamlandı!")
        print(f"Sonuçlar {config.output_dir} konumuna kaydedildi")


if __name__ == "__main__":
    main()


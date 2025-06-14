# API Reference

Complete API documentation for Claude-to-CodeLlama Knowledge Distillation.

## Core Classes

### ClaudeAPIClient

Main client for interacting with Claude API.

```python
from src.claude_client import ClaudeAPIClient, ClaudeConfig

config = ClaudeConfig(api_key="your-key")
client = ClaudeAPIClient(config)
```

#### Methods

##### `generate_code(request: CodeGenerationRequest) -> CodeGenerationResponse`

Generate code using Claude API.

**Parameters:**
- `request`: CodeGenerationRequest object with instruction, language, etc.

**Returns:**
- CodeGenerationResponse with generated code and metadata

**Example:**
```python
request = CodeGenerationRequest(
    instruction="Write a Python function to sort a list",
    language="python",
    difficulty="medium"
)
response = await client.generate_code(request)
print(response.generated_code)
```

##### `generate_batch(requests: List[CodeGenerationRequest]) -> List[CodeGenerationResponse]`

Generate code for multiple requests concurrently.

**Parameters:**
- `requests`: List of CodeGenerationRequest objects
- `max_concurrent`: Maximum concurrent requests (default: 5)
- `progress_callback`: Optional callback function

**Returns:**
- List of CodeGenerationResponse objects

### DatasetGenerator

Generates training datasets using Claude API.

```python
from src.dataset_generator import DatasetGenerator, DatasetConfig

config = DatasetConfig(target_size=1000)
generator = DatasetGenerator(config, claude_config)
```

#### Methods

##### `generate_dataset() -> Dataset`

Generate complete training dataset.

**Parameters:**
- `max_concurrent`: Maximum concurrent API calls (default: 5)
- `save_intermediate`: Save intermediate results (default: True)

**Returns:**
- HuggingFace Dataset object

##### `split_dataset(dataset: Dataset) -> DatasetDict`

Split dataset into train/validation/test sets.

**Parameters:**
- `dataset`: HuggingFace Dataset object

**Returns:**
- DatasetDict with train/validation/test splits

### KnowledgeDistillationSystem

Main training system for knowledge distillation.

```python
from src.distillation_trainer import KnowledgeDistillationSystem, DistillationConfig

config = DistillationConfig(
    student_model_name="codellama/CodeLlama-7b-hf",
    dataset_path="./data/generated"
)
system = KnowledgeDistillationSystem(config)
```

#### Methods

##### `run_full_training() -> Dict[str, Any]`

Run complete training pipeline.

**Returns:**
- Dictionary with training results and evaluation metrics

##### `setup_model_and_tokenizer()`

Initialize student model and tokenizer with LoRA/QLoRA.

##### `load_dataset() -> Tuple[CodeDataset, CodeDataset]`

Load and prepare training and validation datasets.

**Returns:**
- Tuple of (train_dataset, eval_dataset)

### ModelComparator

Evaluation system for comparing models.

```python
from src.evaluation_system import ModelComparator, EvaluationConfig

config = EvaluationConfig(
    student_model_path="./models/distilled_codellama",
    test_datasets=["humaneval", "mbpp"]
)
comparator = ModelComparator(config)
```

#### Methods

##### `compare_models() -> Dict[str, Any]`

Compare student model with baselines.

**Returns:**
- Dictionary with comparison results and rankings

##### `evaluate_model(model_name: str, model_path: str) -> Dict[str, Any]`

Evaluate a single model.

**Parameters:**
- `model_name`: Display name for the model
- `model_path`: Path to model files

**Returns:**
- Dictionary with evaluation metrics

## Configuration Classes

### ClaudeConfig

Configuration for Claude API client.

```python
@dataclass
class ClaudeConfig:
    api_key: str
    model: str = "claude-3-opus-20240229"
    max_tokens: int = 4096
    temperature: float = 0.1
    top_p: float = 0.95
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_rpm: int = 50
    rate_limit_tpm: int = 40000
```

### DatasetConfig

Configuration for dataset generation.

```python
@dataclass
class DatasetConfig:
    target_size: int = 25000
    languages: List[str] = ["python", "javascript", "java", "cpp", "go", "rust"]
    language_distribution: Dict[str, int] = {"python": 40, "javascript": 25, ...}
    difficulty_distribution: Dict[str, int] = {"easy": 30, "medium": 50, "hard": 20}
    style_distribution: Dict[str, int] = {"clean": 50, "documented": 30, "optimized": 20}
    output_dir: str = "./data/generated"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
```

### DistillationConfig

Configuration for knowledge distillation training.

```python
@dataclass
class DistillationConfig:
    # Model configuration
    student_model_name: str = "codellama/CodeLlama-7b-hf"
    max_length: int = 2048
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = ["q_proj", "k_proj", ...]
    
    # Quantization configuration
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # Training configuration
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Distillation configuration
    distillation_weight: float = 0.7
    task_weight: float = 0.3
    temperature: float = 4.0
```

### EvaluationConfig

Configuration for model evaluation.

```python
@dataclass
class EvaluationConfig:
    student_model_path: str = "./models/distilled_codellama"
    teacher_model_name: str = "claude-3-opus-20240229"
    baseline_models: List[str] = ["codellama/CodeLlama-7b-hf"]
    test_datasets: List[str] = ["humaneval", "mbpp", "apps"]
    custom_test_path: Optional[str] = "./data/generated/test.jsonl"
    
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95
    do_sample: bool = True
    num_return_sequences: int = 1
    
    # Evaluation parameters
    timeout_seconds: int = 10
    max_workers: int = 4
    batch_size: int = 8
    
    # Output
    output_dir: str = "./evaluation_results"
    save_predictions: bool = True
    generate_report: bool = True
```

## Data Classes

### CodeGenerationRequest

Request object for code generation.

```python
@dataclass
class CodeGenerationRequest:
    instruction: str
    language: str
    context: Optional[str] = None
    difficulty: str = "medium"  # easy, medium, hard
    style: str = "clean"  # clean, documented, optimized
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
```

### CodeGenerationResponse

Response object from code generation.

```python
@dataclass
class CodeGenerationResponse:
    instruction: str
    generated_code: str
    language: str
    tokens_used: int
    generation_time: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
```

## Advanced Loss Functions

### AdvancedDistillationLoss

Advanced loss function with multiple distillation techniques.

```python
from src.advanced_loss import AdvancedDistillationLoss, LossConfig

config = LossConfig(
    temperature=4.0,
    distillation_weight=0.7,
    task_weight=0.3,
    use_attention_transfer=True,
    use_feature_matching=True
)

loss_fn = AdvancedDistillationLoss(config, vocab_size=32000)
```

#### Methods

##### `forward() -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]`

Compute advanced distillation loss.

**Parameters:**
- `student_logits`: Student model logits
- `teacher_logits`: Teacher model logits (optional)
- `labels`: Ground truth labels
- `input_ids`: Input token IDs
- `student_attentions`: Student attention weights (optional)
- `teacher_attentions`: Teacher attention weights (optional)
- `student_hidden_states`: Student hidden states (optional)
- `teacher_hidden_states`: Teacher hidden states (optional)

**Returns:**
- Tuple of (total_loss, loss_components_dict)

## Utility Functions

### Rate Limiting

```python
from src.claude_client import RateLimiter

limiter = RateLimiter(rpm=50, tpm=40000)
await limiter.wait_if_needed(estimated_tokens=1000)
```

### Cost Calculation

```python
# Estimate API costs
input_tokens = 1000
output_tokens = 500
cost = client._calculate_cost(input_tokens, output_tokens)
print(f"Estimated cost: ${cost:.4f}")
```

### Quality Control

```python
from src.dataset_generator import DatasetQualityController

controller = DatasetQualityController(config)
is_valid = controller.validate_code_quality(response)
report = controller.get_quality_report()
```

## Error Handling

### Common Exceptions

```python
try:
    response = await client.generate_code(request)
except anthropic.RateLimitError:
    # Handle rate limiting
    await asyncio.sleep(60)
except anthropic.APITimeoutError:
    # Handle timeout
    print("Request timed out")
except Exception as e:
    # Handle other errors
    print(f"Unexpected error: {e}")
```

### Retry Logic

```python
import backoff

@backoff.on_exception(
    backoff.expo,
    (anthropic.RateLimitError, anthropic.APITimeoutError),
    max_tries=3,
    max_time=300
)
async def make_request_with_retry():
    return await client.generate_code(request)
```

## Examples

### Complete Training Pipeline

```python
import asyncio
from src import *

async def main():
    # 1. Setup configurations
    claude_config = ClaudeConfig(api_key="your-key")
    dataset_config = DatasetConfig(target_size=1000)
    training_config = DistillationConfig()
    
    # 2. Generate dataset
    generator = DatasetGenerator(dataset_config, claude_config)
    dataset = await generator.generate_dataset()
    dataset_dict = generator.split_dataset(dataset)
    generator.save_dataset(dataset_dict)
    
    # 3. Train model
    system = KnowledgeDistillationSystem(training_config)
    results = system.run_full_training()
    
    # 4. Evaluate model
    eval_config = EvaluationConfig()
    comparator = ModelComparator(eval_config)
    eval_results = comparator.compare_models()
    
    print("Training complete!")

if __name__ == "__main__":
    asyncio.run(main())
```

### Custom Loss Function

```python
from src.advanced_loss import LossConfig, AdvancedDistillationLoss

# Configure advanced loss
loss_config = LossConfig(
    temperature=4.0,
    use_progressive_distillation=True,
    progressive_schedule="cosine",
    use_attention_transfer=True,
    attention_weight=0.1
)

# Create loss function
loss_fn = AdvancedDistillationLoss(loss_config, vocab_size=32000)

# Set training epoch for progressive distillation
loss_fn.set_epoch(current_epoch=1, total_epochs=3)

# Compute loss
total_loss, loss_dict = loss_fn(
    student_logits=student_outputs.logits,
    teacher_logits=teacher_outputs.logits,
    labels=batch["labels"],
    input_ids=batch["input_ids"]
)
```

This API reference provides comprehensive documentation for all major components of the Claude-to-CodeLlama Knowledge Distillation system.
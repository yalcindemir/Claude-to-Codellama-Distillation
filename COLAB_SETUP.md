# 🚀 Google Colab Kurulum Rehberi

Bu proje Google Colab'da çalıştırılmak üzere tasarlanmıştır. İşte adım adım kurulum:

## 🔧 Hızlı Çözüm

Eğer `ModuleNotFoundError` alıyorsanız, notebook'unuzun başına şu hücreyi ekleyin:

```python
# Python path'ini düzelt
import sys
import os

# Src klasörünü path'e ekle
src_path = './src'
if src_path not in sys.path:
    sys.path.append(src_path)
    print(f"✅ {src_path} Python path'ine eklendi")

# Proje kök dizinini de ekle
root_path = '.'
if root_path not in sys.path:
    sys.path.append(root_path)
    print(f"✅ {root_path} Python path'ine eklendi")

# Test import
try:
    from claude_client import ClaudeConfig
    print("✅ Import başarılı")
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    print("Lütfen requirements.txt'teki paketlerin kurulu olduğundan emin olun")
```

## 📦 Tam Kurulum (Colab için)

### 1. Repository'yi Klonlayın
```bash
!git clone https://github.com/yalcindemir/claude-to-codellama-distillation.git
%cd claude-to-codellama-distillation
```

### 2. Gereken Paketleri Kurun
```bash
!pip install -r requirements.txt
```

### 3. Python Path'ini Ayarlayın
```python
import sys
import os
sys.path.append('./src')
```

### 4. API Anahtarınızı Ayarlayın
```python
import os
from getpass import getpass

# Claude API anahtarınızı girin
api_key = getpass('Anthropic API Key: ')
os.environ['ANTHROPIC_API_KEY'] = api_key
```

## 🔍 Yaygın Sorunlar ve Çözümleri

### Problem 1: ModuleNotFoundError
```
ModuleNotFoundError: No module named 'dataset_generator'
```

**Çözüm:**
```python
import sys
sys.path.append('./src')
```

### Problem 2: YAML Hatası
```
ModuleNotFoundError: No module named 'yaml'
```

**Çözüm:**
```bash
!pip install pyyaml
```

### Problem 3: API Key Hatası
```
Error: ANTHROPIC_API_KEY not set
```

**Çözüm:**
```python
import os
os.environ['ANTHROPIC_API_KEY'] = 'your-api-key-here'
```

## 🎯 Hızlı Test

Bu kodu çalıştırarak kurulumun doğru olduğunu test edin:

```python
# Test kurulum
import sys
import os

# Path'leri ekle
sys.path.append('./src')

try:
    from claude_client import ClaudeConfig
    from dataset_generator import DatasetConfig
    print("✅ Tüm modüller başarıyla import edildi")
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    print("Lütfen yukarıdaki adımları takip edin")
```

## 📞 Destek

Hala sorun yaşıyorsanız:
1. Tüm hücreleri yeniden çalıştırın
2. Runtime'ı restart edin (Runtime > Restart runtime)
3. GitHub issues sayfasından yardım isteyin

---
**❤️ ile Yalçın DEMIR tarafından geliştirildi**
# Claude Talimatları

Bu dosya, Claude'a projeyi daha iyi anlaması için önemli bilgileri içerir.

## Proje Hakkında

Bu, Claude Opus 4'ün kod üretim yeteneklerini Code Llama 7B'ye aktaran kapsamlı bir bilgi damıtma sistemidir.

## Komutlar

### Test Çalıştırma
```bash
python -m pytest tests/ -v
```

### Kod Kalitesi Kontrolleri
```bash
# Python syntax kontrolü
python -m py_compile src/*.py

# Tip kontrolü (eğer mypy kuruluysa)
mypy src/

# Kod formatı kontrolü (eğer black kuruluysa)
black --check src/
```

### Geliştirme Ortamı Kurulumu
```bash
# Virtual environment oluştur
python -m venv venv
source venv/bin/activate  # Windows'ta: venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# Geliştirme bağımlılıklarını yükle
pip install -e ".[dev]"
```

### Proje Çalıştırma
```bash
# API anahtarını ayarla
export ANTHROPIC_API_KEY='your-api-key-here'

# Tam pipeline'ı çalıştır
./scripts/run_full_pipeline.sh

# Tek tek modülleri test et
python src/claude_client.py
python src/dataset_generator.py
python src/distillation_trainer.py
```

## Dosya Yapısı

- `src/claude_client.py` - Claude API entegrasyonu
- `src/dataset_generator.py` - Veri seti üretimi
- `src/distillation_trainer.py` - Model eğitimi
- `src/evaluation_system.py` - Model değerlendirmesi
- `src/advanced_loss.py` - Gelişmiş kayıp fonksiyonları
- `configs/config.yml` - Ana konfigürasyon
- `tests/` - Test dosyaları
- `scripts/` - Dağıtım script'leri

## Önemli Notlar

1. Bu proje büyük miktarda GPU belleği gerektirir (6-8GB minimum)
2. Claude API anahtarı gereklidir
3. Tam eğitim yaklaşık $100-200 maliyet oluşturur
4. Google Colab veya GCP kullanımı önerilir

## Bilinen Sorunlar

1. Python interpreter path'i düzeltilmeli (py_compile komutları için)
2. Test coverage artırılabilir
3. Daha fazla entegrasyon testi eklenebilir

## Son Güncellemeler

- Test dosyaları eklendi
- Requirements.txt güncellendi
- Setup.py oluşturuldu
- .gitignore eklendi
- Kod kalitesi iyileştirmeleri yapıldı
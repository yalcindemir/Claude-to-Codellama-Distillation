"""
Bilgi Damıtımı için Gelişmiş Kayıp Fonksiyonları ve Optimizasyon

Bu modül, Claude Opus 4'ten Code Llama'ya etkili bilgi transferi için
sofistike kayıp fonksiyonları ve optimizasyon stratejileri uygular.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LinearLR, SequentialLR, 
    CosineAnnealingWarmRestarts, OneCycleLR
)
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class LossConfig:
    """Kayıp fonksiyonları için yapılandırma."""
    
    # Temel damıtım
    temperature: float = 4.0
    distillation_weight: float = 0.7
    task_weight: float = 0.3
    
    # Gelişmiş teknikler
    use_attention_transfer: bool = True
    attention_weight: float = 0.1
    
    use_feature_matching: bool = True
    feature_weight: float = 0.1
    
    use_progressive_distillation: bool = True
    progressive_schedule: str = "linear"  # linear, cosine, exponential
    
    # Uyarlanabilir ağırlıklandırma
    use_adaptive_weighting: bool = True
    adaptation_rate: float = 0.01
    
    # Düzenlileştirme
    use_consistency_regularization: bool = True
    consistency_weight: float = 0.05
    
    # Token seviyesi ağırlıklandırma
    use_token_weighting: bool = True
    important_token_weight: float = 2.0


class AdvancedDistillationLoss(nn.Module):
    """Birden fazla damıtım tekniği ile gelişmiş kayıp fonksiyonu."""
    
    def __init__(self, config: LossConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # Temel kayıp fonksiyonları
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        self.mse_loss = nn.MSELoss()
        
        # Uyarlanabilir ağırlıklar (öğrenilebilir parametreler)
        if config.use_adaptive_weighting:
            self.adaptive_weights = nn.Parameter(torch.tensor([
                config.distillation_weight,
                config.task_weight,
                config.attention_weight,
                config.feature_weight
            ]))
        
        # İlerlemeli damıtım durumu
        self.current_epoch = 0
        self.total_epochs = 3  # Eğitim sırasında ayarlanacak
        
        logger.info("Gelişmiş damıtım kaybı başlatıldı")
    
    def set_epoch(self, epoch: int, total_epochs: int):
        """İlerlemeli damıtım için mevcut dönemi ayarlar."""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
    
    def _get_progressive_weights(self) -> Tuple[float, float]:
        """Eğitim ilerlemesine göre ilerlemeli ağırlıkları hesaplar."""
        if not self.config.use_progressive_distillation:
            return self.config.distillation_weight, self.config.task_weight
        
        progress = self.current_epoch / max(self.total_epochs, 1)
        
        if self.config.progressive_schedule == "linear":
            # Daha fazla damıtım ile başla, kıdemli olarak görev ağırlığını artır
            distill_weight = self.config.distillation_weight * (1 - progress * 0.3)
            task_weight = self.config.task_weight * (1 + progress * 0.5)
        
        elif self.config.progressive_schedule == "cosine":
            # Yumuşak geçiş için kosinüs tavlama
            distill_weight = self.config.distillation_weight * (
                0.5 * (1 + math.cos(math.pi * progress))
            )
            task_weight = self.config.task_weight * (
                0.5 * (1 + math.cos(math.pi * (1 - progress)))
            )
        
        elif self.config.progressive_schedule == "exponential":
            # Damıtım ağırlığı için üssel azalma
            distill_weight = self.config.distillation_weight * math.exp(-2 * progress)
            task_weight = self.config.task_weight * (1 + progress)
        
        else:
            distill_weight = self.config.distillation_weight
            task_weight = self.config.task_weight
        
        # Ağırlıkları normalize et
        total_weight = distill_weight + task_weight
        distill_weight /= total_weight
        task_weight /= total_weight
        
        return distill_weight, task_weight
    
    def _compute_token_weights(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Farklı token'lar için önem ağırlıklarını hesaplar."""
        if not self.config.use_token_weighting:
            return torch.ones_like(labels, dtype=torch.float)
        
        weights = torch.ones_like(labels, dtype=torch.float)
        
        # Önemli token'lar (anahtar kelimeler, operatörler, vb.)
        # Bu basitleştirilmiş bir versiyon - pratikte daha sofistike bir yöntem kullanırdınız
        important_tokens = {
            # Python anahtar kelimeleri
            1, 2, 3, 4, 5,  # def, class, if, else, for, etc.
            # Yaygın operatörler
            10, 11, 12, 13, 14,  # =, +, -, *, /, etc.
        }
        
        for token_id in important_tokens:
            mask = (input_ids == token_id)
            weights[mask] = self.config.important_token_weight
        
        return weights
    
    def _compute_attention_transfer_loss(
        self, 
        student_attentions: List[torch.Tensor],
        teacher_attentions: List[torch.Tensor]
    ) -> torch.Tensor:
        """Dikkat transferi kaybını hesaplar."""
        if not student_attentions or not teacher_attentions:
            return torch.tensor(0.0, device=student_attentions[0].device if student_attentions else "cpu")
        
        total_loss = 0.0
        num_layers = min(len(student_attentions), len(teacher_attentions))
        
        for i in range(num_layers):
            student_att = student_attentions[i]  # [batch, heads, seq, seq]
            teacher_att = teacher_attentions[i]
            
            # Kafalar üzerinden ortalama al
            student_att = student_att.mean(dim=1)  # [batch, seq, seq]
            teacher_att = teacher_att.mean(dim=1)
            
            # Gerekirse yeniden boyutlandır
            if student_att.shape != teacher_att.shape:
                teacher_att = F.interpolate(
                    teacher_att.unsqueeze(1), 
                    size=student_att.shape[-2:], 
                    mode='bilinear'
                ).squeeze(1)
            
            # Dikkat haritaları arasındaki MSE kaybı
            layer_loss = self.mse_loss(student_att, teacher_att)
            total_loss += layer_loss
        
        return total_loss / num_layers if num_layers > 0 else torch.tensor(0.0)
    
    def _compute_feature_matching_loss(
        self,
        student_hidden_states: List[torch.Tensor],
        teacher_hidden_states: List[torch.Tensor]
    ) -> torch.Tensor:
        """Gizli durumlar arasında özellik eşleştirme kaybını hesaplar."""
        if not student_hidden_states or not teacher_hidden_states:
            return torch.tensor(0.0)
        
        total_loss = 0.0
        num_layers = min(len(student_hidden_states), len(teacher_hidden_states))
        
        for i in range(num_layers):
            student_hidden = student_hidden_states[i]  # [batch, seq, hidden]
            teacher_hidden = teacher_hidden_states[i]
            
            # Gerekirse aynı boyuta projekte et
            if student_hidden.shape[-1] != teacher_hidden.shape[-1]:
                # Basit doğrusal projeksiyon (pratikte öğrenilmiş bir projeksiyon kullanırdınız)
                teacher_hidden = teacher_hidden[..., :student_hidden.shape[-1]]
            
            # Gizli durumlar arasındaki MSE kaybı
            layer_loss = self.mse_loss(student_hidden, teacher_hidden)
            total_loss += layer_loss
        
        return total_loss / num_layers if num_layers > 0 else torch.tensor(0.0)
    
    def _compute_consistency_regularization(
        self,
        student_logits: torch.Tensor,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Tutarlılık düzenlileştirme kaybını hesaplar."""
        if not self.config.use_consistency_regularization:
            return torch.tensor(0.0, device=student_logits.device)
        
        # Girdiye küçük gürültü ekle ve tutarlılığı hesapla
        noise = torch.randn_like(input_ids, dtype=torch.float) * 0.1
        noisy_input = input_ids + noise.long()
        
        # Bu basitleştirilmiş bir versiyon - pratikte modelden tekrar geçirmeniz gerekir
        # Şimdilik logit pürüzsüzlüğüne dayalı basit bir düzenlileştirme kullanacağız
        logits_diff = student_logits[:, 1:] - student_logits[:, :-1]
        consistency_loss = torch.mean(torch.norm(logits_diff, dim=-1))
        
        return consistency_loss
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: Optional[torch.Tensor],
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        student_attentions: Optional[List[torch.Tensor]] = None,
        teacher_attentions: Optional[List[torch.Tensor]] = None,
        student_hidden_states: Optional[List[torch.Tensor]] = None,
        teacher_hidden_states: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Gelişmiş damıtım kaybını hesaplar.
        
        Args:
            student_logits: Öğrenci model logitleri [batch, seq, vocab]
            teacher_logits: Öğretmen model logitleri [batch, seq, vocab] (isteğe bağlı)
            labels: Gerçek etiketler [batch, seq]
            input_ids: Girdi token ID'leri [batch, seq]
            student_attentions: Öğrenci dikkat ağırlıkları (isteğe bağlı)
            teacher_attentions: Öğretmen dikkat ağırlıkları (isteğe bağlı)
            student_hidden_states: Öğrenci gizli durumları (isteğe bağlı)
            teacher_hidden_states: Öğretmen gizli durumları (isteğe bağlı)
            
        Returns:
            Toplam kayıp ve kayıp bileşenleri sözlüğü
        """
        device = student_logits.device
        loss_dict = {}
        
        # İlerlemeli ağırlıkları al
        distill_weight, task_weight = self._get_progressive_weights()
        
        # Etkinleştirilmişse uyarlanabilir ağırlıkları kullan
        if self.config.use_adaptive_weighting and hasattr(self, 'adaptive_weights'):
            weights = F.softmax(self.adaptive_weights, dim=0)
            distill_weight = weights[0].item()
            task_weight = weights[1].item()
            attention_weight = weights[2].item()
            feature_weight = weights[3].item()
        else:
            attention_weight = self.config.attention_weight
            feature_weight = self.config.feature_weight
        
        # 1. Görev Kaybı (Çapraz Entropi)
        token_weights = self._compute_token_weights(input_ids, labels)
        ce_losses = self.ce_loss(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )
        
        # Token ağırlıklarını uygula
        weighted_ce_losses = ce_losses * token_weights.view(-1)
        task_loss = weighted_ce_losses[labels.view(-1) != -100].mean()
        loss_dict["task_loss"] = task_loss
        
        total_loss = task_weight * task_loss
        
        # 2. Damıtım Kaybı (KL ıraksaması)
        if teacher_logits is not None:
            # Sıcaklık ölçeklendirme
            student_soft = F.log_softmax(student_logits / self.config.temperature, dim=-1)
            teacher_soft = F.softmax(teacher_logits / self.config.temperature, dim=-1)
            
            # KL ıraksaması
            kl_loss = self.kl_div(student_soft, teacher_soft) * (self.config.temperature ** 2)
            loss_dict["distillation_loss"] = kl_loss
            
            total_loss += distill_weight * kl_loss
        else:
            loss_dict["distillation_loss"] = torch.tensor(0.0, device=device)
        
        # 3. Dikkat Transferi Kaybı
        if (self.config.use_attention_transfer and 
            student_attentions and teacher_attentions):
            
            attention_loss = self._compute_attention_transfer_loss(
                student_attentions, teacher_attentions
            )
            loss_dict["attention_loss"] = attention_loss
            total_loss += attention_weight * attention_loss
        else:
            loss_dict["attention_loss"] = torch.tensor(0.0, device=device)
        
        # 4. Özellik Eşleştirme Kaybı
        if (self.config.use_feature_matching and 
            student_hidden_states and teacher_hidden_states):
            
            feature_loss = self._compute_feature_matching_loss(
                student_hidden_states, teacher_hidden_states
            )
            loss_dict["feature_loss"] = feature_loss
            total_loss += feature_weight * feature_loss
        else:
            loss_dict["feature_loss"] = torch.tensor(0.0, device=device)
        
        # 5. Tutarlılık Düzenlileştirme
        consistency_loss = self._compute_consistency_regularization(student_logits, input_ids)
        loss_dict["consistency_loss"] = consistency_loss
        total_loss += self.config.consistency_weight * consistency_loss
        
        # Toplam kaybı sakla
        loss_dict["total_loss"] = total_loss
        
        # İzleme için mevcut ağırlıkları sakla
        loss_dict["current_distill_weight"] = torch.tensor(distill_weight, device=device)
        loss_dict["current_task_weight"] = torch.tensor(task_weight, device=device)
        
        return total_loss, loss_dict


class AdaptiveOptimizer:
    """Gelişmiş programlama ile uyarlanabilir optimize edici."""
    
    def __init__(
        self,
        model_parameters,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        optimizer_type: str = "adamw",
        scheduler_type: str = "cosine",
        warmup_ratio: float = 0.1,
        total_steps: int = 1000,
        **kwargs
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.warmup_ratio = warmup_ratio
        self.total_steps = total_steps
        
        # Optimize ediciyi oluştur
        if optimizer_type.lower() == "adamw":
            self.optimizer = AdamW(
                model_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type.lower() == "adam":
            self.optimizer = Adam(
                model_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # Programlayıcıyı oluştur
        self.scheduler = self._create_scheduler()
        
        logger.info(f"Uyarlanabilir optimize edici başlatıldı: {optimizer_type} ile {scheduler_type} programlayıcısı")
    
    def _create_scheduler(self):
        """Öğrenme oranı programlayıcısını oluşturur."""
        warmup_steps = int(self.total_steps * self.warmup_ratio)
        
        if self.scheduler_type.lower() == "cosine":
            # Isınma ile kosinüs tavlama
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_steps - warmup_steps,
                eta_min=self.learning_rate * 0.01
            )
            
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
        
        elif self.scheduler_type.lower() == "onecycle":
            # Tek döngü öğrenme oranı
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate,
                total_steps=self.total_steps,
                pct_start=self.warmup_ratio,
                anneal_strategy='cos'
            )
        
        elif self.scheduler_type.lower() == "cosine_restarts":
            # Sıcak yeniden başlatmalar ile kosinüs tavlama
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.total_steps // 4,
                T_mult=2,
                eta_min=self.learning_rate * 0.01
            )
        
        elif self.scheduler_type.lower() == "linear":
            # Doğrusal ısınma + doğrusal azalma
            scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=0.01,
                total_iters=self.total_steps
            )
        
        else:
            # Programlayıcı yok
            scheduler = None
        
        return scheduler
    
    def step(self):
        """Optimize edici ve programlayıcıyı adımla."""
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
    
    def zero_grad(self):
        """Gradyanları sıfırla."""
        self.optimizer.zero_grad()
    
    def get_lr(self) -> float:
        """Mevcut öğrenme oranını al."""
        return self.optimizer.param_groups[0]['lr']
    
    def state_dict(self):
        """Durum sözlüğünü al."""
        state = {
            'optimizer': self.optimizer.state_dict(),
        }
        if self.scheduler:
            state['scheduler'] = self.scheduler.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        """Durum sözlüğünü yükle."""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler and 'scheduler' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler'])


class GradientClipping:
    """Gelişmiş gradyan kırpma teknikleri."""
    
    def __init__(
        self,
        max_norm: float = 1.0,
        clip_type: str = "norm",  # norm, value, adaptive
        adaptive_factor: float = 0.01
    ):
        self.max_norm = max_norm
        self.clip_type = clip_type
        self.adaptive_factor = adaptive_factor
        self.grad_history = []
    
    def clip_gradients(self, model) -> float:
        """Gradyanları kırpar ve gradyan normunu döndürür."""
        if self.clip_type == "norm":
            # Standart gradyan norm kırpma
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                self.max_norm
            )
        
        elif self.clip_type == "value":
            # Gradyan değeri kırpma
            torch.nn.utils.clip_grad_value_(
                model.parameters(), 
                self.max_norm
            )
            grad_norm = self._compute_grad_norm(model)
        
        elif self.clip_type == "adaptive":
            # Uyarlanabilir gradyan kırpma
            grad_norm = self._compute_grad_norm(model)
            self.grad_history.append(grad_norm.item())
            
            # Sadece son geçmişi tut
            if len(self.grad_history) > 100:
                self.grad_history = self.grad_history[-100:]
            
            # Uyarlanabilir eşiği hesapla
            if len(self.grad_history) > 10:
                mean_grad = np.mean(self.grad_history)
                std_grad = np.std(self.grad_history)
                adaptive_threshold = mean_grad + 2 * std_grad
                
                if grad_norm > adaptive_threshold:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        adaptive_threshold
                    )
        
        else:
            grad_norm = self._compute_grad_norm(model)
        
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    
    def _compute_grad_norm(self, model) -> torch.Tensor:
        """Gradyan normunu hesaplar."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return torch.tensor(total_norm)


# Örnek kullanım
def create_advanced_loss_and_optimizer(
    model,
    vocab_size: int,
    total_steps: int,
    loss_config: Optional[LossConfig] = None,
    learning_rate: float = 2e-4
):
    """Gelişmiş kayıp fonksiyonu ve optimize edici oluşturur."""
    
    if loss_config is None:
        loss_config = LossConfig()
    
    # Kayıp fonksiyonunu oluştur
    loss_fn = AdvancedDistillationLoss(loss_config, vocab_size)
    
    # Optimize ediciyi oluştur
    optimizer = AdaptiveOptimizer(
        model.parameters(),
        learning_rate=learning_rate,
        total_steps=total_steps,
        optimizer_type="adamw",
        scheduler_type="cosine",
        warmup_ratio=0.1
    )
    
    # Gradyan kırpıcıyı oluştur
    grad_clipper = GradientClipping(
        max_norm=1.0,
        clip_type="adaptive"
    )
    
    return loss_fn, optimizer, grad_clipper


if __name__ == "__main__":
    # Örnek kullanım
    config = LossConfig()
    loss_fn = AdvancedDistillationLoss(config, vocab_size=32000)
    
    # Test için kukla tensor'lar
    batch_size, seq_len, vocab_size = 2, 10, 32000
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Kaybı hesapla
    total_loss, loss_dict = loss_fn(
        student_logits, teacher_logits, labels, input_ids
    )
    
    print(f"Total loss: {total_loss.item():.4f}")
    for key, value in loss_dict.items():
        print(f"{key}: {value.item():.4f}")


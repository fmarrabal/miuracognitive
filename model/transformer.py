"""
MiuraCognitive - Transformer Base (Sprint 1)
=============================================
Transformer decoder-only con RMSNorm pre-norm, RoPE, SwiGLU y
hooks preparados para el Homeostatic Background Processor (HBP)
que se integrara en el Sprint 2.

Diseño: ~30M parámetros, 6 capas, d_model=384, 6 heads, context=256.
Autor: Francisco M. Arrabal (Curro) + asistencia Claude
"""

from __future__ import annotations  # Permite anotar tipos aún no definidos
from dataclasses import dataclass   # Para configuración declarativa
import math                          # Para sqrt, log, pi
import torch                         # Framework principal
import torch.nn as nn                # Módulos de red neuronal
import torch.nn.functional as F      # Funciones sin parámetros (softmax, etc.)


# =============================================================================
# CONFIGURACIÓN
# =============================================================================
@dataclass
class MiuraConfig:
    """Configuración del Transformer base. Usamos dataclass para que sea
    serializable, inmutable y autodocumentada."""
    vocab_size: int = 50257          # Tamaño del vocabulario GPT-2 BPE (tiktoken)
    d_model: int = 384               # Dimensión de embeddings y estado oculto
    n_layers: int = 6                # Número de bloques Transformer apilados
    n_heads: int = 6                 # Cabezas de atención (d_head = 384/6 = 64)
    d_ff: int = 1536                 # Dimensión interna del FFN (4x d_model)
    max_seq_len: int = 256           # Contexto máximo (ajustable)
    dropout: float = 0.0             # Sin dropout; modelos pequeños modernos no lo usan
    rope_theta: float = 10000.0      # Base de RoPE (estándar GPT-NeoX/Llama)
    tie_embeddings: bool = True      # Compartir matriz embedding con proyección final
    mod_strength: float = 0.3        # Amplitud de la modulación del HBP (perturbación ±mod_strength
                                     # alrededor de la identidad; NO un gate que aniquila la señal)

    @property
    def d_head(self) -> int:
        """Dimensión por cabeza. Propiedad calculada para evitar desincronizaciones."""
        assert self.d_model % self.n_heads == 0, "d_model debe ser divisible por n_heads"
        return self.d_model // self.n_heads


# =============================================================================
# RMSNorm: normalización más estable y rápida que LayerNorm
# =============================================================================
class RMSNorm(nn.Module):
    """RMSNorm (Zhang & Sennrich 2019). Normaliza por la raíz cuadrática media
    sin restar la media. Más estable numéricamente y ~15% más rápido."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()                         # Inicializa nn.Module
        self.eps = eps                             # Evita división por cero
        self.weight = nn.Parameter(torch.ones(d))  # Ganancia aprendible (inicializada a 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., d). Calculamos RMS sobre la última dimensión.
        # rsqrt(mean(x²) + eps) es la normalización; luego multiplicamos por weight.
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        # Cast a float32 para la normalización por estabilidad, y volver al dtype original
        return (x.float() * rms).to(x.dtype) * self.weight


# =============================================================================
# RoPE: Rotary Position Embedding
# =============================================================================
def build_rope_cache(seq_len: int, d_head: int, theta: float, device, dtype):
    """Pre-calcula los cosenos y senos de RoPE.

    RoPE (Su et al. 2021) codifica posición rotando pares de componentes del
    vector query/key en un ángulo proporcional a la posición. Ventaja: el
    producto escalar q·k solo depende de la posición relativa, no absoluta."""
    # Frecuencias inversas: una por cada par de dimensiones (d_head/2 pares)
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    # Posiciones 0, 1, ..., seq_len-1
    positions = torch.arange(seq_len, device=device).float()
    # freqs[i, j] = position_i * inv_freq_j. Shape: (seq_len, d_head/2)
    freqs = torch.outer(positions, inv_freq)
    # Pre-calculamos cos y sin. Shape: (seq_len, d_head/2). Los expandimos al llamar.
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Aplica la rotación RoPE a un tensor de queries o keys.

    x shape: (batch, n_heads, seq_len, d_head)
    cos/sin shape: (seq_len, d_head/2)
    """
    # Separamos dimensiones pares e impares del último eje
    x1 = x[..., 0::2]  # Componentes pares: x0, x2, x4, ...
    x2 = x[..., 1::2]  # Componentes impares: x1, x3, x5, ...
    # Ampliamos cos/sin a (1, 1, seq_len, d_head/2) para broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    # Rotación 2D en cada par: (x1', x2') = (x1·cos - x2·sin, x1·sin + x2·cos)
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    # Recomponemos intercalando: [rx1_0, rx2_0, rx1_1, rx2_1, ...]
    out = torch.empty_like(x)
    out[..., 0::2] = rx1
    out[..., 1::2] = rx2
    return out


# =============================================================================
# ATENCIÓN CAUSAL MULTI-HEAD con Flash Attention 2 vía SDPA
# =============================================================================
class CausalSelfAttention(nn.Module):
    """Self-attention causal. En Blackwell, F.scaled_dot_product_attention
    usa Flash Attention 2 automáticamente con BF16 y máscara causal."""

    def __init__(self, cfg: MiuraConfig):
        super().__init__()
        self.cfg = cfg
        # Proyección combinada Q, K, V en una sola matriz (más eficiente)
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        # Proyección de salida
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        # Interocepción: entropía de atención (solo se computa si compute_entropy)
        self.compute_entropy = False
        self._last_attn_entropy: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor):
        B, T, C = x.shape  # Batch, Tiempo (seq len), Canales (d_model)
        # Proyectamos a Q, K, V y separamos en 3 tensores
        qkv = self.qkv_proj(x)                      # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)              # Cada uno: (B, T, C)
        # Reshape para multi-head: (B, T, C) → (B, n_heads, T, d_head)
        q = q.view(B, T, self.cfg.n_heads, self.cfg.d_head).transpose(1, 2)
        k = k.view(B, T, self.cfg.n_heads, self.cfg.d_head).transpose(1, 2)
        v = v.view(B, T, self.cfg.n_heads, self.cfg.d_head).transpose(1, 2)
        # Aplicamos RoPE solo a Q y K (no a V; V lleva la información de contenido)
        q = apply_rope(q, rope_cos[:T], rope_sin[:T])
        k = apply_rope(k, rope_cos[:T], rope_sin[:T])
        # SDPA con máscara causal. En Blackwell usa Flash Attention 2.
        # is_causal=True aplica la máscara triangular inferior automáticamente.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        # Rearmamos: (B, n_heads, T, d_head) → (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # --- INTEROCEPCIÓN: entropía de atención (señal s_i para el HBP) ---
        # Solo si el bloque lo pide (compute_entropy): es O(T^2) en FP32 y solo
        # la consumen las variantes con HBP. Se calcula bajo no_grad, sin romper
        # la ruta Flash. Entropía media normalizada a ~[0,1].
        if self.compute_entropy:
            with torch.no_grad():
                scale = 1.0 / math.sqrt(self.cfg.d_head)
                scores = (q.float() @ k.float().transpose(-2, -1)) * scale   # (B,H,T,T)
                mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
                scores = scores.masked_fill(~mask, float("-inf"))
                p = torch.softmax(scores, dim=-1)                            # (B,H,T,T)
                ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1)            # (B,H,T)
                ent = ent / max(math.log(T), 1e-6)                          # guard T=1 (evita NaN)
                self._last_attn_entropy = ent.mean(dim=(1, 2))               # (B,)

        # Proyección final
        return self.out_proj(y)


# =============================================================================
# FFN con SwiGLU
# =============================================================================
class SwiGLU(nn.Module):
    """FFN con activación SwiGLU (Shazeer 2020). Tres proyecciones lineales.
    Usado por Llama, Mistral, Qwen. Mejor que GELU en modelos pequeños."""

    def __init__(self, cfg: MiuraConfig):
        super().__init__()
        # w1: proyecta a la dim. interna (puerta), w2: valores, w3: vuelve a d_model
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)  # Gate
        self.w2 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)  # Up
        self.w3 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)  # Down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(w1(x)) * w2(x), luego proyección de vuelta.
        # SiLU(x) = x * sigmoid(x). El producto elemento a elemento actúa como gate.
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# =============================================================================
# BLOQUE TRANSFORMER con HOOKS para el HBP
# =============================================================================
class TransformerBlock(nn.Module):
    """Bloque Transformer con pre-norm. Incluye desde el diseño los hooks
    que el HBP usará en Sprint 2, aunque ahora estén inactivos."""

    def __init__(self, cfg: MiuraConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx                       # Para identificar nodo en HBP
        self.mod_strength = cfg.mod_strength             # Amplitud de la modulación del HBP
        self.norm1 = RMSNorm(cfg.d_model)                # Pre-norm antes de atención
        self.attn = CausalSelfAttention(cfg)             # Self-attention causal
        self.norm2 = RMSNorm(cfg.d_model)                # Pre-norm antes de FFN
        self.ffn = SwiGLU(cfg)                           # Feed-forward SwiGLU
        # HOOK DE INTEROCEPCIÓN: se llenará en forward. El HBP lo leerá.
        self._interoception: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, rope_cos, rope_sin,
                modulation: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, T, d_model) tensor de entrada
        modulation: (B, d_model) señal opcional del HBP. None en Sprint 1.
        """
        # --- Rama de atención con conexión residual ---
        h = self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + h  # Residual

        # --- Rama FFN con conexión residual ---
        h = self.ffn(self.norm2(x))
        x = x + h  # Residual

        # --- HOOK: MODULACIÓN DESDE EL HBP ---
        # Modulación del HBP como perturbación SUAVE alrededor de la identidad:
        #   gate = 1 + mod_strength·tanh(modulation) ∈ (1-s, 1+s).
        # Crucial para el reasoner recurrente: un gate sigmoid (centrado en 0.5)
        # multiplicaría x por ~0.5 cada iteración -> decaimiento geométrico que
        # rompe la computación iterativa. Esta forma perturba sin aniquilar.
        if modulation is not None:
            gate = (1.0 + self.mod_strength * torch.tanh(modulation)).unsqueeze(1)
            x = x * gate

        # --- HOOK: INTEROCEPCIÓN PARA EL HBP ---
        # Señales por-ejemplo (B,) que el HBP lee para construir el vector s_i
        # DE ESTE NODO del grafo. Solo se recogen si el bloque computa entropía
        # (variantes con HBP): en el resto el hook queda inactivo (ahorro O(T^2)).
        if self.attn.compute_entropy:
            with torch.no_grad():
                self._interoception = {
                    "output_norm": x.norm(dim=-1).mean(dim=1),       # (B,)
                    "activation_abs": x.abs().mean(dim=(1, 2)),      # (B,)
                    "attn_entropy": self.attn._last_attn_entropy,    # (B,)
                }

        return x


# =============================================================================
# MODELO COMPLETO: MiuraCognitive (base, sin HBP aún)
# =============================================================================
class MiuraCognitive(nn.Module):
    """Modelo completo. En Sprint 1 es un Transformer decoder-only estándar.
    En Sprint 2 añadiremos el HBP que consumirá los hooks de interocepción
    y producirá señales de modulación para cada bloque."""

    def __init__(self, cfg: MiuraConfig):
        super().__init__()
        self.cfg = cfg
        # Tabla de embeddings de token
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Los N bloques Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, layer_idx=i) for i in range(cfg.n_layers)
        ])
        # Norm final antes de la proyección a vocabulario
        self.norm_f = RMSNorm(cfg.d_model)
        # Cabeza de lenguaje (proyección a logits de vocabulario)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying: compartir matriz con los embeddings ahorra parámetros
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # Pre-calculamos cos y sin de RoPE. Los registramos como buffers
        # para que se muevan automáticamente con .to(device) y no sean parámetros.
        cos, sin = build_rope_cache(
            cfg.max_seq_len, cfg.d_head, cfg.rope_theta,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Inicialización de pesos (GPT-2 style: std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Inicialización estándar: normal(0, 0.02) para Linear y Embedding."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx: (B, T) tokens de entrada
        targets: (B, T) opcional, para calcular pérdida. None durante generación.
        """
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"Secuencia ({T}) excede max_seq_len"
        # Embeddings de token
        x = self.tok_emb(idx)  # (B, T, d_model)
        # En Sprint 2, aquí llamaremos al HBP para obtener las modulaciones
        # por bloque. Ahora pasamos None a cada bloque.
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin, modulation=None)
        # Norm final y proyección a vocabulario
        x = self.norm_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Si hay targets, calculamos cross-entropy. ignore_index=-1 permite
        # máscaras (por ejemplo, tokens de padding).
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def num_parameters(self, non_embedding: bool = True) -> int:
        """Cuenta parámetros. Si non_embedding=True, descuenta la tabla de
        embeddings (que en modelos pequeños domina el total)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n


# =============================================================================
# SMOKE TEST: ejecutable como script para verificación rápida
# =============================================================================
if __name__ == "__main__":
    # Pequeña prueba: crea modelo, hace un forward+backward, verifica gradientes
    cfg = MiuraConfig()
    model = MiuraCognitive(cfg)
    print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Parámetros non-embedding: {model.num_parameters():,}")

    # Batch sintético
    B, T = 2, 64
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    # Forward + pérdida + backward
    logits, loss = model(idx, targets)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    loss.backward()

    # Verifica que los gradientes fluyen
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"Gradientes OK: {len(grad_norms)} tensores con grad, "
          f"norma media = {sum(grad_norms)/len(grad_norms):.4f}")

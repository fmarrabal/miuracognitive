"""
MiuraCognitive - Verificación del entorno (Sprint 1)
=====================================================
Comprueba que todo está listo antes de empezar a entrenar.

Detecta automáticamente si hay GPU CUDA disponible:
  - Si SÍ hay GPU  -> usa CUDA + BF16, comprueba Flash Attention 2.
  - Si NO hay GPU  -> usa CPU + FP32, omite chequeos específicos de GPU.
"""

import sys
import torch
import torch.nn.functional as F
from model.transformer import MiuraConfig, MiuraCognitive


def check(name: str, cond: bool, detail: str = "") -> bool:
    status = "OK  " if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -> {detail}" if detail else ""))
    return cond


def select_device_and_dtype():
    """CUDA -> BF16, CPU -> FP32 (BF16 en CPU es lento e inestable)."""
    if torch.cuda.is_available():
        return torch.device("cuda:0"), torch.bfloat16, True
    return torch.device("cpu"), torch.float32, False


def main() -> int:
    print("=" * 70)
    print("MiuraCognitive - Verificación del entorno")
    print("=" * 70)
    all_ok = True

    # --- 1. Versiones ---
    print("\n[1] Versiones")
    py_ok = sys.version_info >= (3, 10)
    all_ok &= check("Python >= 3.10", py_ok, f"{sys.version.split()[0]}")
    tv = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    all_ok &= check("PyTorch >= 2.5", tv >= (2, 5), f"{torch.__version__}")

    # --- 2. Dispositivo ---
    print("\n[2] Detección de dispositivo")
    device, dtype, on_gpu = select_device_and_dtype()
    if on_gpu:
        check("CUDA disponible", True)
        n_gpus = torch.cuda.device_count()
        check(f"GPUs detectadas: {n_gpus}", n_gpus >= 1)
        blackwell_found = False
        for i in range(n_gpus):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"      GPU {i}: {name} | sm_{cap[0]}{cap[1]} | {mem_gb:.1f} GB")
            if cap[0] >= 12:
                blackwell_found = True
        if blackwell_found:
            check("RTX Blackwell (sm_12x) presente", True)
        else:
            print("      [INFO] No se detecta Blackwell, usando GPU disponible.")
    else:
        print("  [INFO] No hay CUDA. Cayendo a CPU + FP32.")
        print(f"         Núcleos torch: {torch.get_num_threads()}")
        print("         Útil para validar código; entrenamiento real requiere GPU.")
    print(f"\n      -> Dispositivo: {device} | dtype: {dtype}")

    # --- 3. Test de dtype ---
    print(f"\n[3] Test de dtype ({dtype})")
    try:
        a = torch.randn(512, 512, device=device, dtype=dtype, requires_grad=True)
        b = torch.randn(512, 512, device=device, dtype=dtype)
        c = (a @ b).sum()
        c.backward()
        dt_ok = a.grad is not None and torch.isfinite(c).item()
        all_ok &= check(f"Matmul + backward en {dtype}", dt_ok, f"loss={c.item():.3f}")
    except Exception as e:
        all_ok &= check(f"Matmul + backward en {dtype}", False, str(e))

    # --- 4. SDPA ---
    print("\n[4] Scaled Dot-Product Attention")
    nota = "(Flash Attention 2 esperado en GPU)" if on_gpu else "(implementación CPU genérica; sin Flash Attention)"
    print(f"      {nota}")
    try:
        B, H, T, D = 2, 6, 64, 64
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        sdpa_ok = out.shape == (B, H, T, D) and torch.isfinite(out).all().item()
        all_ok &= check("SDPA causal", sdpa_ok, f"shape={tuple(out.shape)}")
    except Exception as e:
        all_ok &= check("SDPA causal", False, str(e))

    # --- 5. Modelo completo ---
    print(f"\n[5] MiuraCognitive base: forward + backward en {device.type.upper()}")
    try:
        if on_gpu:
            cfg = MiuraConfig()
            B, T = 4, 128
        else:
            cfg = MiuraConfig(max_seq_len=128)
            B, T = 2, 32
            print(f"      [INFO] CPU: usando batch reducido {B}x{T} para velocidad.")

        model = MiuraCognitive(cfg).to(device=device, dtype=dtype)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"      Parámetros totales: {n_params:,} (~{n_params/1e6:.1f}M)")

        idx = torch.randint(0, cfg.vocab_size, (B, T), device=device)
        targets = torch.randint(0, cfg.vocab_size, (B, T), device=device)
        logits, loss = model(idx, targets)
        loss.backward()

        model_ok = torch.isfinite(loss).item() and all(
            p.grad is not None and torch.isfinite(p.grad).all().item()
            for p in model.parameters() if p.requires_grad
        )
        all_ok &= check("Forward + backward del modelo", model_ok, f"loss={loss.item():.4f}")

        if on_gpu:
            mem_mb = torch.cuda.max_memory_allocated(device) / 1e6
            print(f"      VRAM pico: {mem_mb:.1f} MB en batch {B}x{T}")
    except Exception as e:
        all_ok &= check("Forward + backward del modelo", False, str(e))

    # --- Resumen ---
    print("\n" + "=" * 70)
    if all_ok:
        modo = "GPU" if on_gpu else "CPU (modo validación)"
        print(f"  TODO OK en {modo}. Listo para continuar.")
        if not on_gpu:
            print("  Recordatorio: el entrenamiento real requiere la Blackwell.")
        print("  Siguiente: Entrega 2 (dataset TinyStories + trainer).")
    else:
        print("  Hay fallos. Revisa los [FAIL] antes de continuar.")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
MiuraCognitive - Tarea-B: Recall con distractores y confirmación diferida
==========================================================================
Tarea sintética diseñada para que el ESTADO INTERNO importe.

Cada secuencia:
  <K> valor </K>  [ distractores de longitud variable ]  <Q>  -> valor

Un modelo sin memoria de inercia se degrada cuando los distractores son
largos. El HBP (con dinámica de onda y working memory) debería mantener
mejor el valor clave a través de distractores largos.

La dificultad se controla con la longitud de distractores, que VARÍA entre
secuencias. Esto permite estratificar la accuracy por longitud y ver si el
HBP ayuda más cuando la tarea es más difícil.

Autor: Francisco M. Arrabal (Curro) + asistencia Claude
"""

from __future__ import annotations
import itertools
import torch


class SyntheticRecallDataset:
    """Generador de la Tarea-B. Vocabulario reducido y autocontenido.

    Tokens especiales (IDs bajos):
      0: PAD, 1: <K>, 2: </K>, 3: <Q>, 4: -> (separador respuesta)
    Valores: IDs en [value_offset, value_offset + n_values)
    Distractores: IDs en [distractor_offset, distractor_offset + n_distractors)
    """

    PAD, K_OPEN, K_CLOSE, Q, ARROW = 0, 1, 2, 3, 4

    def __init__(self, n_values: int = 50, n_distractors: int = 100,
                 min_distractors: int = 2, max_distractors: int = 60,
                 seq_len: int = 128, seed: int = 0):
        self.n_values = n_values
        self.n_distractors = n_distractors
        self.min_d = min_distractors
        self.max_d = max_distractors
        self.seq_len = seq_len
        self.value_offset = 5
        self.distractor_offset = 5 + n_values
        self.vocab_size = 5 + n_values + n_distractors
        # Rango de tokens de respuesta válidos (para argmax restringido en eval)
        self.answer_offset, self.answer_size = self.value_offset, n_values
        self.gen = torch.Generator().manual_seed(seed)

    def _randint(self, low, high, size=()):
        return torch.randint(low, high, size, generator=self.gen)

    def sample(self):
        """Genera una secuencia. Devuelve (tokens, target_pos, n_distractors).

        target_pos es la posición donde el modelo debe predecir el valor.
        Solo esa posición cuenta en la pérdida (resto = ignore_index=-1)."""
        value = self.value_offset + int(self._randint(0, self.n_values, (1,)).item())
        n_dist = int(self._randint(self.min_d, self.max_d + 1, (1,)).item())

        seq = [self.K_OPEN, value, self.K_CLOSE]
        # Distractores
        for _ in range(n_dist):
            d = self.distractor_offset + int(self._randint(0, self.n_distractors, (1,)).item())
            seq.append(d)
        # Pregunta y respuesta
        seq.append(self.Q)
        seq.append(self.ARROW)
        target_pos = len(seq)        # posición donde irá el valor a predecir
        seq.append(value)            # el valor correcto (target)

        # Truncar o padear a seq_len
        seq = seq[: self.seq_len]
        target_pos = min(target_pos, self.seq_len - 1)
        if len(seq) < self.seq_len:
            seq = seq + [self.PAD] * (self.seq_len - len(seq))

        tokens = torch.tensor(seq, dtype=torch.long)
        return tokens, target_pos, n_dist

    def batch(self, batch_size: int):
        """Genera un batch. Devuelve:
          inputs:  (B, T) tokens de entrada (sin el último)
          targets: (B, T) targets con -1 salvo en target_pos
          n_dists: (B,) número de distractores por secuencia (para estratificar)
        """
        seqs, tpos, ndists = [], [], []
        for _ in range(batch_size):
            s, tp, nd = self.sample()
            seqs.append(s)
            tpos.append(tp)
            ndists.append(nd)
        tokens = torch.stack(seqs)                      # (B, T)
        # Entrada = tokens[:-1], objetivo desplazado
        inputs = tokens[:, :-1].contiguous()
        targets = torch.full_like(inputs, -1)           # ignore por defecto
        for b, tp in enumerate(tpos):
            if 0 < tp < tokens.size(1):
                # El target en la posición tp-1 de inputs es el valor en tp
                targets[b, tp - 1] = tokens[b, tp]
                # Enmascara la respuesta en el input: los canales de pooling
                # global (WM/mean) no son causales y la filtrarían (leak).
                if tp < inputs.size(1):
                    inputs[b, tp] = self.PAD
        return inputs, targets, torch.tensor(ndists)

    @staticmethod
    def bucket(difficulty: int) -> str:
        """Estratificación por longitud de distractor (eje de dificultad)."""
        return "corto" if difficulty <= 15 else ("medio" if difficulty <= 35 else "largo")


class RunningSumModDataset:
    """Tarea-RUNSUM (integración algorítmica). El objetivo es la SUMA CORRIENTE
    mod n de todos los dígitos ESCRITOS (marcados con <W>) a lo largo del stream,
    intercalados con distractores que hay que ignorar:

        [ d d <W> v1  d <W> v2  d d <W> v3 ... <Q> ->  (Σvi mod n) ]

    No es recall/copia: la respuesta no está en ninguna posición (es una
    reducción muchos-a-uno con wraparound no lineal que la atención softmax a
    profundidad fija no implementa exacto). Premia un acumulador estable y
    profundidad adaptativa. Dificultad = K (nº de escrituras): la fase del
    residuo debe rotar e incrementarse SIN decaer -> mapea sobre la oscilación
    2º orden (hbp_full) frente a la relajación 1er orden que la fuga (hbp_first).

    Tokens: 0:PAD, 1:<W> (write), 2:<Q>, 3:-> (ARROW).
    Dígitos/valores y respuesta: IDs en [value_offset, value_offset + mod).
    Distractores: IDs en [distractor_offset, distractor_offset + n_distractors).
    """

    PAD, W, Q, ARROW = 0, 1, 2, 3

    def __init__(self, mod: int = 3, n_distractors: int = 100,
                 min_writes: int = 2, max_writes: int = 16,
                 seq_len: int = 128, seed: int = 0, **kwargs):
        self.mod = mod
        self.n_distractors = n_distractors
        self.min_w = min_writes
        self.max_w = max_writes
        self.seq_len = seq_len
        self.value_offset = 4
        self.distractor_offset = 4 + mod
        self.vocab_size = 4 + mod + n_distractors
        self.answer_offset, self.answer_size = self.value_offset, mod
        self.gen = torch.Generator().manual_seed(seed)

    def _randint(self, low, high):
        return int(torch.randint(low, high, (1,), generator=self.gen).item())

    def _distractor(self):
        return self.distractor_offset + self._randint(0, self.n_distractors)

    def sample(self):
        """Devuelve (tokens, labels, K) con SUPERVISIÓN DENSA.

        labels[p] = residuo corriente (token) que el modelo debe predecir tras
        ver la posición p. Se supervisa en CADA valor escrito (el residuo
        acumulado hasta ahí) y en la posición de <ARROW> (la respuesta final).
        El residuo NUNCA aparece como token de entrada -> sin atajo de copia.
        """
        K = self._randint(self.min_w, self.max_w + 1)
        digits = [self._randint(0, self.mod) for _ in range(K)]

        # Presupuesto de distractores que garantiza que TODAS las K escrituras
        # caben (sin truncar) junto a <Q> -> respuesta.
        budget = max(0, self.seq_len - 3 - 2 * K)
        n_dist = self._randint(0, budget + 1)
        gaps = [0] * (K + 1)
        for _ in range(n_dist):
            gaps[self._randint(0, K + 1)] += 1

        seq, sup = [], []        # sup: (posición, residuo) supervisados
        running = 0
        for i in range(K):
            seq += [self._distractor()] * gaps[i]
            seq.append(self.W)
            vpos = len(seq)                      # posición del valor escrito
            seq.append(self.value_offset + digits[i])
            running = (running + digits[i]) % self.mod
            sup.append((vpos, running))          # tras ver vi -> predecir residuo
        seq += [self._distractor()] * gaps[K]
        seq.append(self.Q)
        arrow_pos = len(seq)
        seq.append(self.ARROW)
        # OJO: la respuesta NO se añade al input. Un token de respuesta tras
        # <ARROW> se filtraría a la posición supervisada por los canales de
        # pooling global del modelo (WM/mean), que no son causales (leak).
        sup.append((arrow_pos, running))         # lectura final en <ARROW>

        seq = seq[: self.seq_len]
        if len(seq) < self.seq_len:
            seq = seq + [self.PAD] * (self.seq_len - len(seq))
        tokens = torch.tensor(seq, dtype=torch.long)
        labels = torch.full((self.seq_len,), -1, dtype=torch.long)
        for pos, r in sup:
            if pos < self.seq_len:
                labels[pos] = self.value_offset + r
        return tokens, labels, K

    def batch(self, batch_size: int):
        toks, labs, diffs = [], [], []
        for _ in range(batch_size):
            t, l, k = self.sample()
            toks.append(t)
            labs.append(l)
            diffs.append(k)
        tokens = torch.stack(toks)
        labels = torch.stack(labs)
        # labels[p] es la etiqueta en la posición de entrada p (no es next-token):
        # el modelo en p ha visto tokens[0..p] y debe predecir el residuo.
        inputs = tokens[:, :-1].contiguous()
        targets = labels[:, :-1].contiguous()
        return inputs, targets, torch.tensor(diffs)

    @staticmethod
    def bucket(difficulty: int) -> str:
        """Estratificación por nº de escrituras K (profundidad de integración)."""
        return "corto" if difficulty <= 4 else ("medio" if difficulty <= 9 else "largo")


class PermutationCompositionDataset:
    """Tarea de SEGUIMIENTO DE ESTADO no conmutativo (composición en S_n).

        [ g1 g2 ... gK <Q> -> ]   target = composición de g1..gK EN ORDEN

    Estado = permutación de n elementos; cada op = transposición ADYACENTE
    (generadores de S_n). La respuesta depende del OUTPUT de cada paso (no de
    los inputs por separado): es una función NO ABELIANA que un transformer de
    profundidad fija no puede paralelizar (palabra en S_5 es NC^1-dura) -> se
    necesita recurrencia. Es el primer test JUSTO del HBP como modulador de
    pensamiento iterativo: el reasoner DEBE iterar ~K veces, y el nº de
    iteraciones debería crecer con la dificultad K.

    Supervisión densa: la permutación corriente tras cada op (token de
    permutación, que NUNCA aparece como input -> sin atajo de copia).
    Difficulty = K (profundidad de composición).

    Tokens: 0:PAD, 1:<Q>, 2:-> ; generadores en [3, 3+n_gen);
    permutaciones (targets) en [3+n_gen, 3+n_gen+n!).
    """

    PAD, Q, ARROW = 0, 1, 2

    def __init__(self, n_elems: int = 5, min_ops: int = 2, max_ops: int = 24,
                 seq_len: int = 128, seed: int = 0, gen_set: str = "adjacent", **kwargs):
        self.n = n_elems
        self.min_ops = min_ops
        self.max_ops = max_ops
        self.seq_len = seq_len
        self.perms = list(itertools.permutations(range(n_elems)))
        self.perm_index = {p: i for i, p in enumerate(self.perms)}
        self.identity = tuple(range(n_elems))
        # Generadores de S_n (dos sets distintos para test de robustez del hallazgo).
        if gen_set == "adjacent":
            # Transposiciones adyacentes t_j = (j, j+1): swaps LOCALES.
            self.gens = []
            for j in range(n_elems - 1):
                g = list(range(n_elems))
                g[j], g[j + 1] = g[j + 1], g[j]
                self.gens.append(tuple(g))
        elif gen_set == "cycle_transp":
            # 5-ciclo (rotación GLOBAL i->i+1) + transposición (0 1). Generan S_n.
            cycle = tuple((i + 1) % n_elems for i in range(n_elems))
            t = list(range(n_elems))
            t[0], t[1] = t[1], t[0]
            self.gens = [cycle, tuple(t)]
        else:
            raise ValueError(f"gen_set desconocido: {gen_set}")
        self.n_gen = len(self.gens)
        self.gen_offset = 3
        self.perm_offset = 3 + self.n_gen
        self.vocab_size = 3 + self.n_gen + len(self.perms)
        self.answer_offset, self.answer_size = self.perm_offset, len(self.perms)
        self.gen = torch.Generator().manual_seed(seed)

    def _compose(self, a, b):
        """(a ∘ b)[i] = a[b[i]]: aplica b y luego a."""
        return tuple(a[b[i]] for i in range(self.n))

    def _randint(self, low, high):
        return int(torch.randint(low, high, (1,), generator=self.gen).item())

    def sample(self):
        K = self._randint(self.min_ops, self.max_ops + 1)
        seq, sup = [], []
        running = self.identity
        for _ in range(K):
            j = self._randint(0, self.n_gen)
            seq.append(self.gen_offset + j)
            running = self._compose(self.gens[j], running)
            sup.append((len(seq) - 1, self.perm_index[running]))   # tras gj -> perm corriente
        seq.append(self.Q)
        arrow_pos = len(seq)
        seq.append(self.ARROW)
        sup.append((arrow_pos, self.perm_index[running]))          # lectura final en <ARROW>
        # La respuesta debe sobrevivir al filtro pos<seq_len Y al corte [:, :-1];
        # si no, evaluate mediría una posición intermedia como si fuera la final.
        assert arrow_pos < self.seq_len - 1, "arrow_pos truncado: sube seq_len o baja max_ops"

        seq = seq[: self.seq_len]
        if len(seq) < self.seq_len:
            seq = seq + [self.PAD] * (self.seq_len - len(seq))
        tokens = torch.tensor(seq, dtype=torch.long)
        labels = torch.full((self.seq_len,), -1, dtype=torch.long)
        for pos, pidx in sup:
            if pos < self.seq_len:
                labels[pos] = self.perm_offset + pidx
        return tokens, labels, K

    def batch(self, batch_size: int):
        toks, labs, diffs = [], [], []
        for _ in range(batch_size):
            t, l, k = self.sample()
            toks.append(t)
            labs.append(l)
            diffs.append(k)
        tokens = torch.stack(toks)
        labels = torch.stack(labs)
        inputs = tokens[:, :-1].contiguous()
        targets = labels[:, :-1].contiguous()
        return inputs, targets, torch.tensor(diffs)

    @staticmethod
    def bucket(difficulty: int) -> str:
        """Estratificación por profundidad de composición K."""
        return "corto" if difficulty <= 6 else ("medio" if difficulty <= 13 else "largo")


class IteratedFunctionDataset:
    """Pointer-chasing / ITERACIÓN FUNCIONAL (segunda tarea, estructura distinta
    a la composición de grupo). Dada una función f:[n]->[n] (su tabla va en la
    entrada) y un estado inicial x0, la respuesta es f^K(x0) = aplicar f K veces:

        [ f(0) f(1) ... f(n-1) <SEP> x0  <A> <A> ... <A>  <Q> -> ]   target = f^K(x0)

    Inherentemente SECUENCIAL: cada paso depende del OUTPUT del anterior (caminar
    el grafo funcional), no paralelizable a profundidad fija -> fuerza al reasoner
    a iterar. f varía por secuencia (no memorizable). Supervisión densa: estado
    corriente tras cada aplicación. Difficulty = K (longitud del camino).

    Tokens: 0:PAD, 1:<SEP>, 2:<A> (aplica), 3:<Q>, 4:-> ; estados en [5, 5+n).
    """

    PAD, SEP, APPLY, Q, ARROW = 0, 1, 2, 3, 4

    def __init__(self, n_states: int = 8, min_ops: int = 2, max_ops: int = 24,
                 seq_len: int = 128, seed: int = 0, **kwargs):
        self.n = n_states
        self.min_ops = min_ops
        self.max_ops = max_ops
        self.seq_len = seq_len
        self.state_offset = 5
        self.vocab_size = 5 + n_states
        self.answer_offset, self.answer_size = self.state_offset, n_states
        self.gen = torch.Generator().manual_seed(seed)

    def _ri(self, lo, hi):
        return int(torch.randint(lo, hi, (1,), generator=self.gen).item())

    def sample(self):
        K = self._ri(self.min_ops, self.max_ops + 1)
        f = [self._ri(0, self.n) for _ in range(self.n)]   # tabla de f
        x0 = self._ri(0, self.n)

        seq, sup = [], []
        for i in range(self.n):
            seq.append(self.state_offset + f[i])           # f(i) en la posición i
        seq.append(self.SEP)
        seq.append(self.state_offset + x0)
        running = x0
        for _ in range(K):
            seq.append(self.APPLY)
            running = f[running]
            sup.append((len(seq) - 1, running))            # tras <A> -> estado corriente
        seq.append(self.Q)
        arrow_pos = len(seq)
        seq.append(self.ARROW)
        sup.append((arrow_pos, running))                   # lectura final

        seq = seq[: self.seq_len]
        if len(seq) < self.seq_len:
            seq = seq + [self.PAD] * (self.seq_len - len(seq))
        tokens = torch.tensor(seq, dtype=torch.long)
        labels = torch.full((self.seq_len,), -1, dtype=torch.long)
        for pos, s in sup:
            if pos < self.seq_len:
                labels[pos] = self.state_offset + s
        return tokens, labels, K

    def batch(self, batch_size: int):
        toks, labs, diffs = [], [], []
        for _ in range(batch_size):
            t, l, k = self.sample()
            toks.append(t)
            labs.append(l)
            diffs.append(k)
        tokens = torch.stack(toks)
        labels = torch.stack(labs)
        inputs = tokens[:, :-1].contiguous()
        targets = labels[:, :-1].contiguous()
        return inputs, targets, torch.tensor(diffs)

    @staticmethod
    def bucket(difficulty: int) -> str:
        """Estratificación por longitud del camino K."""
        return "corto" if difficulty <= 6 else ("medio" if difficulty <= 13 else "largo")


class DualRegimePhaseSettleDataset:
    """Tarea DUAL de regímenes entremezclados (diseño del panel de crítica): dos
    sub-tareas con la MISMA gramática de iteración funcional pero DEMANDA DE
    MODULACIÓN temporal opuesta, señalizadas por un token de MODO observable.

        [MODE] f(0) ... f(n-1) <SEP> x0 { NOOP* <A> }×K NOOP* <Q> ->   target = f^K(x0)

    - Modo PHASE (MODE_P): f = involución SIN puntos fijos (matching perfecto:
      todos los ciclos de longitud 2). f^K(x0) alterna con la PARIDAD de K:
      el estado corriente oscila con período 2 en el eje de iteraciones ->
      la masa de halting debe concentrarse en ticks de la paridad correcta y
      los gates deben alternar: demanda OSCILATORIA (favorece la rama onda,
      que tiene memoria de fase; una dinámica de 1er orden es monótona).
    - Modo SETTLE (MODE_S): f = contracción hacia un punto fijo absorbente
      (árbol de profundidad ≤3, f(x)=padre, f(r)=r). La respuesta se asienta
      en pocos pasos y de ahí es constante: demanda MONÓTONA de asentarse y
      parar (favorece difusión; la onda resuena y produce paradas erróneas).

    Los NOOP (relleno aleatorio) DESCORRELACIONAN la longitud de secuencia de K
    (cierran el atajo "paridad por longitud" vía frac_valid). Ambos modos
    comparten el sub-vocabulario de respuesta (estados) -> eval sin máscaras
    por modo. Supervisión densa: estado corriente tras cada <A> + lectura final.
    Difficulty = K. batch() devuelve además el MODO por muestra (0=PHASE,
    1=SETTLE) para el oráculo de física (alpha_force).

    Tokens: 0:PAD, 1:<SEP>, 2:<A>, 3:<Q>, 4:-> ; estados en [5, 5+n);
    MODE_P=5+n, MODE_S=6+n, NOOP=7+n.
    """

    PAD, SEP, APPLY, Q, ARROW = 0, 1, 2, 3, 4

    def __init__(self, n_states: int = 8, min_ops: int = 2, max_ops: int = 24,
                 seq_len: int = 128, seed: int = 0, mode: str = "dual",
                 max_fillers: int = 32, **kwargs):
        assert n_states % 2 == 0, "PHASE requiere n par (matching perfecto)"
        self.n = n_states
        self.min_ops = min_ops
        self.max_ops = max_ops
        self.seq_len = seq_len
        self.mode = mode                     # "dual" | "phase" | "settle"
        self.max_fillers = max_fillers
        self.state_offset = 5
        self.MODE_P = 5 + n_states
        self.MODE_S = 6 + n_states
        self.NOOP = 7 + n_states
        self.vocab_size = 8 + n_states
        self.answer_offset, self.answer_size = self.state_offset, n_states
        self.gen = torch.Generator().manual_seed(seed)

    def _ri(self, lo, hi):
        return int(torch.randint(lo, hi, (1,), generator=self.gen).item())

    def _sample_phase_f(self):
        """Involución sin puntos fijos: matching perfecto aleatorio de [n]."""
        idx = torch.randperm(self.n, generator=self.gen).tolist()
        f = [0] * self.n
        for i in range(0, self.n, 2):
            a, b = idx[i], idx[i + 1]
            f[a], f[b] = b, a
        return f

    def _sample_settle_f(self):
        """Contracción: árbol aleatorio de profundidad ≤3 con raíz absorbente."""
        order = torch.randperm(self.n, generator=self.gen).tolist()
        r = order[0]
        others = order[1:]
        k1 = max(1, len(others) // 3)
        k2 = max(1, (len(others) - k1) // 2)
        level1, level2, level3 = others[:k1], others[k1:k1 + k2], others[k1 + k2:]
        f = [r] * self.n
        f[r] = r
        for x in level1:
            f[x] = r
        for j, x in enumerate(level2):
            f[x] = level1[j % len(level1)]
        for j, x in enumerate(level3):
            f[x] = level2[j % len(level2)] if level2 else level1[j % len(level1)]
        return f

    def sample(self):
        mode = self.mode if self.mode != "dual" else ("phase" if self._ri(0, 2) == 0 else "settle")
        f = self._sample_phase_f() if mode == "phase" else self._sample_settle_f()
        K = self._ri(self.min_ops, self.max_ops + 1)
        x0 = self._ri(0, self.n)

        seq = [self.MODE_P if mode == "phase" else self.MODE_S]
        seq += [self.state_offset + f[i] for i in range(self.n)]
        seq += [self.SEP, self.state_offset + x0]
        # Relleno NOOP repartido: longitud ⊥ K (cierra el atajo de paridad).
        n_fill = self._ri(0, self.max_fillers + 1)
        gaps = [0] * (K + 1)
        for _ in range(n_fill):
            gaps[self._ri(0, K + 1)] += 1

        sup, running = [], x0
        for i in range(K):
            seq += [self.NOOP] * gaps[i]
            seq.append(self.APPLY)
            running = f[running]
            sup.append((len(seq) - 1, running))          # tras <A> -> estado corriente
        seq += [self.NOOP] * gaps[K]
        seq.append(self.Q)
        arrow_pos = len(seq)
        seq.append(self.ARROW)
        sup.append((arrow_pos, running))                 # lectura final
        assert arrow_pos < self.seq_len - 1, "arrow_pos truncado: sube seq_len o baja max_ops/max_fillers"

        seq = seq[: self.seq_len]
        if len(seq) < self.seq_len:
            seq = seq + [self.PAD] * (self.seq_len - len(seq))
        tokens = torch.tensor(seq, dtype=torch.long)
        labels = torch.full((self.seq_len,), -1, dtype=torch.long)
        for pos, s in sup:
            if pos < self.seq_len:
                labels[pos] = self.state_offset + s
        return tokens, labels, K, (0 if mode == "phase" else 1)

    def batch(self, batch_size: int):
        """Devuelve (inputs, targets, diffs, modes). modes: (B,) 0=PHASE 1=SETTLE."""
        toks, labs, diffs, modes = [], [], [], []
        for _ in range(batch_size):
            t, l, k, m = self.sample()
            toks.append(t); labs.append(l); diffs.append(k); modes.append(m)
        tokens = torch.stack(toks)
        labels = torch.stack(labs)
        inputs = tokens[:, :-1].contiguous()
        targets = labels[:, :-1].contiguous()
        return inputs, targets, torch.tensor(diffs), torch.tensor(modes)

    @staticmethod
    def bucket(difficulty: int) -> str:
        return "corto" if difficulty <= 6 else ("medio" if difficulty <= 13 else "largo")


class TwinStreamCompositionDataset:
    """Tarea dual v2 (tras el piloto PHASE/SETTLE): la demanda oscilatoria se monta
    SOBRE una computación NC¹-dura que ya fuerza iteración (composición en S_5).

        [MODE] <SA> g <SB> g <SA> g ... <QA|QB> ->   target = perm final del stream pedido

    - Modo TWIN (MODE_T): DOS streams independientes de permcomp entrelazados en
      alternancia estricta A,B,A,B. El reasoner (una WM, un estado) debe
      MULTIPLEXAR EN EL TIEMPO el seguimiento de ambos estados: la demanda de
      modulación (a qué estado escribe/lee la WM) alterna con período 2 en el eje
      de ticks -> oscilatoria (favorece la memoria de fase de la rama onda).
    - Modo MONO (MODE_M): un solo stream (permcomp estándar con la misma
      gramática, tags todos <SA>). Demanda monótona (favorece difusión).

    Ambos modos son composición en S_5 (NC¹-dura -> el reasoner DEBE iterar; el
    piloto PHASE fracasó precisamente por colapso de halting con supervisión
    intermedia trivial). Supervisión densa: tras cada op, la perm corriente de SU
    stream. Ambos modos comparten sub-vocab de respuesta (las 120 perms).
    Difficulty = K (ops totales; TWIN reparte ~K/2 por stream).
    batch() devuelve (inputs, targets, diffs, modes); modes: 0=TWIN, 1=MONO.

    Tokens: 0:PAD, 1:<SA>, 2:<SB>, 3:<QA>, 4:<QB>, 5:-> ; generadores en [6,10);
    perms en [10,130); MODE_T=130, MODE_M=131.
    """

    PAD, SA, SB, QA, QB, ARROW = 0, 1, 2, 3, 4, 5

    def __init__(self, n_elems: int = 5, min_ops: int = 2, max_ops: int = 24,
                 seq_len: int = 128, seed: int = 0, mode: str = "dual", **kwargs):
        self.n = n_elems
        self.min_ops = min_ops
        self.max_ops = max_ops
        self.seq_len = seq_len
        self.mode = mode                       # "dual" | "twin" | "mono"
        self.perms = list(itertools.permutations(range(n_elems)))
        self.perm_index = {p: i for i, p in enumerate(self.perms)}
        self.identity = tuple(range(n_elems))
        self.gens = []
        for j in range(n_elems - 1):           # transposiciones adyacentes
            g = list(range(n_elems))
            g[j], g[j + 1] = g[j + 1], g[j]
            self.gens.append(tuple(g))
        self.n_gen = len(self.gens)
        self.gen_offset = 6
        self.perm_offset = 6 + self.n_gen
        self.MODE_T = self.perm_offset + len(self.perms)
        self.MODE_M = self.MODE_T + 1
        self.vocab_size = self.MODE_M + 1
        self.answer_offset, self.answer_size = self.perm_offset, len(self.perms)
        self.gen = torch.Generator().manual_seed(seed)

    def _compose(self, a, b):
        return tuple(a[b[i]] for i in range(self.n))

    def _ri(self, lo, hi):
        return int(torch.randint(lo, hi, (1,), generator=self.gen).item())

    def sample(self):
        mode = self.mode if self.mode != "dual" else ("twin" if self._ri(0, 2) == 0 else "mono")
        K = self._ri(self.min_ops, self.max_ops + 1)
        seq = [self.MODE_T if mode == "twin" else self.MODE_M]
        sup = []
        running = {0: self.identity, 1: self.identity}     # stream A=0, B=1
        for i in range(K):
            st = (i % 2) if mode == "twin" else 0          # alternancia estricta en TWIN
            seq.append(self.SA if st == 0 else self.SB)
            j = self._ri(0, self.n_gen)
            seq.append(self.gen_offset + j)
            running[st] = self._compose(self.gens[j], running[st])
            sup.append((len(seq) - 1, self.perm_index[running[st]]))
        q = (self._ri(0, 2) if mode == "twin" else 0)      # stream consultado
        seq.append(self.QA if q == 0 else self.QB)
        arrow_pos = len(seq)
        seq.append(self.ARROW)
        sup.append((arrow_pos, self.perm_index[running[q]]))
        assert arrow_pos < self.seq_len - 1, "arrow_pos truncado: sube seq_len o baja max_ops"

        seq = seq[: self.seq_len]
        if len(seq) < self.seq_len:
            seq = seq + [self.PAD] * (self.seq_len - len(seq))
        tokens = torch.tensor(seq, dtype=torch.long)
        labels = torch.full((self.seq_len,), -1, dtype=torch.long)
        for pos, pidx in sup:
            if pos < self.seq_len:
                labels[pos] = self.perm_offset + pidx
        return tokens, labels, K, (0 if mode == "twin" else 1)

    def batch(self, batch_size: int):
        """(inputs, targets, diffs, modes); modes: 0=TWIN, 1=MONO."""
        toks, labs, diffs, modes = [], [], [], []
        for _ in range(batch_size):
            t, l, k, m = self.sample()
            toks.append(t); labs.append(l); diffs.append(k); modes.append(m)
        tokens = torch.stack(toks)
        labels = torch.stack(labs)
        inputs = tokens[:, :-1].contiguous()
        targets = labels[:, :-1].contiguous()
        return inputs, targets, torch.tensor(diffs), torch.tensor(modes)

    @staticmethod
    def bucket(difficulty: int) -> str:
        return "corto" if difficulty <= 6 else ("medio" if difficulty <= 13 else "largo")


if __name__ == "__main__":
    ds = SyntheticRecallDataset(seq_len=64, max_distractors=30, seed=42)
    print(f"Vocab size: {ds.vocab_size}")
    inp, tgt, nd = ds.batch(4)
    print(f"inputs: {inp.shape}, targets: {tgt.shape}")
    print(f"n_distractors por muestra: {nd.tolist()}")
    # Verifica que hay exactamente un target válido por secuencia
    valid = (tgt != -1).sum(dim=1)
    print(f"targets válidos por secuencia: {valid.tolist()} (debe ser 1 cada uno)")

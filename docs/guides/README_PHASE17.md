# CorrDiff — Fase 17: Desenho Formal das Ablações de Condicionamento

Versão: `phase17-conditioning-ablations-v1-nvidia-corrdiff-design`

A Fase 17 **não treina modelos**. Ela congela o desenho experimental das
ablações que serão executadas posteriormente **dentro do ambiente nativo
NVIDIA CorrDiff**.

O objetivo é responder, de forma controlada:

```text
quais famílias ERA5 são suficientes?
quais famílias são necessárias?
variáveis derivadas ajudam a representação?
esses efeitos permanecem no regression model e no CorrDiff?
```

O desenho preserva integralmente o protocolo temporal das Fases 15–16.

---

## 1. Contrato temporal congelado

```text
train            = 2011–2020
validation       = 2021–2022
test_primary     = 2023       [LOCKED]
test_stress_ood  = 2024       [LOCKED]
purge            = ±48 h
```

Durante a Fase 17:

```text
2023 e 2024 não são acessados
```

para seleção de canais, arquitetura, loss, checkpoint ou hiperparâmetros.

---

## 2. Famílias dos 12 predictors raw

### Moisture

```text
tcwv
r_850
r_500
```

### Thermal

```text
t2m
t_850
t_500
```

### Dynamics

```text
u10
v10
u_850
v_850
u_500
v_500
```

A ordem canônica `raw12` permanece:

```text
tcwv
t2m
u10
v10
t_850
r_850
u_850
v_850
t_500
r_500
u_500
v_500
```

---

## 3. Variáveis derivadas

Ordem canônica:

```text
wind_speed_10
wind_speed_850
wind_speed_500
delta_t_500_850
delta_t_850_surface
delta_r_500_850
bulk_wind_diff_10_850
bulk_wind_diff_850_500
```

Fórmulas:

```text
wind_speed_10
= sqrt(u10² + v10²)

wind_speed_850
= sqrt(u_850² + v_850²)

wind_speed_500
= sqrt(u_500² + v_500²)

delta_t_500_850
= t_500 - t_850

delta_t_850_surface
= t_850 - t2m

delta_r_500_850
= r_500 - r_850

bulk_wind_diff_10_850
= sqrt((u10-u_850)² + (v10-v_850)²)

bulk_wind_diff_850_500
= sqrt((u_850-u_500)² + (v_850-v_500)²)
```

Duas regras de terminologia permanecem congeladas:

```text
delta_t
→ diferença de temperatura
→ não chamar automaticamente de "instabilidade"

bulk_wind_diff
→ diferença vetorial de vento em m/s
→ não é shear normalizado por altura
```

Além disso, todas as variáveis derivadas são transformações determinísticas
dos `raw12`. Portanto:

> um eventual ganho com derived channels é ganho de representação/otimização,
> não evidência de nova informação meteorológica.

---

# 4. Matriz formal de ablações

## C00 — raw12_reference

```text
12 canais raw
```

É a referência principal.

---

## C01 — moisture_only

```text
tcwv
r_850
r_500
```

Testa **suficiência da umidade**.

Hipótese a testar, não assumir: retenção relativamente forte do sinal de
ocorrência/moderado.

---

## C02 — thermal_only

```text
t2m
t_850
t_500
```

Testa **suficiência térmica**.

Hipótese a testar: contribuição relativa maior para thresholds extremos do que
para ocorrência moderada.

---

## C03 — dynamics_only

```text
u10
v10
u_850
v_850
u_500
v_500
```

Testa **suficiência dinâmica**.

A motivação principal é verificar se os ventos acrescentam organização espacial
mesmo tendo associações escalares fracas nas fases anteriores.

---

## C04 — no_dynamics

```text
moisture + thermal
```

Comparação:

```text
C00 - C04
```

mede a contribuição incremental associada à família dinâmica.

---

## C05 — no_thermal

```text
moisture + dynamics
```

Comparação:

```text
C00 - C05
```

mede a contribuição incremental associada à família térmica.

---

## C06 — no_moisture

```text
thermal + dynamics
```

Comparação:

```text
C00 - C06
```

mede a contribuição incremental associada à família de umidade.

As comparações C04/C05/C06 são os testes **leave-one-family-out** e são mais
apropriadas para discutir necessidade condicional das famílias do que comparar
apenas modelos single-family.

---

## C07 — raw12_plus_all_derived

```text
raw12 + 8 derived
= 20 canais
```

É o teste principal de **representação derivada**.

Comparação:

```text
C00 vs C07
```

---

## C08 — raw12_plus_moisture_derived

```text
raw12
+
delta_r_500_850
```

---

## C09 — raw12_plus_thermal_derived

```text
raw12
+
delta_t_500_850
delta_t_850_surface
```

---

## C10 — raw12_plus_dynamics_derived

```text
raw12
+
wind_speed_10
wind_speed_850
wind_speed_500
bulk_wind_diff_10_850
bulk_wind_diff_850_500
```

C08–C10 ajudam a localizar eventual ganho de C07.

---

# 5. Três perguntas científicas diferentes

A matriz foi desenhada para evitar misturar três questões.

### Suficiência

```text
C01
C02
C03
```

Pergunta:

> quanto uma família consegue explicar sozinha?

### Necessidade condicional

```text
C00 vs C04
C00 vs C05
C00 vs C06
```

Pergunta:

> o que se perde ao retirar uma família do conjunto completo?

Essa é uma questão diferente de suficiência, especialmente sob
multicolinearidade.

### Representação derivada

```text
C00 vs C07/C08/C09/C10
```

Pergunta:

> oferecer transformações físicas/determinísticas explicitamente facilita o
> aprendizado?

Não perguntar:

> essas variáveis contêm nova informação?

porque matematicamente não contêm.

---

# 6. Execução em dois estágios no NVIDIA CorrDiff

## Stage R — regression screening

Executar **todos os 11 experimentos** no regression model nativo do CorrDiff:

```text
C00 ... C10
```

Sementes:

```text
42
43
44
```

Total planejado:

```text
11 × 3 = 33 runs
```

Somente:

```text
train → treinamento
validation → checkpoint/avaliação
```

Objetivos:

```text
avaliar sinal determinístico;
detectar famílias claramente dispensáveis;
avaliar derived representations;
obter baseline aprendido no mesmo ecossistema NVIDIA.
```

---

## Stage D — CorrDiff confirmation

O núcleo obrigatório é:

```text
C00 raw12
C04 no_dynamics
C05 no_thermal
C06 no_moisture
C07 raw12_plus_all_derived
```

com:

```text
3 seeds
```

Total obrigatório:

```text
5 × 3 = 15 CorrDiff runs
```

Esses cinco experimentos permitem testar:

```text
referência
+
necessidade das três famílias
+
efeito agregado de derived representation
```

Experimentos opcionais:

```text
C01 C02 C03
C08 C09 C10
```

Eles podem ser executados se houver orçamento ou se forem necessários para
resolver uma ambiguidade do Stage R.

A decisão de executar um opcional deve ser registrada **antes do treinamento
correspondente**, não após observar 2023/2024.

---

# 7. O que deve permanecer idêntico entre ablações

Dentro de cada estágio, apenas o condicionamento muda.

Devem permanecer congelados:

```text
Phase-15 split
purge temporal
target
patch geometry
regression architecture
CorrDiff architecture
optimizer family
learning-rate schedule
training budget
batch-size policy
checkpoint rule
seeds
metric implementation
validation timestamps
```

Isso é essencial para atribuir diferenças ao condicionamento e não a mudanças
simultâneas de treinamento.

---

# 8. Normalização

Para cada experimento:

```text
fit mean/std somente em train
```

para exatamente os canais presentes naquele experimento.

Exemplo:

```text
C01
→ fit normalization para tcwv/r850/r500 em train

C07
→ raw12 + derived8
→ calcular derived no train
→ fit mean/std dos 20 canais no train
```

Não reutilizar estatística de um canal derivado a partir de uma estatística raw
aproximada.

Nunca ajustar normalization em validation/test.

---

# 9. Métricas de seleção em validation

A Fase 17 herda as métricas congeladas da Fase 16:

```text
RMSE
MAE
bias

20/30/40/45 dBZ:
Precision
POD
FAR
CSI
F1
Frequency Bias
ETS

FSS:
2 / 4 / 8 / 16 km

max dBZ por patch:
all
target >=30
target >=40
target >=45

DJF/MAM/JJA/SON
```

No CorrDiff probabilístico também devem ser habilitados, quando suportados pelo
pipeline de avaliação:

```text
Brier de P(pixel >= threshold)
reliability/ECE
CRPS ou score de distribuição equivalente
```

A definição exata de métricas de ensemble será implementada e congelada na fase
de integração antes da avaliação final.

---

# 10. Não usar um score único

A seleção não deve reduzir todo o problema a:

```text
score = número único
```

A leitura deve ser hierárquica/Pareto.

Prioridade experimental:

```text
1. skill espacial/evento
   FSS / CSI >=30 e >=40

2. intensidade nos eventos
   max-dBZ error em >=30/40/45

3. comportamento extremo
   >=45 como diagnóstico crítico, porém mais raro

4. métricas probabilísticas do ensemble

5. erro global
   RMSE / MAE / bias
```

Isso impede que a enorme massa de zeros domine a escolha de conditioning.

---

# 11. Seeds

Toda configuração executada formalmente deve usar:

```text
42
43
44
```

Reportar:

```text
cada seed
+
mediana
+
dispersão
```

Não reportar apenas:

```text
best seed
```

como resultado do experimento.

---

# 12. Como materializar o plano

Primeiro valide os arquivos do pacote:

```bash
python scripts/17_validate_ablation_plan.py   --config-dir configs
```

Resultado esperado:

```text
STATUS: PASS
```

Depois gere os manifestos, validando também o handoff da Fase 16:

```bash
python scripts/17_build_ablation_manifest.py   --package-config-dir configs   --phase16-handoff-dir analysis_outputs/16_baselines/nvidia_handoff   --output-dir analysis_outputs/17_conditioning_ablations   --overwrite
```

Outputs:

```text
ablation_matrix.csv
ablation_matrix.json

planned_runs_stage_R.csv
planned_runs_stage_D_mandatory.csv
optional_corrdiff_experiments.csv

nvidia_ablation_adapter_contract.json
phase17_manifest.json
```

---

# 13. Contrato para a Fase 18

O arquivo:

```text
nvidia_ablation_adapter_contract.json
```

não inventa nomes de classes ou caminhos Hydra.

Ele descreve apenas:

```text
experiment_id
channel order
channel count
derived channels
normalization policy
split policy
```

A **Fase 18** será responsável por abrir o checkout real do NVIDIA CorrDiff e
mapear esse contrato para:

```text
Dataset
DataLoader
Hydra configs
regression model
CorrDiff
checkpointing
evaluation
```

Isso evita acoplar a Fase 17 a uma API/configuração NVIDIA que ainda não foi
verificada no código real.

---

# 14. Hipóteses pré-registradas

As seguintes hipóteses são registradas antes do treinamento:

1. **Moisture** pode reter mais informação para ocorrência/moderados.
2. **Thermal** pode aumentar sua contribuição relativa nos extremos.
3. **Dynamics** pode ajudar estrutura espacial mesmo se correlações escalares
   forem pequenas.
4. **Derived** pode facilitar representação/otimização, sem adicionar informação.
5. Leave-one-family-out pode revelar valor condicional escondido por
   multicolinearidade.

Elas são hipóteses a testar, não conclusões.

---

# 15. Resumo da fase

Depois de materializar o plano:

```bash
python scripts/17_summarize_ablation_plan.py   --phase17-dir analysis_outputs/17_conditioning_ablations   --output analysis_outputs/summary_phase17.txt
```

A Fase 17 estará concluída quando:

```text
design validation = PASS
channel matrix congelada
Stage R runs congelados
Stage D mandatory runs congelados
optional-run rule congelada
test access = False
adapter contract gerado
```

Nenhum treinamento é necessário para encerrar esta fase.

# CorrDiff - Fase 14b: auditoria dirigida dos episódios extremos

Implementação complementar às fases 9-14, antes do split formal da Fase 15.
Não altera o Zarr, o cache nem define exclusões de treinamento. Não é rastreamento
meteorológico de tempestades: um episódio é uma sequência de timestamps com ao
menos um pixel válido acima do limiar, separados exatamente por 1 hora UTC.
Uma hora seca ou uma lacuna termina o episódio.

## Instalação e execução

Extraia este pacote na raiz do repositório `rionowcastdf` (a pasta
`corrdiff_extreme_audit` permanece dentro da raiz). Use o ambiente Python do projeto.
As dependências são NumPy, pandas, Zarr 2 e matplotlib. Se já instaladas, não é
necessário reinstalar. Para um ambiente separado:

```bash
python -m venv .venv-audit14b
source .venv-audit14b/bin/activate
python -m pip install -r corrdiff_extreme_audit/requirements.txt
```

Execute a partir da raiz do repositório:

```bash
mkdir -p logs
nohup python -u corrdiff_extreme_audit/scripts/14b_audit_extreme_episodes.py \
  --dataset-dir datasets/corrdiff_2011_2024_restored \
  --radar-cache-dir /home/vsantos/repositories/rionowcastdf/data/radar_cache \
  --output-dir analysis_outputs_restored/14b_extreme_episodes \
  --threshold-dbz 45 \
  --batch-samples 512 \
  --pixel-km 2 \
  --focus-year 2024 \
  --top-episodes 10 \
  --historical-episodes 5 \
  > logs/14b_extreme_episodes.log 2>&1 &
echo $!
```

```bash
tail -f logs/14b_extreme_episodes.log
```

Ajuste o caminho do cache se estiver em outro diretório. O cache deve conter
`YYYYMMDD_HH_MM.npy`, já remapeado para a grade do builder. O script não lê PNGs
originais nem altera/preenche caches. O diretório de saída deve ser novo ou vazio;
para repetir, use outro nome. Não há opção de apagar outputs automaticamente.

## Contrato de entrada e custo

Lê `train.zarr/{target,mask,timestamps,patch_row,patch_col}`. Target e mask:
[N,1,H,W]. Timestamps: Unix segundos ordenados. Coordenadas: canto superior
esquerdo do patch em índices da grade (como no builder). Target: float32
`log1p(clip(dBZ, min=0))`; máscara binária. Não lê o tensor ERA5.
Faz uma varredura exata, sequencial em lotes, de todos os patches, sem amostragem.
A memória mantém um lote e um timestamp, mais a tabela agregada. O custo principal
é ler/descomprimir target e mask de aproximadamente um milhão de patches.
Conflitos em pixels sobrepostos, coordenadas duplicadas, timestamps desordenados,
valores inválidos ou máscaras não binárias interrompem a auditoria explicitamente.

## Definições

- `patch_*`: conta pixels com repetição causada pela sobreposição; descreve a
  distribuição de treinamento.
- `unique_*`: deduplica pixels pelas coordenadas dos patches, somente na região
  coberta. Não reconstrói regiões fora dos patches nem uma climatologia completa
  do domínio do radar.
- Limiares 30/40/45: comparação direta no domínio armazenado,
  `target >= float32(log1p(limiar))`, sem arredondamento de `expm1` na decisão.
  Esta convenção deve ser conferida antes de comparar contagens com fases que
  aplicam limiares após transformação inversa.
- `pixel_hours`: soma de pixels únicos com evento nos timestamps do episódio.
  `area_km2_hours` multiplica pela área nominal do pixel. Não é área máxima nem
  número de tempestades independentes.
- Hash BLAKE2b-128: target, máscara e coordenadas ordenadas por posição. A
  repetição só é marcada em pares UTC de exatamente uma hora; inclui toda a
  estrutura armazenada, não apenas a máscara de extremos. Não prova artefato.
- Frações anuais: média das frações por timestamp e razão entre somas de contagens
  e denominadores (`pooled_pixel_fraction`), com máscaras válidas respeitadas.
  Frequência de evento usa timestamps disponíveis; não imputa horas faltantes.

## Saídas

| Arquivo | Conteúdo |
|---|---|
| `timestamp_metrics.csv` | Contagens e frações 30/40/45, máximo, hashes, geometria, episódio e índices Zarr |
| `episodes_ranked.csv` | Ranking global por pixel-horas únicos, duração, pico e contribuição |
| `annual_sensitivity.csv` | Estatísticas anuais completas e sem top 1/5/10 episódios |
| `identical_extreme_transitions.csv` | Segunda hora de cada par extremo idêntico e consecutivo |
| `cache_checks.csv` | Comparação exata target/máscara com cache nos frames selecionados |
| `selected_patch_positions.csv` | Contagens por posição de patch nos frames selecionados |
| `episode_*.png` | Sequência pico ±2h: região dos patches e grade completa do cache |
| `timeline.png` | Série temporal da fração única com evento |
| `summary.json` | Parâmetros, versão, SHA256 do script e limites da análise |

Mapas incluem os maiores episódios globais, os maiores com pico no ano de foco e
os maiores com pico antes desse ano. Quantidades configuráveis; a união pode ter
menos elementos que a soma. Horas faltantes são mostradas explicitamente.
As coordenadas dos mapas são linha/coluna da grade; não se presume orientação
geográfica. Cores usam a mesma escala 0-55 dBZ; os valores não são truncados nas
estatísticas. A comparação do cache é somente na região coberta, para os frames
selecionados, mesmo quando o mapa inferior mostra a grade completa.
`status=ok` significa leitura/comparação concluída: verifique também as colunas de
mismatch, que devem ser zero. Missing/error não é resultado de igualdade.

Sensibilidade: remove episódios inteiros, inclusive quando cruzam anos. Os cenários
`touching_2024` selecionam episódios que possuem alguma hora em 2024, ordenados
pela contribuição total do episódio. Não confundir com ranking da contribuição
restrita a 2024. Todas as remoções mudam a população e o denominador; os resultados
não são estimativas corrigidas nem justificam descarte automático.

## Como decidir o próximo passo

1. Verifique completude em summary.json e quaisquer erros de contrato no log.
2. Consulte os primeiros episódios e os pares idênticos; inspecione os mapas.
3. Verifique cache_checks: diferenças indicam problema de correspondência ou dados
   e precisam de investigação. Igualdade confirma apenas fidelidade ao cache,
   não qualidade física do produto radar original.
4. Compare 2024 e anos anteriores nos cenários de sensibilidade, especialmente
   frequência versus fração média. Avalie padrões fixos e contribuição concentrada.
5. Documente episódios suspeitos e critérios de qualidade antes de qualquer
   exclusão. Só então consolide a Fase 15 e registre a exploração prévia de 2023/24.

Este diagnóstico não mede causas físicas, significância estatística, calibração do
radar ou drift causal; não há bootstrap nesta fase. A investigação de padrões fixos
é apoiada por hashes, mapas e posições, mas exige revisão visual. Não atualiza os
PDFs históricos nem implementa o split final.

## Testes

```bash
python -m unittest discover -s corrdiff_extreme_audit/tests -v
```

Cobrem sobreposição/deduplicação, conflitos, máscara, borda de limiar float32,
lacunas, horas secas, repetição exata e timestamps divididos entre lotes.

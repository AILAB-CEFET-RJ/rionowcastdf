# Passo 2 — perdas e fluxo de máscara

## Escopo e implementação

Arquivos novos: `adapter/reference_masked_losses.py`,
`adapter/reference_masked_flow.py` e `tests/test_reference_masked_losses.py`.
Requer PyTorch e o lote de dicionário do adaptador do passo 1. Não modifica
os arquivos do PhysicsNeMo, o dataset ou as configurações da fase 18.

A implementação foi conferida contra as assinaturas de RegressionLoss e
ResidualLoss de `physicsnemo/diffusion/metrics/legacy_losses.py` no commit
`c15bd23eb56216c32bd3e30180593e5685be9bc2`.
Regressão: `net(zeros_like(target), input)`.
Difusão: `net(noisy_residual, condition, sigma)`.
Isso não comprova execução das classes nativas: os testes usam convoluções
PyTorch pequenas com essas assinaturas. Não há treinamento C00 realizado.

## Contrato do lote

`input`: float32 BCHW já padronizado; `target`: float32 B1HW em código/50;
`mask`: bool B1HW, todos no mesmo dispositivo. A máscara não é inferida do
valor do alvo: zero válido participa da loss. Não há log1p, nova normalização,
clipping de predições, interpretação física de zero ou preenchimento observacional.

O leitor mantém sua validação estrita do armazenamento. As perdas ainda são
robustas a placeholders arbitrários nas posições inválidas: substituem esses
valores com `where` antes de operar. Não se usa `NaN * 0`.
Condicionantes, previsões e alvos válidos precisam ser finitos. Uma previsão
não finita, mesmo fora da máscara, encerra a operação: ela é um problema do
modelo e não um dado observacional ausente.

## Regressão

Para N pixels válidos no lote:

    L_reg = sum_valid((prediction - target)^2) / N

A redução é por pixel válido no lote, não média das médias dos patches. Logo,
patches com mais observações válidas têm maior peso. São contagens de pixels de
patches: sobreposições não são desduplicadas por esta loss. Não chamar esse valor
de erro sobre pixels espaciais únicos. Avaliação em campos completos será separada.

O modelo recebe somente zeros na entrada auxiliar e os canais ERA5. Nem alvo
nem máscara observacional entram como condicionantes da regressão.

## Difusão residual EDM

A regressão precisa estar em `eval()` e `requires_grad_(False)`. É executada
sob `no_grad()`. Não há atualização, backpropagation ou cache de médias antigas.

    mean = regression(zeros, input)
    residual = where(mask, target - mean, 0)
    sigma = exp(p_mean + p_std * normal_B111)
    latent = residual + sigma * epsilon
    weight = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
    L_diff = sum_valid(weight * (denoised - residual)^2) / N

Defaults explícitos: p_mean=-1.2, p_std=1.2, sigma_data=0.5. São parâmetros
configuráveis, não estimativas já calibradas para este dataset. O futuro executor
deve usar o mesmo sigma_data no precondicionador e registrar todos os valores.

`epsilon` é normal padrão sobre TODA a grade. Nas posições inválidas há somente
ruído, sem informação do alvo desconhecido. A máscara é usada na preparação do
residual e na redução; não é adicionada como canal do modelo. Com
`hr_mean_conditioning=True`, a média prevista é concatenada ANTES dos canais ERA5,
sem mascaramento: é uma previsão a partir de ERA5, disponível também na geração.
A configuração do modelo deve prever esse canal adicional.

Essa é uma política explícita de treinamento com observações parciais, não uma
prova de que a distribuição nas regiões não supervisionadas está correta. O ruído
inválido pode influenciar vizinhos por convoluções; o placeholder do alvo não pode.
Não se exige invariância à mudança desse ruído, à máscara, aos condicionantes ou
aos valores válidos. A validação de geração completa ainda é necessária no passo 6.

Para testes/validação reprodutíveis, fornecer `generator` ou `sigma` e `noise`
explícitos. `noise` significa epsilon unitário, não ruído já multiplicado por sigma.
O gerador deve corresponder ao dispositivo. O executor deve reiniciar/separar o
fluxo aleatório de validação; repetir a chamada avançando o mesmo gerador produz
novas amostras. A API não oferece augmentations ou patching adicional: os patches
32x32 já vêm do dataset. Futuras transformações precisam preservar alinhamento de
entrada, alvo, máscara e coordenadas.

## Lotes vazios e atualização

`LossResult` contém numerador, número de pixels válidos e número de amostras sem
suporte. `loss` é None quando N=0. Nesse caso, as funções retornam antes do forward.
`train_step` limpa gradientes e pula backward e optimizer.step. Assim Adam/AdamW
não avança estado nem aplica weight decay em um lote sem observação.
Em lote misto, amostras vazias permanecem no forward, mas não entram no numerador
ou denominador. Camadas que misturam estatísticas entre amostras podem afetar as
outras; não é prometida invariância à inclusão de novas amostras vazias.
O futuro scheduler deve avançar conforme a política registrada de updates efetivos.

`train_step` exige gradientes presentes e finitos e verifica os parâmetros após
a atualização. Não corrige NaN/Inf com nan_to_num. Se detectar um erro após step,
o estado corrente deve ser descartado/restaurado de checkpoint pelo executor;
não há rollback automático. O helper retorna loss, suporte e indicador updated.

`validation_pass` usa eval/no_grad, restaura o modo anterior e acumula numeradores
e contagens. Não calcula média simples das losses por lote. Validação inteira
sem suporte lança erro, em vez de reportar zero enganoso.

Escopo inicial: um processo/GPU, FP32. O helper rejeita DDP com mais de um processo.
Não implementa AMP nem acumulação de gradientes. Esses recursos exigem redução
pelo suporte global e sincronização dos lotes vazios; não podem ser acrescentados
apenas dividindo cada loss local pelo número de GPUs.

## Exemplo de integração

```python
from adapter.reference_masked_losses import regression_loss, MaskedResidualEDMLoss
from adapter.reference_masked_flow import move_batch, train_step

batch = move_batch(batch, device)
record = train_step(regression_model, optimizer, regression_loss, batch)

# Depois de carregar o checkpoint de regressão apropriado:
regression_model.eval().requires_grad_(False)
edm = MaskedResidualEDMLoss(sigma_data=0.5, hr_mean_conditioning=False)
def objective(denoiser, batch):
    return edm(denoiser, regression_model, batch, generator=train_generator)
record = train_step(diffusion_model, diffusion_optimizer, objective, batch)
```

Modelos, optimizers, device e gerador são fornecidos pelo futuro executor. Este
exemplo não é um comando de treinamento pronto nem um sampler de geração.

## Aplicar e testar

Extrair o ZIP fora do repositório. Na raiz de rionowcastdf:

```bash
git apply --check /caminho/masked_losses_step2.patch
git apply /caminho/masked_losses_step2.patch
python -m unittest discover -s tests -p test_reference_masked_losses.py -v
```

Não aplique patch e depois copie os mesmos arquivos novamente. O patch só adiciona
arquivos e não substitui o adaptador do passo 1.

## Evidência e limites

14 testes passaram localmente em CPU, PyTorch 2.15.0+cpu/Python 3.12. Cobrem:
redução manual/gradiente nulo nas saídas inválidas; equivalência a MSE quando toda
máscara é válida; invariância de regressão; invariância de entrada/loss/gradientes
da difusão para ambas as opções de condicionamento; sensibilidade a alvo válido;
AdamW sem atualização em lote vazio; ausência de forward na difusão vazia;
contagem de suporte misto; agregação correta da validação; erro para validação
vazia e restauração de modo; rejeição de não finitos; regressão congelada e sigma
positivo; reprodutibilidade do RNG; equivalência à fórmula EDM sem máscara.

Não valida GPU/PhysicsNeMo nativo, recarga de checkpoint, sampler completo,
convergência ou qualidade meteorológica. Não altera masked_loss_verified nem
marca a fase 18 como aprovada. Próximos passos: ambiente reproduzível, executor,
teste curto C00/42 em FP32 e verificação integrada de geração/checkpoints.

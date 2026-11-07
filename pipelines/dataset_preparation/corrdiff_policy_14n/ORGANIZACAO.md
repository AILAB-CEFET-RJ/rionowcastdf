# Organização do acervo de código e resultados

O objetivo é tornar clara a linha metodológica atual, preservando a comprovação
das correções. Não é necessário apagar investigações para deixar de usá-las.

## Linha ativa

| Componente | Papel | Manter |
|---|---|---|
| 14j | Fonte intermediária preservada, horário e geometria corrigidos | Código, contrato e resumo no Git; cache no servidor |
| 14k inventário | Disponibilidade dos 12 canais ERA5 | Código, configuração, resumos e cobertura no Git; SQLite no servidor |
| 14k pairing v2 | Pareamento completo e splits bloqueados | Código e relatórios pequenos; snapshots e manifestos completos no servidor |
| 14m | Inventário integral de PNGs de todos os minutos | Código, configurações, contagens e exemplos selecionados; banco e exportação completa no servidor |
| 14n | Política confirmada, início do acervo, novos alvos/candidatos | Esta pasta de código e relatórios pequenos; alvos e checkpoint no servidor |
| 14n smoke | Teste técnico da nova interpretação | Resumo e log pequeno; tensores/modelo diagnóstico fora do Git |

A dependência do 14n é 14j + 14k concluídos. O 14m é independente, somente leitura
sobre os PNGs. O 14n smoke depende de 14n + caminhos dos dados ERA5 do 14k.
Não renomear essas pastas durante a execução: há referências absolutas e hashes.

## Histórico importante, fora da linha ativa de treinamento

- 14b/14c: descoberta de extremos suspeitos e origem dos artefatos.
- 14g/14h/14i: investigação de semântica, geometria e fuso; referência de como a
  correção foi identificada. Preservar conclusões/configuração/exemplos.
- 14l: smoke anterior com máscara limitada à paleta. Serve de comparação; seus
  2,58% de suporte não são a disponibilidade final do dataset.
- `14k_pairing_v1` incompleto/abandonado: não é a base ativa; preservar metadados
  até conferir o que foi concluído no servidor.
- Dataset e auditorias anteriores à correção geométrica/temporal: marcar como
  históricos e não usar os números antigos como descrição do dataset final.

Recomenda-se um índice na raiz do repositório apontando para esta tabela e para
o diretório ativo. A movimentação para uma pasta histórica pode ser feita depois
que os processos terminarem e as dependências forem conferidas. Nesta entrega
nenhum script move ou exclui automaticamente essas pastas.

## Versionamento e recuperação de espaço

1. Execute `repository_report.py --apply-gitignore`.
2. Examine `tracked_artifacts_review.csv` e `git_status.txt`.
3. Se necessário, remova do ÍNDICE somente os caminhos grandes identificados,
   usando `git rm --cached -- CAMINHO` para arquivo ou `git rm -r --cached -- PASTA`.
   Esses comandos preservam o conteúdo no disco; não foram executados pelo pacote.
4. Revise `git diff --cached --stat` antes de commit/push.
5. Se arquivos grandes já estão em commits locais rejeitados pelo GitHub, ignorar
   e desversionar no último commit não basta. Planeje a correção do histórico com
   base no estado real do Git, sem force push ou reset indiscriminado.
6. Só excluir cópias antigas do disco depois de assegurar backup, ausência de
   referências nos manifests e término dos processos. O espaço livre informado
   anteriormente permite priorizar rastreabilidade em vez de apagar agora.

O relatório de organização examina o índice e a árvore atual; não percorre todos
os blobs do histórico. Ele não promete resolver sozinho a rejeição anterior do push.

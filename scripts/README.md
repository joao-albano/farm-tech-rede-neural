# 📂 Diretório SCRIPTS

Este diretório contém scripts auxiliares e utilitários do projeto FarmTech YOLO.

## 📁 Arquivos:

### 🔧 Scripts Utilitários:
- `data_preprocessing.py` - Pré-processamento do dataset (quando disponível)
- `model_converter.py` - Conversão entre formatos de modelo (quando disponível)
- `batch_inference.py` - Inferência em lote (quando disponível)
- `metrics_calculator.py` - Cálculo de métricas avançadas (quando disponível)

### 📊 Scripts de Análise:
- `dataset_analyzer.py` - Análise estatística do dataset (quando disponível)
- `performance_benchmark.py` - Benchmark de performance (quando disponível)
- `visualization_generator.py` - Geração de visualizações (quando disponível)

### 🚀 Scripts de Deploy:
- `model_optimizer.py` - Otimização de modelo para produção (quando disponível)
- `docker_setup.py` - Configuração Docker (quando disponível)
- `api_server.py` - Servidor API para inferência (quando disponível)

## 🎯 Funcionalidades:

### 📈 Análise de Dados:
- Estatísticas detalhadas do dataset
- Distribuição de classes e anotações
- Análise de qualidade das imagens
- Detecção de outliers e problemas

### 🔄 Processamento:
- Redimensionamento automático de imagens
- Normalização e augmentação de dados
- Conversão entre formatos (YOLO, COCO, Pascal VOC)
- Limpeza e validação de anotações

### 📊 Métricas e Avaliação:
- Cálculo de mAP (mean Average Precision)
- Análise de curvas Precision-Recall
- Benchmark de velocidade de inferência
- Comparação entre modelos

### 🚀 Deploy e Produção:
- Otimização de modelos (quantização, pruning)
- Conversão para formatos de produção (ONNX, TensorRT)
- Configuração de servidores de inferência
- Monitoramento de performance

## 🛠️ Como Usar:

Os scripts são projetados para serem executados independentemente ou integrados ao pipeline principal do projeto. Cada script possui documentação interna e parâmetros configuráveis para máxima flexibilidade.
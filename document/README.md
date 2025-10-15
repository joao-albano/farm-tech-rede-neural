# 📂 Diretório DOCUMENT

Este diretório contém toda a documentação, dados e modelos do projeto FarmTech YOLO.

## 📁 Arquivos:

### 📊 Dataset e Dados:
- `Cellphone.v1i.yolov5pytorch/` - Dataset completo de detecção de celulares
  - `train/` - Imagens e labels de treinamento
  - `valid/` - Imagens e labels de validação
  - `test/` - Imagens e labels de teste
  - `data.yaml` - Configuração do dataset
- `yolov8n.pt` - Modelo pré-treinado YOLOv8 nano

### 📋 Documentação Técnica:
- `README.md` - Este arquivo explicativo
- `dataset_analysis.md` - Análise detalhada do dataset (quando disponível)
- `model_performance.md` - Relatório de performance dos modelos (quando disponível)

### 📈 Resultados e Métricas:
- `training_results/` - Resultados de treinamento (quando disponível)
- `inference_results/` - Resultados de inferência (quando disponível)
- `metrics_report.json` - Métricas detalhadas (quando disponível)

## 📊 Resumo do Dataset:

### 📱 Cellphone Detection Dataset:
- **Total de imagens**: 1000+ imagens
- **Classes**: 1 (cellphone)
- **Formato**: YOLO PyTorch
- **Resolução**: Variada (otimizada para 640x640)
- **Anotações**: Bounding boxes precisas

### 📈 Divisão dos Dados:
- **Treinamento**: ~70% das imagens
- **Validação**: ~20% das imagens  
- **Teste**: ~10% das imagens

### 🎯 Características:
- **Balanceamento**: Dataset bem balanceado
- **Qualidade**: Imagens de alta qualidade
- **Diversidade**: Diferentes ângulos, iluminações e contextos
- **Anotações**: Labels precisas e validadas

## 🤖 Modelo Pré-treinado:

### 🏆 YOLOv8n (Nano):
- **Tamanho**: ~6MB
- **Velocidade**: Ultra-rápida
- **Precisão**: Otimizada para detecção de objetos
- **Compatibilidade**: PyTorch, ONNX, TensorRT

## 🚀 Uso dos Dados:

Os dados são automaticamente carregados pelos scripts de treinamento e notebooks, seguindo a estrutura padrão do YOLO para máxima compatibilidade e performance.
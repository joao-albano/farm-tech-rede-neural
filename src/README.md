# 📂 Diretório SRC

Este diretório contém todo o código-fonte do projeto FarmTech YOLO.

## 📁 Arquivos:

### 🐍 Scripts Python:
- `generate_notebook.py` - Script para geração automática do notebook Jupyter
- `farmtech_yolo_complete.py` - Script completo de treinamento YOLO (quando disponível)
- `farmtech_yolo_inference.py` - Script de inferência e detecção (quando disponível)
- `cnn_custom_model.py` - Implementação de CNN customizada (quando disponível)

### 📓 Notebooks:
- `FarmTech_YOLO.ipynb` - Notebook Jupyter principal com análise completa

### 🔧 Módulos Auxiliares:
- `data_loader.py` - Carregamento e pré-processamento de dados (quando disponível)
- `model_trainer.py` - Classes para treinamento de modelos (quando disponível)
- `utils.py` - Funções utilitárias (quando disponível)

## 🚀 Como Executar:

### Pré-requisitos:
```bash
pip install ultralytics torch torchvision matplotlib seaborn pandas numpy opencv-python pillow
```

### Executar Notebook Principal:
```bash
jupyter notebook FarmTech_YOLO.ipynb
```

### Gerar Notebook Automaticamente:
```bash
python generate_notebook.py
```

### Executar Treinamento Completo:
```bash
python farmtech_yolo_complete.py
```

### Executar Inferência:
```bash
python farmtech_yolo_inference.py --input path/to/image.jpg
```

## 🎯 Principais Características:

### 📊 Análise de Dados:
- **Exploração completa** do dataset de celulares
- **Visualizações profissionais** automatizadas
- **Estatísticas detalhadas** de distribuição
- **Análise de qualidade** das anotações

### 🤖 Modelos Implementados:
- **YOLOv8n** - Modelo principal para detecção
- **YOLOv8s/m/l** - Variações para diferentes necessidades
- **CNN Customizada** - Rede neural personalizada
- **Transfer Learning** - Aproveitamento de modelos pré-treinados

### 🏋️ Treinamento:
- **Otimização automática** de hiperparâmetros
- **Data Augmentation** avançada
- **Early Stopping** para evitar overfitting
- **Métricas em tempo real** durante treinamento

### 📈 Avaliação:
- **mAP (mean Average Precision)** detalhado
- **Curvas Precision-Recall** 
- **Matriz de Confusão** para análise de erros
- **Benchmark de velocidade** de inferência

## 🔧 Configurações:

### 🎛️ Parâmetros Principais:
- **Epochs**: 30-60 (configurável)
- **Batch Size**: Adaptável ao hardware disponível
- **Learning Rate**: Otimizado automaticamente
- **Input Size**: 640x640 pixels
- **Confidence Threshold**: 0.25
- **IoU Threshold**: 0.45

### 📊 Dataset:
- **Formato**: YOLO PyTorch
- **Classes**: 1 (cellphone)
- **Divisão**: train/valid/test automática
- **Augmentação**: Rotação, flip, escala, cor

## 💡 Funcionalidades Avançadas:

- ✨ **Geração automática** de notebooks Jupyter
- 📊 **Visualizações interativas** com matplotlib/seaborn
- 🔄 **Pipeline completo** de ML (dados → modelo → avaliação)
- 🚀 **Otimização para produção** (ONNX, TensorRT)
- 📱 **Detecção em tempo real** via webcam
- 🎯 **Análise de performance** detalhada
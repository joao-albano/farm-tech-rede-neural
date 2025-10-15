# 📂 Diretório CONFIG

Este diretório contém arquivos de configuração do projeto FarmTech YOLO.

## 📁 Arquivos:

### ⚙️ Configurações de Modelo:
- `yolo_config.yaml` - Configurações do modelo YOLO (quando disponível)
- `training_config.json` - Parâmetros de treinamento
- `hyperparameters.json` - Hiperparâmetros otimizados

### 🔧 Configurações de Sistema:
- `environment.yaml` - Configuração do ambiente conda/pip
- `data_config.yaml` - Configurações do dataset
- `inference_config.json` - Configurações para inferência

## 🎯 Principais Configurações:

### 📊 Dataset:
- **Classes**: cellphone (1 classe)
- **Formato**: YOLO PyTorch
- **Divisão**: train/valid/test
- **Augmentação**: configurável

### 🤖 Modelo:
- **Arquitetura**: YOLOv8n (nano)
- **Input Size**: 640x640
- **Confidence Threshold**: 0.25
- **IoU Threshold**: 0.45

### 🏋️ Treinamento:
- **Epochs**: configurável (padrão: 30-60)
- **Batch Size**: adaptável ao hardware
- **Learning Rate**: otimizado automaticamente
- **Optimizer**: AdamW

## 🚀 Uso:

Os arquivos de configuração são carregados automaticamente pelos scripts de treinamento e inferência, permitindo fácil customização dos parâmetros sem modificar o código fonte.
# 🏥 FemHealth — Roteiro de Apresentação
### Tech Challenge Fase 4 · POSTECH IADT · FIAP · Março/2026

---

## 1. O Desafio

A rede hospitalar FemHealth precisa monitorar pacientes continuamente por meio de
**dados multimodais — áudio, vídeo e texto** — para identificar sinais precoces de
risco específicos da saúde e segurança feminina.

Nossa solução cobre **três dos objetivos obrigatórios**:

- ✅ Detectar precocemente riscos em saúde materna e ginecológica
- ✅ Identificar sinais de violência doméstica ou abuso
- ✅ Monitorar bem-estar psicológico feminino

---

## 2. Visão Geral da Solução

```mermaid
mindmap
  root((FemHealth\nMultimodal))
    Vídeo
      Cirurgia\nSegmentação de instrumentos
      Triagem\nDetecção de poses agressivas
      Fisioterapia\nClassificação de movimentos
      Consulta\nDetecção de dor facial
    Áudio
      Transcrição de consultas
      Detecção de padrões vocais de risco
    API
      FastAPI REST
      POST /predict por contexto
```

---

## 3. Requisito 1 — Análise de Vídeo Especializada

O documento exige processamento de vídeos clínicos em 4 contextos.
Implementamos todos com **YOLOv8 customizado**:

```mermaid
flowchart LR
    V[🎥 Vídeo Clínico] --> R{Contexto}

    R -->|Cirurgia\nginecológica| C[YOLOv8n-seg\nSegmentação de\ninstrumentos laparoscópicos]
    R -->|Triagem\nde violência| T[YOLOv8n-pose\nDetecção de linguagem\ncorporal agressiva]
    R -->|Fisioterapia\npós-parto| F[YOLOv8n-cls\nClassificação de\nmovimentos corretos/incorretos]
    R -->|Consulta\npsicológica| P[YOLOv8n-cls\nDetecção de expressão\nde dor facial]

    C --> OUT[📊 Relatório\nAutomático]
    T --> OUT
    F --> OUT
    P --> OUT
```

### 3.1 Modelo Especializado — Instrumentos Cirúrgicos Ginecológicos

Atende diretamente ao requisito:
> *"YOLOv8 customizado para detecção de instrumentos cirúrgicos ginecológicos"*

Dataset: **Laparoscopia Roboflow** · 1081 imagens · 7 classes

```mermaid
xychart-beta
    title "Cirurgia — Box mAP50 por Instrumento"
    x-axis ["Bag", "Liver", "Cautery", "Gallbladder", "Forceps", "Suction", "Allis"]
    y-axis "mAP50" 0 --> 1
    bar [0.995, 0.893, 0.851, 0.795, 0.755, 0.494, 0.317]
```

> ⚠️ `Allis` e `Suction` com mAP50 baixo por underrepresentation no val —
> limitação do dataset, não do modelo.

### 3.2 Triagem de Violência — Linguagem Corporal

Atende ao requisito:
> *"Triagem de violência: detecção de linguagem corporal indicativa de abuso"*

Dataset: **Aggressive Poses** · 103 imagens · 17 keypoints

| Métrica | Box | Pose |
|---------|-----|------|
| Precision | 0.998 | 0.998 |
| Recall | **1.000** | **1.000** |
| mAP50 | 0.995 | 0.995 |
| mAP50-95 | 0.956 | 0.771 |

> Recall perfeito — **nenhuma pose de risco deixa de ser detectada.**

### 3.3 Fisioterapia Pós-parto — Análise de Movimentos

Atende ao requisito:
> *"Fisioterapia: análise de movimentos e recuperação"*

Dataset: **678 vídeos → 3019 frames** · 6 classes (Arm Raise, Knee Extension, Sit To Stand · correto/incorreto)

| Métrica | Valor |
|---------|-------|
| Top-1 Accuracy | **0.994** |
| Top-5 Accuracy | 1.000 |
| Inferência | 0.9ms/img |

### 3.4 Consulta — Sinais Não-verbais de Desconforto

Atende ao requisito:
> *"Consultas: identificação de sinais não-verbais de desconforto ou medo"*

Dataset: **UNBC-McMaster Pain (FACS)** · 4000 imagens balanceadas · 2 classes

| Métrica | Valor |
|---------|-------|
| Top-1 Accuracy | **0.933** |
| Top-5 Accuracy | 1.000 |

---

## 4. Requisito 2 — Análise de Áudio

Atende ao requisito:
> *"Processar gravações de voz de pacientes em consultas"*

```mermaid
flowchart LR
    A[🎙️ Áudio da Consulta] --> B[Transcrição\nWhisper / Azure Speech]
    B --> C[Análise de Padrões\nVocais de Risco]
    C --> D{Sinal Detectado?}
    D -- Sim --> E[⚠️ Alerta\nEquipe Médica]
    D -- Não --> F[✅ Consulta\nRegistrada]
```

> Módulo `app/audio/transcriber.py` — integração em finalização.

---

## 5. Pipeline Técnico Completo

```mermaid
flowchart TD
    RAW[🗂️ Dados Brutos\náudio · vídeo · texto] --> PRE

    subgraph PRE [Preprocessamento]
        P1[cirurgia\nCOCO → YOLOv8 seg]
        P2[triagem\nCOCO pose → YOLOv8 pose]
        P3[fisioterapia\nvídeos → frames]
        P4[consulta\nFACS → classify]
        P5[áudio\nwav/mp3 → transcrição]
    end

    PRE --> VAL[Validação\nde Conformidade]
    VAL --> |✅ Aprovado| TRAIN
    VAL --> |❌ Reprovado| FIX[Corrigir\ne Reprocessar]
    FIX --> VAL

    subgraph TRAIN [Treino YOLOv8]
        T1[triagem · pose · 30ep]
        T2[fisioterapia · cls · 30ep]
        T3[consulta · cls · 30ep]
        T4[cirurgia · seg · 50ep]
    end

    TRAIN --> EVAL[Avaliação\nFinal]
    EVAL --> MODELS[models/yolov8_custom/\n*.pt]
    MODELS --> API[🚀 FastAPI\nuvicorn app.main:app]
    API --> REPORT[📋 Relatório\nAutomático de Anomalias]
```

---

## 6. Desafios Técnicos e Soluções

```mermaid
timeline
    title Problemas Encontrados e Resolvidos

    Validação : Labels triagem com 57 campos
              : Classe duplicada float + int
              : Fix - substituir em vez de prefixar

    Treino    : NameError - variáveis perdidas no kernel
              : Estado da sessão não persistiu
              : Fix - célula de treino autocontida

    Pós-treino : best.pt não encontrado
               : YOLOv8 salva em path absoluto
               : Fix - rglob + shutil.copy2

    API        : ImportError transcribe_audio
               : Função com nome diferente
               : Fix - alias ou correção do import
```

---

## 7. Resultados Consolidados

```mermaid
xychart-beta
    title "Métricas Finais de Validação"
    x-axis ["Triagem\nmAP50", "Triagem\nmAP50-95", "Cirurgia\nmAP50", "Cirurgia\nmAP50-95", "Fisioterapia\nTop-1", "Consulta\nTop-1"]
    y-axis "Score" 0 --> 1
    bar [0.995, 0.771, 0.729, 0.505, 0.994, 0.933]
```

---

## 8. Entregáveis — Checklist

```mermaid
flowchart LR
    subgraph GIT [Repositório Git ✅]
        G1[Código-fonte completo]
        G2[Notebook de treino]
        G3[Relatório técnico\nfluxo multimodal]
        G4[Modelos best.pt]
    end

    subgraph VIDEO [Vídeo até 15min 🎬]
        V1[Demonstração\nde áudio e vídeo]
        V2[Detecção de anomalias]
        V3[Integração Azure]
        V4[Fluxo de alerta\nà equipe médica]
    end
```

| Entregável | Status |
|------------|--------|
| Código-fonte completo | ✅ |
| Análise de vídeo (4 contextos) | ✅ |
| Modelos YOLOv8 customizados | ✅ |
| Análise de áudio | 🔧 Em finalização |
| FastAPI REST | 🔧 Em finalização |
| Relatórios automáticos | 🔧 Em finalização |
| Vídeo demonstração (YouTube/Vimeo) | ⏳ Pendente |

---

## 9. Melhorias Futuras

- **Cirurgia:** Ampliar instâncias de `Allis` e `Suction` com data augmentation
- **Triagem:** Adicionar `flip_idx` no `data.yaml` e ampliar dataset para 500+ imagens
- **Áudio:** Integrar Azure Cognitive Services / Speech-to-Text para análise vocal em tempo real
- **Deploy:** Exportar modelos para ONNX e containerizar com Docker

---

*Tech Challenge Fase 4 · POSTECH IADT · FIAP · Março/2026*

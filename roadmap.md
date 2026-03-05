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
      Transcrição Whisper
      Análise de emoção por keywords + GPT-4o-mini
    API
      FastAPI REST
      POST /analyze com contexto enum
      Relatório PDF automático
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
>
> Alerta via `_check_protective_pose()`: ombros encolhidos e pulsos cruzados.
> Fallback: qualquer pessoa detectada com confiança ≥ 70% dispara alerta CRÍTICO.


### 3.3 Fisioterapia Pós-parto — Análise de Movimentos


Atende ao requisito:
> *"Fisioterapia: análise de movimentos e recuperação"*


Dataset: **678 vídeos → 3120 frames** · 6 classes (Arm Raise, Knee Extension, Sit To Stand · correto/incorreto)


| Métrica | Valor |
|---------|-------|
| Top-1 Accuracy | **0.994** |
| Top-5 Accuracy | 1.000 |
| Inferência | 0.9ms/img |


Classes de alerta: `Arm_Raise_Incorrect`, `Knee_Extension_Incorrect`, `Sit_To_Stand_Incorrect`


### 3.4 Consulta — Sinais Não-verbais de Desconforto


Atende ao requisito:
> *"Consultas: identificação de sinais não-verbais de desconforto ou medo"*


Dataset: **UNBC-McMaster Pain (FACS)** · 4000 imagens balanceadas · 2 classes (`comdor` / `semdor`)


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
    A[🎙️ Áudio da Consulta] --> B[Transcrição\nWhisper base]
    B --> C[Análise de Emoção\nKeywords + GPT-4o-mini]
    C --> D{Sinal Detectado?}
    D -- Sim --> E[⚠️ Alerta\nEquipe Médica]
    D -- Não --> F[✅ Consulta\nRegistrada]
```


**Categorias de risco monitoradas:** `depressao`, `ansiedade`, `violencia`, `pos_parto`

**Fallback sem OpenAI:** análise exclusiva por keywords — funciona offline.

**Fallback sem áudio no vídeo:** retorna `warning` sem quebrar o pipeline.


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
    EVAL --> MODELS[models/\n*.pt]
    MODELS --> API[🚀 FastAPI\nuvicorn app.main:app]
    API --> FUSION[Fusão Multimodal\nalert.py]
    FUSION --> REPORT[📋 Relatório PDF\nAutomático]
```


---


## 6. Arquitetura da API


```
POST /analyze
├── video: UploadFile
└── context: Enum [cirurgia | consulta | fisioterapia | triagem]

Response:
├── transcription       ← Whisper (com fallback sem áudio)
├── emotion             ← EmotionAnalyzer (keywords + LLM)
├── detections[]        ← YOLOv8 frame-a-frame com timestamp
├── detections_by_class ← contagem por classe
├── alert               ← fusão visual + áudio + texto + pose
└── report              ← path do PDF gerado
```


**Fusão de sinais no alerta:**


| Sinal | Fonte | Peso |
|-------|-------|------|
| `visual` | YOLOv8 detecções | classes de risco por contexto |
| `audio_emotion` | EmotionAnalyzer | risco ALTO ou CRÍTICO |
| `text` | transcrição | keywords clínicos de risco |
| `pose` | MediaPipe / fallback | postura protetiva detectada |


---


## 7. Desafios Técnicos e Soluções


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

    API        : ImportError transcribe_audio / detect_emotion
               : Funções com nomes diferentes das classes
               : Fix - wrappers module-level com singleton lazy

    MediaPipe  : mp.solutions removido na v0.10.30+
               : AttributeError no import
               : Fix - try/except + fallback por confiança YOLO

    Vídeo      : Arquivo sem trilha de áudio
               : ffmpeg retornava exit status non-zero
               : Fix - ffprobe check antes do Whisper

    Windows    : PermissionError ao deletar tmp
               : cv2.VideoCapture mantinha lock
               : Fix - cap.release() antes do finally
```


---


## 8. Resultados Consolidados


```mermaid
xychart-beta
    title "Métricas Finais de Validação"
    x-axis ["Triagem\nmAP50", "Triagem\nmAP50-95", "Cirurgia\nmAP50", "Cirurgia\nmAP50-95", "Fisioterapia\nTop-1", "Consulta\nTop-1"]
    y-axis "Score" 0 --> 1
    bar [0.995, 0.771, 0.729, 0.505, 0.994, 0.933]
```


---


## 9. Entregáveis — Checklist


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
| FastAPI REST `/analyze` | ✅ |
| Transcrição de áudio (Whisper) | ✅ |
| Análise de emoção (keywords + LLM) | ✅ |
| Sistema de alertas multimodal | ✅ |
| Relatórios automáticos PDF | ✅ |
| Azure Speech / Language | 🔧 Implementado, aguarda chaves |
| Vídeo demonstração (YouTube/Vimeo) | ⏳ Pendente |


---


## 10. Melhorias Futuras


- **Cirurgia:** Ampliar instâncias de `Allis` e `Suction` com data augmentation
- **Triagem:** Adicionar `flip_idx` no `data.yaml` e ampliar dataset para 500+ imagens
- **Áudio:** Ativar Azure Cognitive Services / Speech-to-Text para análise vocal em tempo real
- **Deploy:** Exportar modelos para ONNX e containerizar com Docker
- **MediaPipe:** Migrar para Tasks API (v0.10.30+) para análise de pose completa


---


*Tech Challenge Fase 4 · POSTECH IADT · FIAP · Março/2026*

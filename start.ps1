Write-Host "Iniciando FemHealth Multimodal AI..." -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python nao encontrado. Instale em https://python.org" -ForegroundColor Red
    exit 1
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "FFmpeg nao encontrado. Instalando..." -ForegroundColor Yellow
    winget install ffmpeg
    Write-Host "FFmpeg instalado! Reinicie o script." -ForegroundColor Green
    exit 0
}

if (-not (Test-Path "venv")) {
    Write-Host "Criando ambiente virtual..." -ForegroundColor Yellow
    python -m venv venv
}

Write-Host "Ativando ambiente virtual..." -ForegroundColor Yellow
. venv\Scripts\Activate.ps1

Write-Host "Instalando dependencias..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet

if (-not (Test-Path ".env")) {
    Write-Host "Criando .env a partir do .env.example..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "Configure suas chaves no arquivo .env!" -ForegroundColor Red
}

New-Item -ItemType Directory -Force -Path "data\samples" | Out-Null

Write-Host "Servidor iniciando em: http://localhost:8000/docs" -ForegroundColor Green
uvicorn app.main:app --reload

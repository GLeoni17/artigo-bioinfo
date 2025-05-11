@echo off
setlocal

set ENV_DIR=.venv
set REQ_FILE=requirements.txt

:: Verifica se o ambiente virtual existe
if not exist %ENV_DIR%\Scripts\activate.bat (
    echo Ambiente virtual não encontrado. Criando...
    python -m venv %ENV_DIR%

    if exist %REQ_FILE% (
        echo Instalando dependências de %REQ_FILE%...
        call %ENV_DIR%\Scripts\activate.bat
        python -m pip install --upgrade pip
        pip install -r %REQ_FILE%
    ) else (
        echo ERRO: Arquivo %REQ_FILE% não encontrado!
        exit /b 1
    )
) else (
    echo Ambiente virtual encontrado. Ativando...
    call %ENV_DIR%\Scripts\activate.bat
)

:: Executa os scripts Python
echo Etapa 1: Baixando sequência básica e variações...
python get_dataset.py

echo Etapa 2: Alinhando variantes com referência usando MAFFT...
python fragment_mafft.py

echo Etapa 3: Padronizando nomes de países nos metadados...
python padroniza_paises.py

echo Etapa 4: Identificando mutações nas variantes...
python mutation_identification.py

echo Etapa 5: Extraindo features a partir de metadados e mutações...
python extrair_features.py

echo Etapa 6: Aplicando e avaliando modelos de IA...
python aplicar_modelos.py

echo Pipeline finalizado com sucesso.
endlocal

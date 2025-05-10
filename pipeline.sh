#!/bin/bash

ENV_DIR=".venv"
REQ_FILE="requirements.txt"

# Verifica se o ambiente virtual existe
if [ ! -d "$ENV_DIR" ]; then
    echo "Ambiente virtual não encontrado. Criando..."
    python3 -m venv "$ENV_DIR"

    echo "Ativando ambiente virtual..."
    source "$ENV_DIR/bin/activate"

    if [ -f "$REQ_FILE" ]; then
        echo "Instalando dependências de $REQ_FILE..."
        pip install --upgrade pip
        pip install -r "$REQ_FILE"
    else
        echo "Arquivo $REQ_FILE não encontrado! Abortando."
        exit 1
    fi
else
    echo "Ambiente virtual encontrado. Ativando..."
    source "$ENV_DIR/bin/activate"
fi

# Etapas do pipeline
echo "Etapa 1: Baixando sequência básica e variações..."
python get_dataset.py

echo "Etapa 2: Alinhando variantes com referência usando MAFFT..."
python fragment_mafft.py

echo "Etapa 3: Buscando metadados de linhagem..."
python lineage_metadata.py

echo "Etapa 4: Padronizando nomes de países nos metadados..."
python padroniza_paises.py

echo "Etapa 5: Identificando mutações nas variantes..."
python mutation_indentification.py

echo "Etapa 6: Extraindo features a partir de metadados e mutações..."
python extrair_features.py

echo "Etapa 7: Aplicando e avaliando modelos de IA..."
python aplicar_modelos.py

echo "Pipeline finalizado com sucesso."
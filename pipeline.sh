#!/bin/bash

# Ambiente virtual para execução ---------------------------------------------------------------------------------------
ENV_NAME="bioinfo"
ENV_FILE="environment.yml"

if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "Ambiente Conda '$ENV_NAME' não encontrado. Criando a partir de $ENV_FILE..."

    if [ -f "$ENV_FILE" ]; then
        conda env create -f "$ENV_FILE"
    else
        echo "Arquivo $ENV_FILE não encontrado! Abortando."
        exit 1
    fi
else
    echo "Ambiente Conda '$ENV_NAME' encontrado. Atualizando pacotes..."
    conda env update --file $ENV_FILE --prune
fi

echo "Ativando ambiente Conda..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Etapas do pipeline ---------------------------------------------------------------------------------------------------
printf "\nEtapa 1: Baixando sequência básica e variações..."
python get_dataset.py

printf "\nEtapa 2: Alinhando variantes com referência usando MAFFT..."
python fragment_mafft.py

printf "\nEtapa 3: Padronizando nomes de países nos metadados..."
python padroniza_paises.py

printf "\nEtapa 4: Identificando mutações nas variantes..."
python mutation_identification.py

printf "\nEtapa 5: Extraindo features a partir de metadados e mutações..."
python extrair_features.py

printf "\nEtapa 6: Aplicando e avaliando modelos de IA..."
python aplicar_modelos.py

printf "\nPipeline finalizado com sucesso."

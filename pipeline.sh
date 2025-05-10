#!/bin/bash

# Ativar o ambiente virtual se necessário
# source venv/bin/activate

echo "Etapa 1: Baixando sequência básica e variações..."
python3 get_dataset.py

echo "Etapa 2: Alinhando variantes com referência usando MAFFT..."
python3 fragment_mafft.py

echo "Etapa 3: Buscando metadados de linhagem..."
python3 lineage_metadata.py

echo "Etapa 4: Padronizando nomes de países nos metadados..."
python3 padroniza_paises.py

echo "Etapa 5: Identificando mutações nas variantes..."
python3 mutation_indentification.py

echo "Etapa 6: Extraindo features a partir de metadados e mutações..."
python3 extrair_features.py

echo "Etapa 7: Aplicando e avaliando modelos de IA..."
python3 aplicar_modelos.py

echo "Pipeline finalizado com sucesso."

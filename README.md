# Previsão da Origem Geográfica de Vírus Através de Aprendizado de Máquina Aplicado a Dados Genômicos

Para executar todo o pipeline em Linux, executar ```pipeline.sh```

## Pipeline

* Criação de um ambiente Conda com as bibliotecas necessárias

* ```get_dataset.py```: baixa a sequência básica e as variantes

* ```fragment_mafft.py```: faz o alinhamento de cada uma das variantes com a referência, com MAFFT
 
* ```mutation_identification.py```: compara variantes com a referência e salva as mutações

* ```extrair_features.py```: extrai features dos hotspots de mutações com base nos metadados e informações de mutações

* ```aplicar_modelos.py```: aplica e avalia modelos de IA sobre os dados

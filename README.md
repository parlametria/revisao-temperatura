# Revisão do cálculo da temperatura

Este projeto visa reformular a forma de cálculo da temperatura de proposições no congresso nacional.
A temperatura é uma medida de quão ativa está a proposição, de quanto os parlamentares estão agindo
sobre ela. 

## 1. Descrição do projeto

O objetivo deste projeto é fazer aprimoramentos no processo de cálculo da temperatura. Dentre esses
aprimoramentos, fizemos aqui:

* uma contrução de funções em python para ajudar a analisar a qualidade das expressões regulares
(regexes) utilizadas para detecção de ações legislativas a partir do despacho da tramitação de
proposições (veja o notebook `analises/refinamento_dos_regex_de_deteccao_de_eventos_camara.ipynb`);

* um aprimoramento das expressões regulares utilizadas pelo leggoR e pelo rcongresso, já incorporado
em algum branch desses códigos (resultados em `resultados/new_environment_camara_2021-03-06.json` e
`resultados/new_rcongresso_camara_2021-03-06.json`, criados a partir do notebook acima);
 
* a construção de um modelo de _machine learning_ de cálculo da temperatura. Até o momento, a temperatura
era uma fórmula definida manualmente com base na experiência dos pesquisadores envolvidos no projeto. A
proposta neste projeto é transformar essa fórmula em um modelo cuja estrutura e parâmetros sejam baseados
diretamente em dados e que possa ter sua performance avaliada com dados de maneira mais objetiva.
Para isso, definimos um alvo a ser predito pelo modelo, e esse alvo será chamado de temperatura.
A proposta de alvo adotada é a probabilidade de ocorrência de ações legislativas importantes num futuro
próximo (e.g. em duas semanas). Esse aprimoramento foi feito no notebook
`analises/revisao_da_temperatura.ipynb`.

## 2. Estrutura do projeto

    .
    ├── README.md         <- Este documento
    ├── requirements.txt  <- Pacotes python necessários, junto com suas versões
    ├── analises          <- Notebooks do jupyter em python
    ├── codes             <- Pedaços do código do `leggoR` e `rcongresso` utilizados
    ├── dados             <- Dados utilizados pelos notebooks
    └── resultados        <- Produtos das análises (relatórios e gráficos, entre outros)

## 3. Autores

* **Henrique S. Xavier** - [@hsxavier](https://github.com/hsxavier)


## Osteosarcoma LLM 

[![Current devel version: 0.3.0](https://img.shields.io/badge/devel%20version-0.3.0-blue.svg)](https://github.com/randef1ned/labyrinth)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/09b138b2fa9242229f081cd180f6fc91)](https://app.codacy.com/gh/randef1ned/labyrinth/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

## Introduction

This is a single-disease osteosarcoma knowledge graph-based Q&A system incorporating ChatGLM3-6b.

By manually annotating and integrating more than `20,000` articles on osteosarcoma from Pumbed, we established the first knowledge graph for osteosarcoma. We also applied this graph and chatGLM3 to develop a question-answering model. It can overcome the "illusion" of the large model and help researchers find the knowledge they need more easily and accurately.

![A simple schema of the labyrinth](figures/Figure1.jpg)

## A notice on operating system compatibility
- **Python**: We developed this method using python version 3.10.x.
- **CUDA**:  We developed this method using python version 12.0
- **neo4j**:  We developed this method using python version 5.15.0.

## Before installation
We used a computer equipped with an RTX 3090 graphics card for the experiment. For stability and improved computing efficiency, we recommend using a computer with RTX 3090 or above.
## usage
``` bash
# Firstly, run the neo4j database. (localhost:7474) 
  neo4j start
#Secondly, run the ChatGLM3-6b model. (localhost:8000) 
  python openai_api.py 
#Back-end Run 
  python main.py 
```



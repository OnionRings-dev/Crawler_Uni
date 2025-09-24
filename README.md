# 🤖 Crawler Semantico UniMi

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Semantic Search](https://img.shields.io/badge/AI-Semantic%20Search-purple.svg)](https://github.com)

Un sistema avanzato di web crawling e ricerca semantica che utilizza trasformatori neurali per l'analisi e visualizzazione di contenuti web universitari.

## 📋 Indice

- [Caratteristiche](#-caratteristiche)
- [Tecnologie](#️-tecnologie)
- [Installazione](#-installazione)
- [Utilizzo](#-utilizzo)
- [Risultati](#-risultati)
- [Visualizzazioni](#-visualizzazioni)
- [Architettura](#️-architettura)
- [Contributi](#-contributi)

## ✨ Caratteristiche

- 🌐 **Web Crawling Intelligente**: Estrazione automatica di contenuti da domini universitari
- 🧠 **Embeddings Semantici**: Utilizzo di modelli transformer multilingua per la vettorizzazione
- 🔍 **Ricerca Semantica**: Algoritmi di similarità coseno per trovare contenuti correlati
- 📊 **Visualizzazione 2D**: Riduzione dimensionale con PCA per rappresentazioni grafiche
- 📈 **Analisi Multi-Dominio**: Confronto semantico tra diversi tipi di contenuti web
- 💾 **Export Multipli**: Output in JSON, TXT e visualizzazioni PNG

## 🛠️ Tecnologie

### Core Technologies
- **Python 3.8+**: Linguaggio di programmazione principale
- **Requests**: HTTP client per web crawling
- **BeautifulSoup**: Parsing HTML avanzato
- **NumPy**: Computazione numerica ad alte prestazioni

### Machine Learning Stack
- **Sentence Transformers**: Modelli transformer per embeddings semantici
- **Scikit-learn**: PCA e metriche di similarità
- **Matplotlib**: Visualizzazione scientifica dei dati

### Modello AI
- **paraphrase-multilingual-MiniLM-L12-v2**: Modello pre-addestrato multilingua (384D)

## 🚀 Installazione

### Requirements.txt
```txt
requests>=2.28.0
beautifulsoup4>=4.11.0
sentence-transformers>=2.2.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
```

## 📊 Risultati

### Test 1: Confronto Multi-Dominio
```
📈 STATISTICHE:
- UniMi: 30 pagine
- YouTube: 30 pagine  
- Treccani: 30 pagine
- Totale: 90 pagine crawlate
- Durata: 4:15 minuti
- Query: "iscrizione università corsi magistrale triennale erasmus"
```

### Test 2: Focus UniMi con Ricerca Semantica
```
🔍 TOP 10 RISULTATI (Similarity Score):

1. 0.6515 - Preparare i test in ingresso
2. 0.6474 - Orientarsi e scegliere
3. 0.6370 - Corsi di laurea magistrale
4. 0.6369 - Iscriversi a una prima laurea
5. 0.6305 - Incontri per le matricole
6. 0.6166 - Iscriversi a una seconda laurea
7. 0.6158 - Studiare (pagina generale)
8. 0.6140 - Abilità informatica
9. 0.5999 - Frequentare un corso post laurea

📊 PERFORMANCE:
- 300 pagine UniMi crawlate
- 301 vettori generati (300 + query)
- Query: "come iscriversi al corso di informatica all'università"
```

## 📈 Visualizzazioni

### Confronto Multi-Dominio
![Visualizzazione Multi-Dominio](https://github.com/user-attachments/assets/d14a657e-7051-4024-b0c9-6fa2199bfa63)

**Legenda:**
- 🟢 **Verde**: Pagine UniMi (contenuti accademici)
- 🔴 **Rosso**: Pagine YouTube (contenuti multimediali)
- 🔵 **Blu**: Pagine Treccani (contenuti enciclopedici)
- 🟣 **Viola**: Query di test

### Analisi Semantica UniMi
![Visualizzazione UniMi](https://github.com/user-attachments/assets/e7f1d619-2772-4026-9062-f324623c778a)

**Legenda:**
- 🟢 **Verde**: 300 pagine UniMi
- 🟣 **Stella Viola**: Query "come iscriversi al corso di informatica all'università"

## 🏗️ Architettura

### Algoritmi Utilizzati

#### Sentence Embeddings
```python
# Trasformazione testo → vettore semantico
embedding = model.encode(text, normalize_embeddings=True)
# Output: vector(384D) normalizzato
```

#### Ricerca Semantica
```python
# Calcolo similarità coseno
similarities = cosine_similarity(query_vector, document_vectors)
top_k = np.argsort(similarities)[::-1][:10]
```

#### Riduzione Dimensionale
```python
# PCA per visualizzazione 2D
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_matrix)
```

## 📁 Output Files

### File Generati Automaticamente

| File | Descrizione | Formato |
|------|-------------|---------|
| `top_10_similar_pages.json` | Risultati ricerca semantica | JSON |
| `top_10_similar_pages.txt` | Risultati in formato leggibile | TXT |
| `visualizzazione_unimi_vettori.png` | Grafico PCA embeddings | PNG |
| `unimi_crawling_results.json` | Summary completo crawling | JSON |
| `multi_domain_results.json` | Confronto multi-dominio | JSON |

### Struttura JSON Risultati
```json
{
  "query": "come iscriversi al corso di informatica all'università",
  "timestamp": "2025-01-XX",
  "total_pages_crawled": 300,
  "top_10_results": [
    {
      "rank": 1,
      "url": "https://unimi.it/...",
      "title": "Pagina Title",
      "similarity": 0.6515,
      "content_preview": "..."
    }
  ]
}
```

## ⚙️ Configurazione Avanzata

### Parametri Crawling
```python
# Personalizzazione comportamento crawler
DELAY = 0.2          # Pausa tra richieste (secondi)
TARGET_PAGES = 300   # Numero pagine target
MAX_CONTENT = 4000   # Limite caratteri per pagina
TIMEOUT = 15         # Timeout richieste HTTP
```

### Modelli Alternativi
```python
# Altri modelli sentence-transformers supportati
models = [
    "paraphrase-multilingual-MiniLM-L12-v2",  # Default
    "paraphrase-multilingual-mpnet-base-v2",  # Più accurato
    "distiluse-base-multilingual-cased"       # Più veloce
]
```



# Web Crawler Bot

Un crawler web multi-threaded ad alte prestazioni scritto in Python, progettato per mappare completamente un dominio specifico e raccogliere tutti i link interni ed esterni.

## 🚀 Caratteristiche

- **Multi-threading**: Utilizza 10 thread simultanei per massimizzare la velocità di crawling
- **Crawling intelligente**: Distingue tra link interni (da analizzare) e esterni (da raccogliere ma non seguire)
- **Auto-terminazione**: Si ferma automaticamente quando ha esaurito tutte le pagine del dominio
- **Interruzione sicura**: Gestisce Ctrl+C salvando automaticamente i risultati parziali
- **Deduplica**: Evita link duplicati utilizzando strutture dati Set per prestazioni O(1)
- **Session HTTP persistenti**: Riutilizza le connessioni per ridurre l'overhead
- **Gestione errori robusta**: Continua il crawling anche in presenza di pagine non raggiungibili

## 🛠️ Tecnologie

- **Python 3.x**
- **requests**: Per le richieste HTTP
- **BeautifulSoup4**: Per il parsing HTML e l'estrazione dei link
- **threading**: Per il crawling parallelo
- **urllib.parse**: Per la gestione e validazione degli URL

## 📋 Requisiti

```bash
pip install requests beautifulsoup4
```

## 🎯 Come funziona

1. **Input**: L'utente inserisce il dominio target (es. "unimi.it")
2. **Inizializzazione**: Crea una queue con l'URL base e avvia i worker thread
3. **Crawling ricorsivo**: 
   - Ogni thread estrae una pagina dalla queue
   - Analizza la pagina per trovare tutti i link
   - Aggiunge link interni al dominio alla queue per ulteriore analisi
   - Raccoglie tutti i link (interni ed esterni) nel set finale
4. **Terminazione**: Si ferma quando la queue è vuota (nessuna nuova pagina da analizzare)
5. **Output**: Salva tutti i link unici in `links.txt`, ordinati alfabeticamente

## 🔧 Utilizzo

```bash
python crawler.py
```

Inserisci il dominio quando richiesto (es. "example.com" o "https://example.com")

### Controlli

- **Avvio automatico**: Il crawler inizia immediatamente dopo l'inserimento del dominio
- **Monitoraggio real-time**: Visualizza in tempo reale le pagine in analisi
- **Interruzione manuale**: Premi `Ctrl+C` per fermare il crawling e salvare i risultati parziali
- **Completamento automatico**: Il programma termina da solo quando ha mappato tutto il dominio

## 📊 Output

Il programma genera:
- **Feedback real-time**: Mostra ogni URL in fase di analisi
- **Statistiche finali**: Numero totale di link unici trovati
- **File `links.txt`**: Contiene tutti i link ordinati alfabeticamente, uno per riga

## 🎯 Casi d'uso

- **Audit SEO**: Mappatura completa di un sito web
- **Analisi della struttura**: Comprensione dell'architettura informativa
- **Data collection**: Raccolta di tutti gli URL per ulteriori analisi
- **Security testing**: Individuazione di tutte le pagine pubbliche di un dominio
- **Ricerca accademica**: Analisi di grandi portali istituzionali

## ⚡ Prestazioni

- **Velocità**: Crawling parallelo con 10 thread simultanei
- **Efficienza memoria**: Utilizzo di Set per deduplica O(1)
- **Scalabilità**: Gestisce domini con migliaia di pagine
- **Robustezza**: Timeout di 10 secondi per pagina, gestione automatica degli errori

## 🔒 Rispetto dei server

- User-Agent standard per identificazione
- Timeout configurato per evitare sovraccarico server
- Gestione cortese degli errori HTTP

## 📝 Note tecniche

- **Thread safety**: Utilizzo di lock per accesso sicuro alle strutture dati condivise
- **URL normalization**: Pulizia e standardizzazione degli URL estratti
- **Domain validation**: Controllo rigoroso dell'appartenenza al dominio target
- **Graceful shutdown**: Chiusura pulita dei thread in caso di interruzione

#!/usr/bin/env python3
"""
Bot Integrato UniMi - Ricerca Semantica + Web Scraping + AI Response
Combina ricerca vettoriale, scraping e generazione AI in un unico flusso
Genera risposte in formato LaTeX con guida passo-passo
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import logging
import os
import glob
from typing import List, Dict, Tuple
import argparse
import re
from urllib.parse import urljoin, urlparse
from groq import Groq
import tiktoken

class UniMiIntegratedBot:
    def __init__(self, groq_api_key=None, model_name="openai/gpt-oss-20b", embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        # Configurazione AI
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY', '')
        self.model_name = model_name
        self.groq_client = None
        
        # Configurazione embedding
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        
        # Database vettoriale
        self.database = None
        self.database_info = None
        
        # Configurazioni scraping
        self.delay = 0.5
        self.timeout = 15
        self.max_content_length = 10000
        
        # Dati di sessione
        self.current_query = ""
        self.search_results = []
        self.scraped_pages = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Headers per scraping
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'it-IT,it;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Inizializzazione componenti
        self.init_embedding_model()
        self.init_groq_client()
        self.init_tokenizer()
    
    def init_embedding_model(self):
        """Inizializza il modello di embedding per la ricerca semantica"""
        try:
            self.logger.info(f"Caricamento modello embedding: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.logger.info("Modello embedding caricato con successo")
        except Exception as e:
            self.logger.error(f"Errore nel caricare il modello embedding: {e}")
            raise
    
    def init_groq_client(self):
        """Inizializza il client Groq per l'AI"""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY non fornita. Imposta la variabile d'ambiente o passala come parametro.")
        
        try:
            self.groq_client = Groq(api_key=self.groq_api_key)
            self.logger.info("Client Groq inizializzato con successo")
        except Exception as e:
            self.logger.error(f"Errore nell'inizializzazione del client Groq: {e}")
            raise
    
    def init_tokenizer(self):
        """Inizializza il tokenizer per conteggio token"""
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def find_database_file(self, database_path=None):
        """Trova automaticamente il file del database più recente"""
        if database_path and os.path.exists(database_path):
            return database_path
        
        patterns = [
            'unimi_vector_database_*.json',
            'unimi_vector_database_compact_*.json'
        ]
        
        database_files = []
        for pattern in patterns:
            database_files.extend(glob.glob(pattern))
        
        if not database_files:
            raise FileNotFoundError("Nessun file database trovato. Assicurati di aver eseguito prima il crawler.")
        
        latest_file = max(database_files, key=os.path.getctime)
        self.logger.info(f"Utilizzo database: {latest_file}")
        return latest_file
    
    def load_database(self, database_path=None):
        """Carica il database vettoriale"""
        try:
            db_file = self.find_database_file(database_path)
            
            print(f"📂 Caricamento database: {db_file}")
            
            with open(db_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.database_info = data['database_info']
            self.database = data['pages']
            
            print(f"✅ Database caricato:")
            print(f"   - Pagine: {len(self.database)}")
            print(f"   - Dimensione embedding: {self.database_info.get('embedding_dimension', 'N/A')}")
            print(f"   - Dominio: {self.database_info.get('domain', 'N/A')}")
            print(f"   - Creato il: {self.database_info.get('created_at', 'N/A')}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nel caricamento del database: {e}")
            return False
    
    def create_query_embedding(self, query):
        """Crea l'embedding per la query dell'utente"""
        try:
            if not query.strip():
                return None
            
            embedding = self.embedding_model.encode(query, normalize_embeddings=True)
            return embedding.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Errore nella creazione embedding per query: {e}")
            return None
    
    def semantic_search(self, query, top_k=20):
        """Esegue la ricerca semantica e restituisce i risultati"""
        if not self.database:
            raise ValueError("Database non caricato. Esegui prima load_database()")
        
        self.current_query = query
        
        print(f"\n🔍 === RICERCA SEMANTICA ===")
        print(f"Query: '{query}'")
        print(f"Ricerca dei top {top_k} risultati...")
        
        # Crea embedding della query
        query_embedding = self.create_query_embedding(query)
        if query_embedding is None:
            raise ValueError("Impossibile creare embedding per la query")
        
        # Estrai tutti gli embeddings dal database
        embeddings_matrix = []
        valid_pages = []
        
        for page in self.database:
            if 'embedding' in page and page['embedding']:
                embeddings_matrix.append(page['embedding'])
                valid_pages.append(page)
        
        if not embeddings_matrix:
            raise ValueError("Nessun embedding valido trovato nel database")
        
        embeddings_matrix = np.array(embeddings_matrix)
        query_vector = query_embedding.reshape(1, -1)
        
        # Calcola similarità coseno
        similarities = cosine_similarity(query_vector, embeddings_matrix)[0]
        
        # Trova i top K più simili
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices, 1):
            similarity_score = similarities[idx]
            page = valid_pages[idx]
            
            result = {
                'rank': i,
                'id': page['id'],
                'url': page['url'],
                'title': page['title'],
                'description': page.get('description', ''),
                'keywords': page.get('keywords', ''),
                'similarity_score': float(similarity_score),
                'content_length': page.get('content_length', 0),
                'crawled_at': page.get('crawled_at', ''),
                'content_preview': page.get('content_preview', '')[:200] + '...' if page.get('content_preview') else ''
            }
            results.append(result)
            
            print(f"{i:2d}. {similarity_score:.4f} - {page['title'][:60]}...")
        
        self.search_results = results
        return results
    
    def extract_page_content(self, soup, url):
        """Estrae il contenuto principale da una pagina"""
        try:
            # Rimuovi elementi non utili
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']):
                element.decompose()
            
            # Cerca contenuto principale
            main_content = ""
            
            # Prova diversi selettori per il contenuto principale
            main_selectors = [
                'main', 'article', '.content', '#content', '.main-content', 
                '.entry-content', '.post-content', '.page-content',
                '[role="main"]', '.container', '.wrapper'
            ]
            
            for selector in main_selectors:
                main_elem = soup.select_one(selector)
                if main_elem:
                    main_content = main_elem.get_text(separator=' ', strip=True)
                    break
            
            if not main_content:
                body = soup.find('body')
                if body:
                    main_content = body.get_text(separator=' ', strip=True)
                else:
                    main_content = soup.get_text(separator=' ', strip=True)
            
            # Pulizia del testo
            main_content = re.sub(r'\s+', ' ', main_content)
            main_content = main_content.strip()
            
            # Limita la lunghezza
            if len(main_content) > self.max_content_length:
                main_content = main_content[:self.max_content_length] + "..."
            
            return main_content
            
        except Exception as e:
            self.logger.error(f"Errore nell'estrazione contenuto da {url}: {e}")
            return ""
    
    def scrape_single_page(self, url):
        """Fa scraping di una singola pagina"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Verifica che sia HTML
            content_type = response.headers.get('content-type', '')
            if 'text/html' not in content_type.lower():
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Estrai metadati
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            description = ""
            desc_tag = soup.find('meta', attrs={'name': 'description'})
            if desc_tag:
                description = desc_tag.get('content', '').strip()
            
            # Estrai contenuto principale
            content = self.extract_page_content(soup, url)
            
            if content:
                page_data = {
                    'url': url,
                    'title': title,
                    'description': description,
                    'content': content,
                    'content_length': len(content),
                    'scraped_at': datetime.now().isoformat()
                }
                
                return page_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Errore scraping {url}: {e}")
            return None
    
    def scrape_top_pages(self):
        """Fa scraping delle pagine dai risultati della ricerca semantica"""
        if not self.search_results:
            raise ValueError("Nessun risultato di ricerca disponibile")
        
        print(f"\n🕷️ === SCRAPING DELLE PAGINE ===")
        print(f"Pagine da processare: {len(self.search_results)}")
        
        self.scraped_pages = []
        scraped_count = 0
        failed_count = 0
        
        for i, result in enumerate(self.search_results, 1):
            url = result['url']
            
            print(f"[{i}/{len(self.search_results)}] Scraping: {url[:60]}...")
            
            page_data = self.scrape_single_page(url)
            
            if page_data:
                # Aggiungi info dalla ricerca semantica
                page_data['rank'] = result['rank']
                page_data['similarity_score'] = result['similarity_score']
                page_data['search_title'] = result['title']
                page_data['search_description'] = result['description']
                
                self.scraped_pages.append(page_data)
                scraped_count += 1
                
                if scraped_count % 5 == 0:
                    print(f"   ✅ Completate {scraped_count} pagine")
            else:
                failed_count += 1
            
            # Delay tra requests
            if self.delay > 0:
                time.sleep(self.delay)
        
        print(f"✅ Scraping completato:")
        print(f"   - Pagine elaborate: {scraped_count}")
        print(f"   - Fallimenti: {failed_count}")
        
        return scraped_count
    
    def count_tokens(self, text):
        """Conta i token nel testo"""
        try:
            return len(self.tokenizer.encode(text))
        except:
            return len(text.split()) * 1.3
    
    def create_latex_ai_prompt(self):
        """Crea il prompt per l'AI per generare risposta in LaTeX"""
        
        sorted_pages = sorted(self.scraped_pages, key=lambda x: x.get('rank', 999))
        
        prompt = f"""Sei un assistente esperto dell'Università Statale di Milano (UniMi). 
Un utente ha fatto questa domanda: "{self.current_query}"

Ho raccolto informazioni da {len(self.scraped_pages)} pagine del sito unimi.it per aiutarti a rispondere. 
Devi fornire una risposta completa in formato LaTeX che includa:

1. **SEZIONE INTRODUTTIVA**: Riassunto della domanda e panoramica della risposta
2. **RISPOSTA DIRETTA**: Risposta chiara e concisa alla domanda specifica  
3. **GUIDA PASSO-PASSO**: Una guida dettagliata numerata su come completare/risolvere quanto richiesto
4. **LINK E RISORSE**: Solo i link più pertinenti e utili organizzati per categoria
5. **NOTE AGGIUNTIVE**: Eventuali avvertenze, scadenze, requisiti importanti

REQUISITI PER IL FORMATO LaTeX:
- Usa \\documentclass{{article}} con pacchetti italiani
- Struttura con \\section{{}} e \\subsection{{}}
- Usa \\begin{{enumerate}} per le guide passo-passo
- Usa \\href{{URL}}{{testo}} per i link
- Usa \\textbf{{}} per evidenziare parti importanti
- Usa \\textit{{}} per note e osservazioni
- Includi \\begin{{itemize}} per elenchi di risorse

IMPORTANTE: 
- Scrivi tutto in italiano
- Cita SOLO i link veramente rilevanti, non tutti quelli forniti
- Organizza la guida in passi chiari e actionable
- Usa un tono professionale ma amichevole
- Se informazioni mancanti, indicalo chiaramente

CONTENUTO DELLE PAGINE ANALIZZATE:

"""
        
        # Aggiungi contenuto delle pagine con limite di token
        total_tokens = self.count_tokens(prompt)
        max_tokens = 6000
        
        for i, page in enumerate(sorted_pages[:15]):
            page_content = f"""
--- PAGINA {page['rank']} (Rilevanza: {page['similarity_score']:.4f}) ---
URL: {page['url']}
Titolo: {page['title']}
Descrizione: {page.get('search_description', page.get('description', ''))}
Contenuto: {page['content']}

"""
            
            page_tokens = self.count_tokens(page_content)
            
            if total_tokens + page_tokens > max_tokens:
                print(f"⚠️ Raggiunto limite token. Processate {i} pagine su {len(sorted_pages)}")
                break
                
            prompt += page_content
            total_tokens += page_tokens
        
        prompt += f"""
GENERA UNA RISPOSTA COMPLETA IN FORMATO LaTeX che aiuti concretamente l'utente a risolvere la sua domanda.
La risposta deve essere pronta per la compilazione LaTeX e ben strutturata.
"""
        
        return prompt
    
    def generate_ai_response(self, prompt):
        """Genera la risposta usando Groq AI"""
        try:
            print(f"\n🤖 === GENERAZIONE RISPOSTA AI ===")
            print(f"Modello: {self.model_name}")
            print(f"Token prompt: ~{self.count_tokens(prompt)}")
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": "Sei un assistente esperto dell'Università Statale di Milano. Fornisci risposte accurate in formato LaTeX, ben strutturate e dettagliate basate sulle informazioni fornite. Sempre in italiano."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model_name,
                max_tokens=3000,
                temperature=0.1,
                top_p=1,
                stream=False
            )
            
            ai_response = response.choices[0].message.content
            
            print(f"✅ Risposta AI generata ({len(ai_response)} caratteri)")
            
            return ai_response
            
        except Exception as e:
            self.logger.error(f"Errore nella generazione risposta AI: {e}")
            raise
    
    def save_results(self, ai_response):
        """Salva tutti i risultati in diversi formati"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Pulisci la query per il filename
        clean_query = re.sub(r'[^\w\s-]', '', self.current_query)[:50]
        clean_query = re.sub(r'\s+', '_', clean_query)
        
        # Dati completi
        complete_results = {
            'query_info': {
                'user_query': self.current_query,
                'timestamp': datetime.now().isoformat(),
                'ai_model': self.model_name,
                'embedding_model': self.embedding_model_name,
                'pages_found': len(self.search_results),
                'pages_scraped': len(self.scraped_pages),
                'database_info': self.database_info
            },
            'search_results': self.search_results,
            'scraped_pages': self.scraped_pages,
            'ai_response_latex': ai_response,
            'processing_stats': {
                'total_similarity_avg': sum(r['similarity_score'] for r in self.search_results) / len(self.search_results),
                'content_chars_total': sum(len(p['content']) for p in self.scraped_pages),
                'successful_scraping_rate': len(self.scraped_pages) / len(self.search_results) if self.search_results else 0
            }
        }
        
        # File JSON completo
        json_filename = f'unimi_complete_response_{clean_query}_{timestamp}.json'
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, ensure_ascii=False, indent=2)
        
        # File LaTeX puro per compilazione
        latex_filename = f'unimi_response_{clean_query}_{timestamp}.tex'
        with open(latex_filename, 'w', encoding='utf-8') as f:
            f.write(ai_response)
        
        # File di testo leggibile con metadata
        txt_filename = f'unimi_response_{clean_query}_{timestamp}.txt'
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("🎓 UNIVERSITÀ STATALE DI MILANO - RISPOSTA AI ASSISTENTE\n")
            f.write("=" * 80 + "\n")
            f.write(f"DOMANDA: {self.current_query}\n")
            f.write(f"DATA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"MODELLO AI: {self.model_name}\n")
            f.write(f"MODELLO EMBEDDING: {self.embedding_model_name}\n")
            f.write(f"PAGINE TROVATE: {len(self.search_results)}\n")
            f.write(f"PAGINE ANALIZZATE: {len(self.scraped_pages)}\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write("RISPOSTA LaTeX:\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(ai_response)
            
            f.write(f"\n\n" + "=" * 80 + "\n")
            f.write("FONTI CONSULTATE:\n")
            f.write("=" * 80 + "\n")
            
            for page in sorted(self.scraped_pages, key=lambda x: x.get('rank', 999)):
                f.write(f"\n{page['rank']}. {page['title']}\n")
                f.write(f"   URL: {page['url']}\n")
                f.write(f"   Rilevanza: {page['similarity_score']:.4f}\n")
                f.write(f"   Contenuto: {len(page['content'])} caratteri\n")
        
        print(f"\n💾 Risultati salvati in:")
        print(f"   - {json_filename} (dati completi)")
        print(f"   - {latex_filename} (file LaTeX per compilazione)")
        print(f"   - {txt_filename} (versione leggibile con metadata)")
        
        return json_filename, latex_filename, txt_filename
    
    def process_query(self, query, database_path=None, top_k=20):
        """Processo completo: ricerca semantica -> scraping -> risposta AI"""
        print("🎓 === AVVIO BOT INTEGRATO UNIMI ===")
        
        try:
            # 1. Carica database se necessario
            if not self.database:
                if not self.load_database(database_path):
                    raise ValueError("Impossibile caricare il database")
            
            # 2. Ricerca semantica
            search_results = self.semantic_search(query, top_k)
            
            if not search_results:
                print("❌ Nessun risultato trovato nella ricerca semantica")
                return
            
            # 3. Scraping delle pagine
            scraped_count = self.scrape_top_pages()
            
            if scraped_count == 0:
                print("❌ Nessuna pagina è stata elaborata con successo nel scraping")
                return
            
            # 4. Genera prompt LaTeX
            prompt = self.create_latex_ai_prompt()
            
            # 5. Genera risposta AI
            ai_response = self.generate_ai_response(prompt)
            
            # 6. Salva risultati
            files = self.save_results(ai_response)
            
            # 7. Mostra riassunto
            self.print_summary()
            
            print(f"\n✅ Processo completato con successo!")
            print(f"📄 Risposta LaTeX generata e salvata")
            
            return ai_response
            
        except Exception as e:
            self.logger.error(f"Errore durante l'elaborazione: {e}")
            print(f"❌ Errore: {e}")
    
    def print_summary(self):
        """Stampa il riassunto dell'elaborazione"""
        print(f"\n" + "=" * 70)
        print("📊 RIASSUNTO ELABORAZIONE COMPLETA")
        print("=" * 70)
        print(f"Domanda: {self.current_query}")
        print(f"Ricerca semantica: {len(self.search_results)} risultati")
        print(f"Scraping: {len(self.scraped_pages)} pagine elaborate")
        print(f"Similarità media: {sum(r['similarity_score'] for r in self.search_results) / len(self.search_results):.4f}")
        print(f"Contenuto raccolto: {sum(len(p['content']) for p in self.scraped_pages):,} caratteri")
        print(f"Modelli utilizzati:")
        print(f"  - Embedding: {self.embedding_model_name}")
        print(f"  - AI: {self.model_name}")
        print("=" * 70)
    
    def interactive_mode(self, database_path=None):
        """Modalità interattiva per domande multiple"""
        print("\n🤖 === MODALITÀ INTERATTIVA BOT UNIMI ===")
        print("Fai le tue domande sull'Università Statale di Milano")
        print("Il bot genererà risposte LaTeX complete con guide passo-passo")
        print("Digita 'quit' per uscire")
        print("-" * 60)
        
        # Carica database una volta
        if not self.database:
            if not self.load_database(database_path):
                print("❌ Impossibile caricare il database")
                return
        
        while True:
            try:
                query = input("\n🔍 La tua domanda: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Arrivederci!")
                    break
                
                if not query:
                    print("⚠️ Inserisci una domanda valida")
                    continue
                
                # Pulisci stato precedente
                self.search_results = []
                self.scraped_pages = []
                
                # Processa la query
                self.process_query(query, top_k=20)
                
            except KeyboardInterrupt:
                print("\n👋 Interruzione utente. Arrivederci!")
                break
            except Exception as e:
                self.logger.error(f"Errore durante l'elaborazione: {e}")
                print(f"❌ Errore: {e}")

def main():
    parser = argparse.ArgumentParser(description='Bot Integrato UniMi - Ricerca Semantica + Scraping + AI LaTeX')
    parser.add_argument('--database', '-d', help='Percorso del file database vettoriale')
    parser.add_argument('--query', '-q', help='Domanda specifica (modalità singola)')
    parser.add_argument('--groq-key', '-k', help='Chiave API Groq')
    parser.add_argument('--ai-model', '-m', default='openai/gpt-oss-20b',
                       choices=['llama-3.1-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768'],
                       help='Modello AI Groq')
    parser.add_argument('--embedding-model', '-e', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                       help='Modello per embedding semantici')
    parser.add_argument('--top-k', '-t', type=int, default=20, help='Numero di pagine da analizzare')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay tra richieste scraping (secondi)')
    parser.add_argument('--interactive', '-i', action='store_true', help='Modalità interattiva')
    
    args = parser.parse_args()
    
    print("🎓 Bot Integrato Università Statale di Milano")
    print("=" * 60)
    print("Ricerca Semantica + Web Scraping + Risposta AI in LaTeX")
    print("=" * 60)
    
    try:
        # Inizializza il bot
        bot = UniMiIntegratedBot(
            groq_api_key=args.groq_key,
            model_name=args.ai_model,
            embedding_model_name=args.embedding_model
        )
        
        bot.delay = args.delay
        
        # Modalità interattiva o singola query
        if args.interactive or not args.query:
            bot.interactive_mode(args.database)
        else:
            # Modalità singola query
            result = bot.process_query(args.query, args.database, args.top_k)
            if result:
                print(f"\n🎯 Query elaborata con successo!")
                print(f"📄 Risposta LaTeX generata")
            
    except Exception as e:
        print(f"❌ Errore critico: {e}")
        logging.error(f"Errore critico: {e}")

if __name__ == "__main__":
    main()
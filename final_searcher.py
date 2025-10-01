#!/usr/bin/env python3

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

class MultiDomainSearchBot:
    def __init__(self, groq_api_key=None, model_name="openai/gpt-oss-20b", embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.groq_api_key = groq_api_key or 'API'
        self.model_name = model_name
        self.groq_client = None
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.database = None
        self.database_info = None
        self.scraping_database = {}
        self.scraping_db_path = "multi_domain_scraping_database.json"
        self.delay = 0.3
        self.timeout = 15
        self.max_content_length = 8000
        self.current_query = ""
        self.cleaned_query = ""
        self.search_results = []
        self.scraped_pages = []
        self.current_domain = None
        
        logging.basicConfig(level=logging.ERROR)
        self.logger = logging.getLogger(__name__)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'it-IT,it;q=0.9,en;q=0.8',
            'Connection': 'keep-alive',
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        self.init_embedding_model()
        self.init_groq_client()
        self.init_tokenizer()
        self.load_scraping_database()
    
    def init_embedding_model(self):
        try:
            print("Loading embedding model...")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print("Embedding model loaded successfully")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise
    
    def init_groq_client(self):
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY required")
        
        try:
            self.groq_client = Groq(api_key=self.groq_api_key)
            print("Groq client initialized")
        except Exception as e:
            print(f"Error initializing Groq client: {e}")
            raise
    
    def init_tokenizer(self):
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def get_available_domains(self):
        database_files = glob.glob('*_database_*.json')
        domains = []
        
        for db_file in database_files:
            try:
                with open(db_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    domain = data.get('database_info', {}).get('domain')
                    if domain and domain not in domains:
                        domains.append(domain)
            except:
                continue
        
        return sorted(domains)
    
    def find_database_for_domain(self, domain):
        domain_clean = domain.replace('.', '_')
        pattern = f'{domain_clean}_database_*.json'
        database_files = glob.glob(pattern)
        
        if not database_files:
            return None
        
        latest_file = max(database_files, key=os.path.getctime)
        return latest_file
    
    def load_database_by_domain(self, domain):
        try:
            db_file = self.find_database_for_domain(domain)
            
            if not db_file:
                raise FileNotFoundError(f"No database found for domain: {domain}")
            
            print(f"Loading database from {db_file}...")
            with open(db_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.database_info = data.get('database_info', {})
            self.database = data.get('pages', [])
            self.current_domain = self.database_info.get('domain')
            
            if not self.database:
                raise ValueError("Database contains no pages")
            
            if self.current_domain != domain:
                print(f"Warning: Requested domain {domain} but loaded {self.current_domain}")
            
            print(f"Database loaded successfully:")
            print(f"   - Domain: {self.current_domain}")
            print(f"   - Pages: {len(self.database)}")
            print(f"   - Version: {self.database_info.get('version', 'unknown')}")
            
            return True
            
        except Exception as e:
            print(f"Error loading database for domain {domain}: {e}")
            return False
    
    def clean_user_query(self, user_query):
        try:
            print(f"Cleaning query: '{user_query}'")
            
            cleaning_prompt = f"""Sei un esperto di ottimizzazione delle query per ricerca semantica universitaria.

L'utente ha fatto questa domanda: "{user_query}"

Il tuo compito è estrarre SOLO le parole chiave essenziali per una ricerca semantica efficace su un database universitario.

REGOLE:
1. Rimuovi parole di cortesia
2. Rimuovi articoli, preposizioni e congiunzioni non essenziali
3. Mantieni solo i concetti chiave e i termini tecnici
4. Converti forme verbali in sostantivi quando possibile
5. Massimo 8-10 parole chiave

Rispondi SOLO con le parole chiave ottimizzate, separate da spazi. Niente altro."""

            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Sei un esperto di ottimizzazione query per ricerca semantica universitaria."
                    },
                    {
                        "role": "user", 
                        "content": cleaning_prompt
                    }
                ],
                model=self.model_name,
                max_tokens=100,
                temperature=0.1,
                top_p=1,
                stream=False
            )
            
            cleaned_query = response.choices[0].message.content.strip()
            
            if not cleaned_query or len(cleaned_query.split()) < 2:
                cleaned_query = self.basic_query_cleaning(user_query)
            
            print(f"Cleaned query: '{cleaned_query}'")
            self.cleaned_query = cleaned_query
            return cleaned_query
            
        except Exception as e:
            print(f"Error in AI query cleaning: {e}")
            cleaned_query = self.basic_query_cleaning(user_query)
            self.cleaned_query = cleaned_query
            return cleaned_query
    
    def basic_query_cleaning(self, user_query):
        courtesy_words = [
            'potresti', 'riusciresti', 'puoi', 'riesci', 'per favore', 'grazie',
            'vorrei', 'volevo', 'mi servirebbe', 'mi serve', 'ho bisogno',
            'come faccio', 'dove posso', 'è possibile', 'si può'
        ]
        
        stop_words = [
            'il', 'la', 'lo', 'le', 'gli', 'i', 'un', 'una', 'uno', 'del', 'della',
            'dei', 'delle', 'degli', 'al', 'alla', 'alle', 'agli', 'dal', 'dalla',
            'dalle', 'dagli', 'nel', 'nella', 'nelle', 'negli', 'sul', 'sulla',
            'sulle', 'sugli', 'per', 'con', 'tra', 'fra', 'di', 'da', 'in', 'su',
            'a', 'e', 'o', 'ma', 'però', 'quindi', 'anche', 'ancora', 'già'
        ]
        
        text = user_query.lower()
        
        for phrase in courtesy_words:
            text = text.replace(phrase, ' ')
        
        words = text.split()
        filtered_words = []
        
        for word in words:
            word = re.sub(r'[^\w\s]', '', word)
            if word and word not in stop_words and len(word) > 2:
                filtered_words.append(word)
        
        return ' '.join(filtered_words[:10])
    
    def load_scraping_database(self):
        try:
            if os.path.exists(self.scraping_db_path):
                with open(self.scraping_db_path, 'r', encoding='utf-8') as f:
                    self.scraping_database = json.load(f)
                print(f"Loaded scraping database with {len(self.scraping_database.get('pages', {}))} cached pages")
            else:
                self.scraping_database = {
                    'pages': {},
                    'metadata': {
                        'created_at': datetime.now().isoformat(),
                        'last_updated': datetime.now().isoformat(),
                        'total_pages': 0
                    }
                }
        except Exception as e:
            print(f"Error loading scraping database: {e}")
            self.scraping_database = {
                'pages': {},
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat(),
                    'total_pages': 0
                }
            }
    
    def save_scraping_database(self):
        try:
            self.scraping_database['metadata']['last_updated'] = datetime.now().isoformat()
            self.scraping_database['metadata']['total_pages'] = len(self.scraping_database['pages'])
            
            with open(self.scraping_db_path, 'w', encoding='utf-8') as f:
                json.dump(self.scraping_database, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def create_query_embedding(self, query):
        try:
            if not query.strip():
                return None
            
            embedding = self.embedding_model.encode(query, normalize_embeddings=True)
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"Error creating query embedding: {e}")
            return None
    
    def semantic_search(self, query, top_k=15):
        if not self.database:
            raise ValueError("Database not loaded")
        
        print(f"Original query: '{query}'")
        
        cleaned_query = self.clean_user_query(query)
        print(f"Searching with cleaned query: '{cleaned_query}'")
        
        self.current_query = query
        
        query_embedding = self.create_query_embedding(cleaned_query)
        if query_embedding is None:
            raise ValueError("Unable to create query embedding")
        
        embeddings_matrix = []
        valid_pages = []
        
        for page in self.database:
            if 'embedding' in page and page['embedding']:
                try:
                    embedding = np.array(page['embedding'], dtype=np.float32)
                    embeddings_matrix.append(embedding)
                    valid_pages.append(page)
                except Exception as e:
                    continue
        
        if not embeddings_matrix:
            raise ValueError("No valid embeddings found in database")
        
        print(f"Comparing query against {len(embeddings_matrix)} page embeddings...")
        
        embeddings_matrix = np.array(embeddings_matrix)
        query_vector = query_embedding.reshape(1, -1)
        
        similarities = cosine_similarity(query_vector, embeddings_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices, 1):
            similarity_score = similarities[idx]
            page = valid_pages[idx]
            
            content_info = self.extract_page_content_info(page)
            
            result = {
                'rank': i,
                'id': page.get('id', i),
                'url': page['url'],
                'title': page.get('title', 'No title'),
                'description': page.get('description', ''),
                'keywords': page.get('keywords', ''),
                'similarity_score': float(similarity_score),
                'content_length': content_info.get('content_length', 0),
                'crawled_at': page.get('crawled_at', ''),
                'content_preview': content_info.get('preview', ''),
                'has_pdf': content_info.get('has_pdf', False),
                'pdf_count': content_info.get('pdf_count', 0),
                'internal_links_count': content_info.get('internal_links_count', 0),
                'internal_pdfs_count': content_info.get('internal_pdfs_count', 0),
                'timing': page.get('timing', {}),
                'links': page.get('links', {})
            }
            results.append(result)
        
        print(f"Found {len(results)} relevant pages")
        
        self.search_results = results
        return results
    
    def extract_page_content_info(self, page):
        content_info = {
            'content_length': 0,
            'preview': '',
            'has_pdf': False,
            'pdf_count': 0,
            'internal_links_count': 0,
            'internal_pdfs_count': 0,
            'main_content': ''
        }
        
        if 'content' in page and isinstance(page['content'], dict):
            content_data = page['content']
            content_info['content_length'] = content_data.get('content_length', 0)
            content_info['main_content'] = content_data.get('main_content', '')
            content_info['preview'] = content_data.get('main_content', '')[:300] + '...' if content_data.get('main_content') else ''
            content_info['pdf_count'] = content_data.get('pdfs_count', 0)
            content_info['has_pdf'] = content_data.get('pdfs_count', 0) > 0 or bool(content_data.get('pdf_content'))
            content_info['internal_links_count'] = content_data.get('internal_links_count', 0)
            content_info['internal_pdfs_count'] = content_data.get('internal_pdfs_count', 0)
        
        elif 'content_preview' in page:
            content_info['preview'] = page['content_preview'][:300] + '...' if page['content_preview'] else ''
            content_info['content_length'] = page.get('content_length', 0)
        
        elif 'main_content' in page:
            content_info['main_content'] = page['main_content']
            content_info['content_length'] = len(page['main_content'])
            content_info['preview'] = page['main_content'][:300] + '...' if page['main_content'] else ''
        
        return content_info
    
    def get_database_content(self, url):
        for page in self.database:
            if page['url'] == url:
                content_info = self.extract_page_content_info(page)
                if content_info['main_content']:
                    return {
                        'url': url,
                        'title': page.get('title', ''),
                        'description': page.get('description', ''),
                        'content': content_info['main_content'],
                        'content_length': content_info['content_length'],
                        'source': 'vector_database',
                        'scraped_at': page.get('crawled_at', ''),
                        'links': page.get('links', {}),
                        'timing': page.get('timing', {})
                    }
        return None
    
    def extract_page_content(self, soup, url):
        try:
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript', 'comment']):
                element.decompose()
            
            main_content = ""
            
            main_selectors = [
                'main', 'article', '.content', '#content', '.main-content', 
                '.entry-content', '.post-content', '.page-content',
                '[role="main"]', '.container', '.wrapper', '.post', '.article-content'
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
            
            main_content = re.sub(r'\s+', ' ', main_content).strip()
            
            if len(main_content) > self.max_content_length:
                main_content = main_content[:self.max_content_length] + "..."
            
            return main_content
            
        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            return ""
    
    def scrape_single_page(self, url):
        try:
            print(f"Scraping: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if 'text/html' not in content_type.lower():
                print(f"Skipping non-HTML content: {content_type}")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            description = ""
            desc_tag = soup.find('meta', attrs={'name': 'description'})
            if desc_tag:
                description = desc_tag.get('content', '').strip()
            
            content = self.extract_page_content(soup, url)
            
            if content:
                print(f"Scraped {len(content)} characters from {url}")
                return {
                    'url': url,
                    'title': title,
                    'description': description,
                    'content': content,
                    'content_length': len(content),
                    'scraped_at': datetime.now().isoformat()
                }
            else:
                print(f"No content extracted from {url}")
            
            return None
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
    
    def get_page_from_database(self, url):
        return self.scraping_database['pages'].get(url)
    
    def update_scraping_database(self, url, page_data):
        self.scraping_database['pages'][url] = page_data
        self.save_scraping_database()
    
    def scrape_top_pages(self):
        if not self.search_results:
            raise ValueError("No search results available")
        
        print(f"Processing {len(self.search_results)} pages...")
        self.scraped_pages = []
        
        for result in self.search_results:
            url = result['url']
            
            page_data = None
            
            page_data = self.get_database_content(url)
            if page_data:
                page_data['source'] = 'vector_database'
                print(f"Using vector database content for {url}")
            
            if not page_data:
                page_data = self.get_page_from_database(url)
                if page_data:
                    page_data['source'] = 'cached'
                    print(f"Using cached content for {url}")
            
            if not page_data:
                page_data = self.scrape_single_page(url)
                if page_data:
                    page_data['source'] = 'fresh_scrape'
                    self.update_scraping_database(url, page_data)
                    
                    if self.delay > 0:
                        time.sleep(self.delay)
            
            if page_data:
                page_data.update({
                    'rank': result['rank'],
                    'similarity_score': result['similarity_score'],
                    'search_title': result['title'],
                    'search_description': result['description'],
                    'has_pdf': result.get('has_pdf', False),
                    'pdf_count': result.get('pdf_count', 0),
                    'internal_links_count': result.get('internal_links_count', 0),
                    'internal_pdfs_count': result.get('internal_pdfs_count', 0),
                    'links': result.get('links', {}),
                    'timing': result.get('timing', {})
                })
                self.scraped_pages.append(page_data)
            else:
                print(f"Failed to get content for {url}")
        
        print(f"Successfully processed {len(self.scraped_pages)} pages")
        return len(self.scraped_pages)
    
    def count_tokens(self, text):
        try:
            return len(self.tokenizer.encode(text))
        except:
            return int(len(text.split()) * 1.3)
    
    def create_latex_ai_prompt(self):
        sorted_pages = sorted(self.scraped_pages, key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        prompt = f"""Sei un assistente esperto del dominio {self.current_domain}. 
Un utente ha fatto questa domanda: "{self.current_query}"
La query è stata ottimizzata per la ricerca semantica in: "{self.cleaned_query}"

Ho raccolto informazioni da {len(self.scraped_pages)} pagine del sito {self.current_domain} per aiutarti a rispondere. 
LA RISPOSTA DATA DEVE CONTENERE SOLAMENTE I DATI CONTENUTI NEI FILE CHE TI VENGONO PASSATI, NON IMPROVVISARE NESSUNA RISPOSTA O INFORMAZIONE. 
Devi fornire una risposta completa in formato LaTeX che includa:

1. **RISPOSTA DIRETTA**: Risposta chiara e concisa alla domanda specifica  
2. **GUIDA PASSO-PASSO**: Una guida dettagliata numerata su come completare/risolvere quanto richiesto
3. **LINK E RISORSE**: Solo i link più pertinenti e utili organizzati per categoria
4. **NOTE AGGIUNTIVE**: Eventuali avvertenze, scadenze, requisiti importanti

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
        
        total_tokens = self.count_tokens(prompt)
        max_tokens = 7000
        
        pages_added = 0
        for page in sorted_pages:
            if pages_added >= 12:
                break
                
            enhanced_info = ""
            if page.get('links'):
                links_data = page['links']
                internal_links_count = len(links_data.get('internal_links', []))
                internal_pdfs_count = len(links_data.get('internal_pdfs', []))
                enhanced_info = f"\nLink interni: {internal_links_count}, PDF interni: {internal_pdfs_count}"
            
            if page.get('timing'):
                timing = page['timing']
                enhanced_info += f"\nTempo elaborazione: scraping {timing.get('scraping_time', 0)}s, embedding {timing.get('embedding_time', 0)}s"
                
            page_content = f"""
--- PAGINA {page['rank']} (Rilevanza: {page['similarity_score']:.4f}) ---
URL: {page['url']}
Titolo: {page['title']}
Descrizione: {page.get('description', page.get('search_description', ''))}
Fonte: {page.get('source', 'unknown')}{enhanced_info}
Contenuto: {page['content'][:3000]}{'...' if len(page['content']) > 3000 else ''}

"""
            
            page_tokens = self.count_tokens(page_content)
            
            if total_tokens + page_tokens > max_tokens:
                print(f"Token limit reached, using {pages_added} pages")
                break
                
            prompt += page_content
            total_tokens += page_tokens
            pages_added += 1
        
        prompt += f"""
GENERA UNA RISPOSTA COMPLETA IN FORMATO LaTeX che aiuti concretamente l'utente a risolvere la sua domanda.
La risposta deve essere pronta per la compilazione LaTeX e ben strutturata.
Ricorda di includere tutti i dettagli importanti trovati nelle pagine analizzate.
Ricorda che la query originale era: "{self.current_query}" e quella ottimizzata: "{self.cleaned_query}"
"""
        
        print(f"Generated prompt with {total_tokens} tokens using {pages_added} pages")
        return prompt
    
    def generate_ai_response(self, prompt):
        try:
            print("Generating AI response...")
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": f"Sei un assistente esperto del dominio {self.current_domain}. Fornisci risposte accurate in formato LaTeX, ben strutturate e dettagliate basate sulle informazioni fornite. Sempre in italiano."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model_name,
                max_tokens=3500,
                temperature=0.1,
                top_p=1,
                stream=False
            )
            
            ai_response = response.choices[0].message.content
            print(f"Generated response ({len(ai_response)} characters)")
            return ai_response
            
        except Exception as e:
            print(f"Error generating AI response: {e}")
            raise
    
    def save_results(self, ai_response):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        clean_query = re.sub(r'[^\w\s-]', '', self.current_query)[:50]
        clean_query = re.sub(r'\s+', '_', clean_query)
        domain_clean = self.current_domain.replace('.', '_')
        
        latex_filename = f'{domain_clean}_response_{clean_query}_{timestamp}.tex'
        with open(latex_filename, 'w', encoding='utf-8') as f:
            f.write(ai_response)
        
        txt_filename = f'{domain_clean}_response_{clean_query}_{timestamp}.txt'
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(f"DOMAIN: {self.current_domain} - RISPOSTA AI\n")
            f.write("=" * 60 + "\n")
            f.write(f"DOMANDA ORIGINALE: {self.current_query}\n")
            f.write(f"QUERY OTTIMIZZATA: {self.cleaned_query}\n")
            f.write(f"DATA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"DOMINIO: {self.current_domain}\n")
            f.write(f"PAGINE TROVATE: {len(self.search_results)}\n")
            f.write(f"PAGINE ANALIZZATE: {len(self.scraped_pages)}\n\n")
            
            f.write("PAGINE PIÙ RILEVANTI:\n")
            f.write("-" * 40 + "\n")
            for page in sorted(self.scraped_pages, key=lambda x: x.get('similarity_score', 0), reverse=True)[:10]:
                f.write(f"• {page['title']} (Score: {page.get('similarity_score', 0):.4f})\n")
                f.write(f"  {page['url']}\n\n")
            
            f.write("RISPOSTA LaTeX:\n")
            f.write("=" * 60 + "\n\n")
            f.write(ai_response)
        
        print(f"Results saved:")
        print(f"   - LaTeX: {latex_filename}")
        print(f"   - Report: {txt_filename}")
        
        return latex_filename, txt_filename
    
    def process_query(self, query, domain, top_k=15):
        try:
            print(f"\nProcessing query: '{query}' for domain: {domain}")
            
            if not self.database or self.current_domain != domain:
                print(f"Loading database for domain: {domain}...")
                if not self.load_database_by_domain(domain):
                    raise ValueError(f"Unable to load database for domain: {domain}")
            
            search_results = self.semantic_search(query, top_k)
            
            if not search_results:
                print("No relevant pages found")
                return None
            
            scraped_count = self.scrape_top_pages()
            
            if scraped_count == 0:
                print("No content could be retrieved")
                return None
            
            prompt = self.create_latex_ai_prompt()
            ai_response = self.generate_ai_response(prompt)
            
            files = self.save_results(ai_response)
            
            print(f"Query processing completed successfully!")
            print(f"   - Domain: {self.current_domain}")
            print(f"   - Original query: '{self.current_query}'")
            print(f"   - Cleaned query: '{self.cleaned_query}'")
            print(f"   - Pages found: {len(self.search_results)}")
            print(f"   - Pages processed: {len(self.scraped_pages)}")
            
            return ai_response
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return None
    
    def interactive_mode(self):
        print("\n" + "="*60)
        print("Multi-Domain Search Bot - Interactive Mode")
        print("="*60)
        
        available_domains = self.get_available_domains()
        
        if not available_domains:
            print("No databases found. Please run the crawler first.")
            return
        
        print(f"Available domains: {', '.join(available_domains)}")
        print("\nCommands:")
        print("  - Type your question in Italian")
        print("  - 'domains' to list available domains")
        print("  - 'switch <domain>' to change domain")
        print("  - 'quit' to exit")
        print("="*60)
        
        current_domain = available_domains[0]
        print(f"\nLoading default domain: {current_domain}...")
        if not self.load_database_by_domain(current_domain):
            print("Error: Unable to load database")
            return
        
        while True:
            try:
                query = input(f"\n[{self.current_domain}] Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if query.lower() == 'domains':
                    print(f"\nAvailable domains:")
                    for domain in available_domains:
                        marker = " (current)" if domain == self.current_domain else ""
                        print(f"  - {domain}{marker}")
                    continue
                
                if query.lower().startswith('switch '):
                    new_domain = query[7:].strip()
                    if new_domain in available_domains:
                        print(f"Switching to domain: {new_domain}...")
                        if self.load_database_by_domain(new_domain):
                            print(f"Now using domain: {self.current_domain}")
                        else:
                            print(f"Failed to switch to domain: {new_domain}")
                    else:
                        print(f"Domain not found: {new_domain}")
                        print(f"Available: {', '.join(available_domains)}")
                    continue
                
                if not query:
                    print("Please enter a question")
                    continue
                
                self.search_results = []
                self.scraped_pages = []
                self.cleaned_query = ""
                
                result = self.process_query(query, self.current_domain, top_k=15)
                
                if result:
                    print("Response generated and saved successfully!")
                else:
                    print("No results found or error occurred")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def create_bot(groq_api_key=None, model_name="openai/gpt-oss-20b", embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    return MultiDomainSearchBot(
        groq_api_key=groq_api_key,
        model_name=model_name,
        embedding_model_name=embedding_model_name
    )

def search_domain(query, domain, groq_api_key=None, top_k=15):
    bot = create_bot(groq_api_key=groq_api_key)
    return bot.process_query(query, domain, top_k=top_k)

def get_all_domains():
    bot = create_bot()
    return bot.get_available_domains()

def main():
    parser = argparse.ArgumentParser(description='Multi-Domain Search Bot')
    parser.add_argument('--query', '-q', help='Specific question to ask')
    parser.add_argument('--domain', '-d', help='Domain to search')
    parser.add_argument('--groq-key', '-k', help='Groq API key')
    parser.add_argument('--ai-model', '-m', default='openai/gpt-oss-20b')
    parser.add_argument('--top-k', '-t', type=int, default=15)
    parser.add_argument('--interactive', '-i', action='store_true')
    
    args = parser.parse_args()
    
    try:
        print("Initializing Multi-Domain Search Bot...")
        
        bot = MultiDomainSearchBot(
            groq_api_key=args.groq_key,
            model_name=args.ai_model
        )
        
        if args.interactive or not args.query:
            bot.interactive_mode()
        else:
            if not args.domain:
                available_domains = bot.get_available_domains()
                if not available_domains:
                    print("No databases found")
                    return 1
                domain = available_domains[0]
                print(f"Using default domain: {domain}")
            else:
                domain = args.domain
            
            result = bot.process_query(args.query, domain, args.top_k)
            if result:
                print("Query processed successfully")
            else:
                print("Query processing failed")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Critical error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
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

class EnhancedUniMiBot:
    def __init__(self, groq_api_key=None, model_name="openai/gpt-oss-20b", embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        # Use the API key from the original file
        self.groq_api_key = groq_api_key or 'API'
        self.model_name = model_name
        self.groq_client = None
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.database = None
        self.database_info = None
        self.scraping_database = {}
        self.scraping_db_path = "unimi_scraping_database.json"
        self.delay = 0.3
        self.timeout = 15
        self.max_content_length = 8000
        self.current_query = ""
        self.cleaned_query = ""
        self.search_results = []
        self.scraped_pages = []
        
        logging.basicConfig(level=logging.ERROR)
        self.logger = logging.getLogger(__name__)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'it-IT,it;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
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
            print("🤖 Loading embedding model...")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print("✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            raise
    
    def init_groq_client(self):
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY required")
        
        try:
            self.groq_client = Groq(api_key=self.groq_api_key)
            print("✅ Groq client initialized")
        except Exception as e:
            print(f"❌ Error initializing Groq client: {e}")
            raise
    
    def init_tokenizer(self):
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def clean_user_query(self, user_query):
        """Clean and optimize user query using AI for better semantic search"""
        try:
            print(f"🧹 Cleaning query: '{user_query}'")
            
            cleaning_prompt = f"""Sei un esperto di ottimizzazione delle query per ricerca semantica universitaria.
            
L'utente ha fatto questa domanda: "{user_query}"

Il tuo compito è estrarre SOLO le parole chiave essenziali per una ricerca semantica efficace su un database universitario (Università Statale di Milano).

REGOLE:
1. Rimuovi parole di cortesia ("potresti", "riusciresti", "per favore", ecc.)
2. Rimuovi articoli, preposizioni e congiunzioni non essenziali
3. Mantieni solo i concetti chiave e i termini tecnici
4. Converti forme verbali in sostantivi quando possibile
5. Mantieni i termini specifici dell'università (corsi, servizi, procedure)
6. Massimo 8-10 parole chiave

ESEMPI:
- "riusciresti a trovare le pagine per andare all'estero con l'università" → "andare estero università programmi internazionali"
- "come faccio a iscrivermi al corso di laurea in informatica" → "iscrizione corso laurea informatica"
- "dove posso trovare informazioni sui tirocini curriculari" → "tirocini curriculari informazioni"
- "quali sono i requisiti per la borsa di studio" → "requisiti borsa studio"

Rispondi SOLO con le parole chiave ottimizzate, separate da spazi. Niente altro."""

            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Sei un esperto di ottimizzazione query per ricerca semantica universitaria. Estrai solo le parole chiave essenziali."
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
            
            # Basic validation and fallback
            if not cleaned_query or len(cleaned_query.split()) < 2:
                print("⚠️ AI cleaning failed, using basic text processing")
                # Fallback to basic text processing
                cleaned_query = self.basic_query_cleaning(user_query)
            
            print(f"✨ Cleaned query: '{cleaned_query}'")
            self.cleaned_query = cleaned_query
            return cleaned_query
            
        except Exception as e:
            print(f"⚠️ Error in AI query cleaning: {e}")
            print("🔄 Falling back to basic text processing")
            cleaned_query = self.basic_query_cleaning(user_query)
            self.cleaned_query = cleaned_query
            return cleaned_query
    
    def basic_query_cleaning(self, user_query):
        """Fallback basic text processing for query cleaning"""
        # Remove common courtesy words and phrases
        courtesy_words = [
            'potresti', 'riusciresti', 'puoi', 'riesci', 'per favore', 'grazie',
            'vorrei', 'volevo', 'mi servirebbe', 'mi serve', 'ho bisogno',
            'come faccio', 'dove posso', 'è possibile', 'si può'
        ]
        
        # Remove articles, prepositions, conjunctions
        stop_words = [
            'il', 'la', 'lo', 'le', 'gli', 'i', 'un', 'una', 'uno', 'del', 'della',
            'dei', 'delle', 'degli', 'al', 'alla', 'alle', 'agli', 'dal', 'dalla',
            'dalle', 'dagli', 'nel', 'nella', 'nelle', 'negli', 'sul', 'sulla',
            'sulle', 'sugli', 'per', 'con', 'tra', 'fra', 'di', 'da', 'in', 'su',
            'a', 'e', 'o', 'ma', 'però', 'quindi', 'anche', 'ancora', 'già'
        ]
        
        text = user_query.lower()
        
        # Remove courtesy phrases
        for phrase in courtesy_words:
            text = text.replace(phrase, ' ')
        
        # Split into words and filter
        words = text.split()
        filtered_words = []
        
        for word in words:
            # Remove punctuation
            word = re.sub(r'[^\w\s]', '', word)
            # Keep if not a stop word and length > 2
            if word and word not in stop_words and len(word) > 2:
                filtered_words.append(word)
        
        return ' '.join(filtered_words[:10])  # Limit to 10 words
    
    def load_scraping_database(self):
        try:
            if os.path.exists(self.scraping_db_path):
                with open(self.scraping_db_path, 'r', encoding='utf-8') as f:
                    self.scraping_database = json.load(f)
                print(f"📚 Loaded scraping database with {len(self.scraping_database.get('pages', {}))} cached pages")
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
            print(f"⚠️ Error loading scraping database: {e}")
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
    
    def find_database_file(self, database_path=None):
        if database_path and os.path.exists(database_path):
            return database_path
        
        # Updated patterns to include the new enhanced database
        patterns = [
            'unimi_enhanced_database_*.json',
            'unimi_fast_database_*.json', 
            'unimi_vector_database_*.json'
        ]
        
        database_files = []
        for pattern in patterns:
            database_files.extend(glob.glob(pattern))
        
        if not database_files:
            raise FileNotFoundError("No UniMi database found. Please run the enhanced crawler first.")
        
        latest_file = max(database_files, key=os.path.getctime)
        print(f"📂 Using database: {latest_file}")
        return latest_file
    
    def load_database(self, database_path=None):
        try:
            db_file = self.find_database_file(database_path)
            
            print(f"📖 Loading database from {db_file}...")
            with open(db_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.database_info = data.get('database_info', {})
            self.database = data.get('pages', [])
            
            if not self.database:
                raise ValueError("Database contains no pages")
            
            # Verify database structure for enhanced database
            sample_page = self.database[0]
            required_fields = ['id', 'url', 'title', 'embedding']
            missing_fields = [field for field in required_fields if field not in sample_page]
            
            if missing_fields:
                raise ValueError(f"Database missing required fields: {missing_fields}")
            
            embedding_dimension = len(sample_page['embedding']) if sample_page.get('embedding') else 0
            
            # Check if it's the enhanced database with links
            has_enhanced_features = 'links' in sample_page
            
            print(f"✅ Database loaded successfully:")
            print(f"   - Type: {'Enhanced' if has_enhanced_features else 'Standard'} Database")
            print(f"   - Pages: {len(self.database)}")
            print(f"   - Domain: {self.database_info.get('domain', 'unknown')}")
            print(f"   - Embedding dimension: {embedding_dimension}")
            print(f"   - Version: {self.database_info.get('version', 'unknown')}")
            print(f"   - Created: {self.database_info.get('created_at', 'unknown')}")
            
            if has_enhanced_features:
                # Show enhanced features statistics
                total_internal_links = sum(len(page.get('links', {}).get('internal_links', [])) for page in self.database)
                total_internal_pdfs = sum(len(page.get('links', {}).get('internal_pdfs', [])) for page in self.database)
                print(f"   - Total internal links: {total_internal_links}")
                print(f"   - Total internal PDFs: {total_internal_pdfs}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            return False
    
    def create_query_embedding(self, query):
        try:
            if not query.strip():
                return None
            
            embedding = self.embedding_model.encode(query, normalize_embeddings=True)
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Error creating query embedding: {e}")
            return None
    
    def semantic_search(self, query, top_k=15):
        if not self.database:
            raise ValueError("Database not loaded")
        
        print(f"🔍 Original query: '{query}'")
        
        # Clean the query using AI
        cleaned_query = self.clean_user_query(query)
        print(f"🎯 Searching with cleaned query: '{cleaned_query}'")
        
        self.current_query = query  # Keep original for response generation
        
        query_embedding = self.create_query_embedding(cleaned_query)
        if query_embedding is None:
            raise ValueError("Unable to create query embedding")
        
        embeddings_matrix = []
        valid_pages = []
        
        # Extract embeddings from database pages
        for page in self.database:
            if 'embedding' in page and page['embedding']:
                try:
                    # Handle both list and numpy array formats
                    embedding = np.array(page['embedding'], dtype=np.float32)
                    embeddings_matrix.append(embedding)
                    valid_pages.append(page)
                except Exception as e:
                    print(f"⚠️ Skipping page {page.get('id', 'unknown')} - invalid embedding: {e}")
                    continue
        
        if not embeddings_matrix:
            raise ValueError("No valid embeddings found in database")
        
        print(f"📊 Comparing query against {len(embeddings_matrix)} page embeddings...")
        
        embeddings_matrix = np.array(embeddings_matrix)
        query_vector = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, embeddings_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices, 1):
            similarity_score = similarities[idx]
            page = valid_pages[idx]
            
            # Extract content information based on enhanced database structure
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
                # Include enhanced database link information
                'links': page.get('links', {})
            }
            results.append(result)
        
        print(f"✅ Found {len(results)} relevant pages")
        print(f"   - High relevance (>0.3): {sum(1 for r in results if r['similarity_score'] > 0.3)}")
        print(f"   - Medium relevance (0.1-0.3): {sum(1 for r in results if 0.1 <= r['similarity_score'] <= 0.3)}")
        print(f"   - Low relevance (<0.1): {sum(1 for r in results if r['similarity_score'] < 0.1)}")
        
        self.search_results = results
        return results
    
    def extract_page_content_info(self, page):
        """Extract content information from enhanced database structure"""
        content_info = {
            'content_length': 0,
            'preview': '',
            'has_pdf': False,
            'pdf_count': 0,
            'internal_links_count': 0,
            'internal_pdfs_count': 0,
            'main_content': ''
        }
        
        # Handle enhanced database structure
        if 'content' in page and isinstance(page['content'], dict):
            content_data = page['content']
            content_info['content_length'] = content_data.get('content_length', 0)
            content_info['main_content'] = content_data.get('main_content', '')
            content_info['preview'] = content_data.get('main_content', '')[:300] + '...' if content_data.get('main_content') else ''
            content_info['pdf_count'] = content_data.get('pdfs_count', 0)
            content_info['has_pdf'] = content_data.get('pdfs_count', 0) > 0 or bool(content_data.get('pdf_content'))
            content_info['internal_links_count'] = content_data.get('internal_links_count', 0)
            content_info['internal_pdfs_count'] = content_data.get('internal_pdfs_count', 0)
        
        # Handle legacy structure
        elif 'content_preview' in page:
            content_info['preview'] = page['content_preview'][:300] + '...' if page['content_preview'] else ''
            content_info['content_length'] = page.get('content_length', 0)
        
        # Handle direct content field
        elif 'main_content' in page:
            content_info['main_content'] = page['main_content']
            content_info['content_length'] = len(page['main_content'])
            content_info['preview'] = page['main_content'][:300] + '...' if page['main_content'] else ''
        
        return content_info
    
    def get_database_content(self, url):
        """Get content directly from the vector database if available"""
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
                        # Include enhanced database features
                        'links': page.get('links', {}),
                        'timing': page.get('timing', {})
                    }
        return None
    
    def extract_page_content(self, soup, url):
        try:
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript', 'comment']):
                element.decompose()
            
            main_content = ""
            
            # Try different selectors for main content
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
            
            # Fallback to body content
            if not main_content:
                body = soup.find('body')
                if body:
                    main_content = body.get_text(separator=' ', strip=True)
                else:
                    main_content = soup.get_text(separator=' ', strip=True)
            
            # Clean up content
            main_content = re.sub(r'\s+', ' ', main_content).strip()
            
            # Limit content length
            if len(main_content) > self.max_content_length:
                main_content = main_content[:self.max_content_length] + "..."
            
            return main_content
            
        except Exception as e:
            print(f"❌ Error extracting content from {url}: {e}")
            return ""
    
    def scrape_single_page(self, url):
        try:
            print(f"🌐 Scraping: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if 'text/html' not in content_type.lower():
                print(f"⚠️ Skipping non-HTML content: {content_type}")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract metadata
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            description = ""
            desc_tag = soup.find('meta', attrs={'name': 'description'})
            if desc_tag:
                description = desc_tag.get('content', '').strip()
            
            # Extract main content
            content = self.extract_page_content(soup, url)
            
            if content:
                print(f"✅ Scraped {len(content)} characters from {url}")
                return {
                    'url': url,
                    'title': title,
                    'description': description,
                    'content': content,
                    'content_length': len(content),
                    'scraped_at': datetime.now().isoformat()
                }
            else:
                print(f"⚠️ No content extracted from {url}")
            
            return None
            
        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")
            return None
    
    def get_page_from_database(self, url):
        """Get cached page content from scraping database"""
        return self.scraping_database['pages'].get(url)
    
    def update_scraping_database(self, url, page_data):
        """Update scraping database with new page data"""
        self.scraping_database['pages'][url] = page_data
        self.save_scraping_database()
    
    def scrape_top_pages(self):
        if not self.search_results:
            raise ValueError("No search results available")
        
        print(f"📄 Processing {len(self.search_results)} pages...")
        self.scraped_pages = []
        
        for result in self.search_results:
            url = result['url']
            
            # Priority order: database content -> cached content -> fresh scrape
            page_data = None
            
            # 1. Try to get content from vector database first
            page_data = self.get_database_content(url)
            if page_data:
                page_data['source'] = 'vector_database'
                print(f"📚 Using vector database content for {url}")
            
            # 2. Try cached content
            if not page_data:
                page_data = self.get_page_from_database(url)
                if page_data:
                    page_data['source'] = 'cached'
                    print(f"💾 Using cached content for {url}")
            
            # 3. Fresh scrape as last resort
            if not page_data:
                page_data = self.scrape_single_page(url)
                if page_data:
                    page_data['source'] = 'fresh_scrape'
                    self.update_scraping_database(url, page_data)
                    
                    # Add delay only for fresh scrapes
                    if self.delay > 0:
                        time.sleep(self.delay)
            
            # Add search metadata to page data
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
                print(f"❌ Failed to get content for {url}")
        
        print(f"✅ Successfully processed {len(self.scraped_pages)} pages")
        return len(self.scraped_pages)
    
    def count_tokens(self, text):
        try:
            return len(self.tokenizer.encode(text))
        except:
            return int(len(text.split()) * 1.3)
    
    def create_latex_ai_prompt(self):
        # Sort pages by similarity score (highest first)
        sorted_pages = sorted(self.scraped_pages, key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        prompt = f"""Sei un assistente esperto dell'Università Statale di Milano (UniMi). 
Un utente ha fatto questa domanda: "{self.current_query}"
La query è stata ottimizzata per la ricerca semantica in: "{self.cleaned_query}"

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
        
        # Calculate token usage
        total_tokens = self.count_tokens(prompt)
        max_tokens = 7000  # Leave room for response
        
        pages_added = 0
        for page in sorted_pages:
            if pages_added >= 12:  # Limit number of pages
                break
                
            # Include enhanced database information
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
                print(f"⚠️ Token limit reached, using {pages_added} pages")
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
        
        print(f"📝 Generated prompt with {total_tokens} tokens using {pages_added} pages")
        return prompt
    
    def generate_ai_response(self, prompt):
        try:
            print("🤖 Generating AI response...")
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
                max_tokens=3500,
                temperature=0.1,
                top_p=1,
                stream=False
            )
            
            ai_response = response.choices[0].message.content
            print(f"✅ Generated response ({len(ai_response)} characters)")
            return ai_response
            
        except Exception as e:
            print(f"❌ Error generating AI response: {e}")
            raise
    
    def save_results(self, ai_response):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Clean query for filename
        clean_query = re.sub(r'[^\w\s-]', '', self.current_query)[:50]
        clean_query = re.sub(r'\s+', '_', clean_query)
        
        # Save LaTeX file
        latex_filename = f'unimi_enhanced_response_{clean_query}_{timestamp}.tex'
        with open(latex_filename, 'w', encoding='utf-8') as f:
            f.write(ai_response)
        
        # Save detailed text report with enhanced information
        txt_filename = f'unimi_enhanced_response_{clean_query}_{timestamp}.txt'
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("UNIVERSITÀ STATALE DI MILANO - RISPOSTA AI ENHANCED\n")
            f.write("=" * 60 + "\n")
            f.write(f"DOMANDA ORIGINALE: {self.current_query}\n")
            f.write(f"QUERY OTTIMIZZATA: {self.cleaned_query}\n")
            f.write(f"DATA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"PAGINE TROVATE: {len(self.search_results)}\n")
            f.write(f"PAGINE ANALIZZATE: {len(self.scraped_pages)}\n")
            f.write(f"DATABASE: {self.database_info.get('name', 'Unknown')}\n")
            f.write(f"VERSIONE DB: {self.database_info.get('version', 'Unknown')}\n")
            f.write(f"MODELLO EMBEDDING: {self.embedding_model_name}\n")
            f.write(f"MODELLO AI: {self.model_name}\n\n")
            
            # Enhanced database statistics
            if self.database_info.get('features'):
                f.write("CARATTERISTICHE DATABASE:\n")
                for feature in self.database_info['features']:
                    f.write(f"• {feature}\n")
                f.write("\n")
            
            f.write("PAGINE PIÙ RILEVANTI:\n")
            f.write("-" * 40 + "\n")
            for page in sorted(self.scraped_pages, key=lambda x: x.get('similarity_score', 0), reverse=True)[:10]:
                f.write(f"• {page['title']} (Score: {page.get('similarity_score', 0):.4f})\n")
                f.write(f"  {page['url']}\n")
                f.write(f"  Source: {page.get('source', 'unknown')}\n")
                
                # Include enhanced database information
                if page.get('internal_links_count', 0) > 0:
                    f.write(f"  Link interni: {page['internal_links_count']}\n")
                if page.get('internal_pdfs_count', 0) > 0:
                    f.write(f"  PDF interni: {page['internal_pdfs_count']}\n")
                if page.get('timing'):
                    timing = page['timing']
                    f.write(f"  Timing: scrape {timing.get('scraping_time', 0):.3f}s, embed {timing.get('embedding_time', 0):.3f}s\n")
                f.write("\n")
            
            # Query cleaning analysis
            f.write("ANALISI QUERY CLEANING:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Query originale: {self.current_query}\n")
            f.write(f"Query ottimizzata: {self.cleaned_query}\n")
            f.write(f"Parole rimosse: {len(self.current_query.split()) - len(self.cleaned_query.split())}\n")
            f.write(f"Efficacia pulizia: {(1 - len(self.cleaned_query)/len(self.current_query))*100:.1f}% riduzione\n\n")
            
            # Search performance analysis
            high_relevance = sum(1 for r in self.search_results if r['similarity_score'] > 0.3)
            medium_relevance = sum(1 for r in self.search_results if 0.1 <= r['similarity_score'] <= 0.3)
            low_relevance = sum(1 for r in self.search_results if r['similarity_score'] < 0.1)
            
            f.write("PERFORMANCE RICERCA:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Alta rilevanza (>0.3): {high_relevance}\n")
            f.write(f"Media rilevanza (0.1-0.3): {medium_relevance}\n")
            f.write(f"Bassa rilevanza (<0.1): {low_relevance}\n")
            f.write(f"Score medio: {np.mean([r['similarity_score'] for r in self.search_results]):.4f}\n\n")
            
            f.write("RISPOSTA LaTeX:\n")
            f.write("=" * 60 + "\n\n")
            f.write(ai_response)
        
        print(f"💾 Results saved:")
        print(f"   - LaTeX: {latex_filename}")
        print(f"   - Enhanced Report: {txt_filename}")
        
        return latex_filename, txt_filename
    
    def process_query(self, query, database_path=None, top_k=15):
        try:
            print(f"\n🎯 Processing query: '{query}'")
            
            # Load database if not already loaded
            if not self.database:
                print("📖 Loading database...")
                if not self.load_database(database_path):
                    raise ValueError("Unable to load database")
            
            # Perform semantic search with AI query cleaning
            search_results = self.semantic_search(query, top_k)
            
            if not search_results:
                print("❌ No relevant pages found")
                return None
            
            # Scrape/get content for top results
            scraped_count = self.scrape_top_pages()
            
            if scraped_count == 0:
                print("❌ No content could be retrieved")
                return None
            
            # Generate AI response
            prompt = self.create_latex_ai_prompt()
            ai_response = self.generate_ai_response(prompt)
            
            # Save results with enhanced information
            files = self.save_results(ai_response)
            
            print(f"✅ Query processing completed successfully!")
            print(f"   - Original query: '{self.current_query}'")
            print(f"   - Cleaned query: '{self.cleaned_query}'")
            print(f"   - Pages found: {len(self.search_results)}")
            print(f"   - Pages processed: {len(self.scraped_pages)}")
            
            return ai_response
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            return None
    
    def interactive_mode(self, database_path=None):
        print("\n" + "="*60)
        print("🎓 Enhanced UniMi Search Bot - Interactive Mode")
        print("="*60)
        print("Features:")
        print("  - AI-powered query cleaning for better semantic search")
        print("  - Enhanced database with links and timing information")
        print("  - Improved LaTeX response generation")
        print("Commands:")
        print("  - Type your question in Italian")
        print("  - 'reload' to reload database")
        print("  - 'stats' to show database statistics")
        print("  - 'test-clean <query>' to test query cleaning")
        print("  - 'quit' to exit")
        print("="*60)
        
        # Load database
        if not self.database:
            print("📖 Loading database...")
            if not self.load_database(database_path):
                print("❌ Error: Unable to load database")
                return
        
        while True:
            try:
                query = input("\n🤔 Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if query.lower() == 'reload':
                    print("🔄 Reloading database...")
                    if self.load_database(database_path):
                        print("✅ Database reloaded")
                    else:
                        print("❌ Failed to reload database")
                    continue
                
                if query.lower() == 'stats':
                    print(f"\n📊 Enhanced Database Statistics:")
                    print(f"   - Pages: {len(self.database)}")
                    print(f"   - Domain: {self.database_info.get('domain', 'unknown')}")
                    print(f"   - Version: {self.database_info.get('version', 'unknown')}")
                    print(f"   - Created: {self.database_info.get('created_at', 'unknown')}")
                    print(f"   - Cached pages: {len(self.scraping_database.get('pages', {}))}")
                    
                    if self.database_info.get('features'):
                        print("   - Features:")
                        for feature in self.database_info['features']:
                            print(f"     • {feature}")
                    
                    # Calculate enhanced statistics
                    if len(self.database) > 0:
                        total_internal_links = sum(len(page.get('links', {}).get('internal_links', [])) for page in self.database)
                        total_internal_pdfs = sum(len(page.get('links', {}).get('internal_pdfs', [])) for page in self.database)
                        pages_with_timing = sum(1 for page in self.database if page.get('timing'))
                        
                        print(f"   - Total internal links: {total_internal_links}")
                        print(f"   - Total internal PDFs: {total_internal_pdfs}")
                        print(f"   - Pages with timing data: {pages_with_timing}")
                    
                    continue
                
                if query.lower().startswith('test-clean '):
                    test_query = query[11:].strip()
                    if test_query:
                        print(f"🧹 Testing query cleaning:")
                        print(f"   Original: '{test_query}'")
                        cleaned = self.clean_user_query(test_query)
                        print(f"   Cleaned:  '{cleaned}'")
                        print(f"   Reduction: {(1 - len(cleaned)/len(test_query))*100:.1f}%")
                    continue
                
                if not query:
                    print("⚠️ Please enter a question")
                    continue
                
                # Reset state
                self.search_results = []
                self.scraped_pages = []
                self.cleaned_query = ""
                
                # Process query with AI cleaning
                result = self.process_query(query, top_k=15)
                
                if result:
                    print("✅ Response generated and saved successfully!")
                    print(f"📈 Query cleaning effectiveness: {(1 - len(self.cleaned_query)/len(query))*100:.1f}% reduction")
                else:
                    print("❌ No results found or error occurred")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced UniMi Search Bot with AI Query Cleaning')
    parser.add_argument('--database', '-d', help='Path to enhanced vector database file')
    parser.add_argument('--query', '-q', help='Specific question to ask')
    parser.add_argument('--groq-key', '-k', help='Groq API key (defaults to embedded key)')
    parser.add_argument('--ai-model', '-m', default='openai/gpt-oss-20b',
                       choices=['openai/gpt-oss-20b', 'mixtral-8x7b-32768', 'openai/gpt-oss-20b'])
    parser.add_argument('--embedding-model', '-e', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    parser.add_argument('--top-k', '-t', type=int, default=15, help='Number of pages to retrieve')
    parser.add_argument('--delay', type=float, default=0.3, help='Delay between requests (seconds)')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--max-content', type=int, default=8000, help='Maximum content length per page')
    parser.add_argument('--no-query-cleaning', action='store_true', help='Disable AI query cleaning')
    
    args = parser.parse_args()
    
    try:
        print("🚀 Initializing Enhanced UniMi Search Bot...")
        print("   Features: AI Query Cleaning, Enhanced Database Support, Improved LaTeX Generation")
        
        bot = EnhancedUniMiBot(
            groq_api_key=args.groq_key,
            model_name=args.ai_model,
            embedding_model_name=args.embedding_model
        )
        
        # Set configuration
        bot.delay = args.delay
        bot.max_content_length = args.max_content
        
        if args.interactive or not args.query:
            bot.interactive_mode(args.database)
        else:
            print(f"🔍 Processing single query: '{args.query}'")
            result = bot.process_query(args.query, args.database, args.top_k)
            if result:
                print("✅ Query processed successfully")
                print(f"📊 Original query: '{bot.current_query}'")
                print(f"🎯 Cleaned query: '{bot.cleaned_query}'")
                print("\nFirst 200 characters of response:")
                print("-" * 50)
                print(result[:200] + "..." if len(result) > 200 else result)
            else:
                print("❌ Query processing failed")
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

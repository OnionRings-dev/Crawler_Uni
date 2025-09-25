#!/usr/bin/env python3
"""
Crawler UniMi per Creazione Database Vettoriale
Crawla link da unimi.it e crea un database JSON con vettori associati
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
import time
import re
import json
from collections import deque
from datetime import datetime
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Set
import random

class UniMiVectorDatabaseCreator:
    def __init__(self, delay=0.3, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.delay = delay
        self.target_pages = 300
        self.base_url = 'https://unimi.it'
        self.domain = urlparse(self.base_url).netloc
        
        self.pages_database = []
        self.embedding_model = None
        
        self.stats = {
            'total_crawled': 0,
            'total_failed': 0,
            'start_time': datetime.now()
        }
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; UniMiBot/1.0)',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'it-IT,it;q=0.9,en;q=0.8'
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        self.init_embedding_model(model_name)
        
    def init_embedding_model(self, model_name):
        try:
            self.logger.info(f"Caricamento modello embedding: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info("Modello embedding caricato con successo")
        except Exception as e:
            self.logger.error(f"Errore nel caricare il modello embedding: {e}")
            raise
    
    def normalize_url(self, url):
        try:
            parsed = urlparse(url)
            normalized = urlunparse((
                parsed.scheme or 'https',
                parsed.netloc,
                parsed.path or '/',
                parsed.params,
                parsed.query,
                ''
            ))
            
            if len(parsed.path) > 1 and normalized.endswith('/'):
                normalized = normalized[:-1]
                
            return normalized
        except:
            return url
    
    def is_valid_url(self, url):
        if not url:
            return False
            
        try:
            parsed = urlparse(url)
        except:
            return False
        
        valid_domain = (parsed.netloc == self.domain or 
                       parsed.netloc.endswith('.' + self.domain))
        
        if not valid_domain:
            return False
        
        skip_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.zip', '.jpg', '.png', '.gif', '.mp4', '.avi'}
        if any(parsed.path.lower().endswith(ext) for ext in skip_extensions):
            return False
        
        return True
    
    def extract_text_content(self, soup):
        # Rimuovi elementi non utili
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        main_content = ""
        
        # Cerca il contenuto principale
        main_selectors = ['main', 'article', '.content', '#content', '.main-content', '.entry-content']
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
        
        return main_content[:4000]  # Limita la lunghezza
    
    def create_embedding(self, text):
        try:
            if not text.strip():
                return None
            
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            return embedding.astype(np.float32).tolist()  # Converti in lista per JSON
            
        except Exception as e:
            self.logger.error(f"Errore nella creazione embedding: {e}")
            return None
    
    def extract_links(self, url, soup):
        new_links = set()
        
        try:
            all_links = soup.find_all('a', href=True)
            
            for link in all_links:
                href = link.get('href')
                if not href:
                    continue
                
                try:
                    absolute_url = urljoin(url, href)
                    normalized_url = self.normalize_url(absolute_url)
                    
                    if self.is_valid_url(normalized_url):
                        new_links.add(normalized_url)
                
                except Exception:
                    continue
        
        except Exception as e:
            self.logger.error(f"Errore nell'estrazione link da {url}: {e}")
        
        return new_links
    
    def crawl_page(self, url):
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if 'text/html' not in content_type.lower():
                return None, set()
            
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
            
            keywords = ""
            keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
            if keywords_tag:
                keywords = keywords_tag.get('content', '').strip()
            
            content = self.extract_text_content(soup)
            
            # Crea testo combinato per l'embedding
            combined_text = f"{title}. {description}. {keywords}. {content}".strip()
            
            # Crea embedding
            embedding = self.create_embedding(combined_text)
            
            if embedding is not None:
                # Crea record per il database
                page_record = {
                    'id': len(self.pages_database) + 1,
                    'url': url,
                    'title': title,
                    'description': description,
                    'keywords': keywords,
                    'content_length': len(content),
                    'content_preview': content[:500],  # Solo anteprima del contenuto
                    'embedding': embedding,
                    'crawled_at': datetime.now().isoformat(),
                    'domain': self.domain
                }
                
                self.pages_database.append(page_record)
                self.stats['total_crawled'] += 1
                
                new_links = self.extract_links(url, soup)
                return page_record, new_links
            
            return None, set()
            
        except Exception as e:
            self.logger.error(f"Errore crawling {url}: {e}")
            self.stats['total_failed'] += 1
            return None, set()
    
    def crawl_unimi(self):
        print(f"\n🎯 === CREAZIONE DATABASE VETTORIALE UNIMI ===")
        print(f"Target: {self.target_pages} pagine da {self.base_url}")
        
        visited_urls = set()
        url_queue = deque([self.base_url])
        found_urls = set([self.base_url])
        
        pages_crawled = 0
        attempts = 0
        max_attempts = self.target_pages * 4
        
        while url_queue and pages_crawled < self.target_pages and attempts < max_attempts:
            attempts += 1
            current_url = url_queue.popleft()
            
            if current_url in visited_urls:
                continue
            
            visited_urls.add(current_url)
            
            if pages_crawled % 10 == 0:
                print(f"[{pages_crawled}/{self.target_pages}] Progresso crawling...")
            
            page_record, new_links = self.crawl_page(current_url)
            
            if page_record:
                pages_crawled += 1
                if pages_crawled % 50 == 0:
                    print(f"✅ Aggiunte {pages_crawled} pagine al database")
            
            # Aggiungi nuovi link alla coda (limitati per evitare esplosione)
            limited_new_links = list(new_links - found_urls)[:15]
            random.shuffle(limited_new_links)
            
            for link in limited_new_links:
                if link not in found_urls:
                    url_queue.append(link)
                    found_urls.add(link)
            
            if self.delay > 0:
                time.sleep(self.delay)
        
        print(f"✅ Completato crawling: {len(self.pages_database)} pagine nel database")
        return len(self.pages_database)
    
    def save_vector_database(self):
        print(f"\n💾 === SALVATAGGIO DATABASE VETTORIALE ===")
        
        # Crea metadata del database
        database_metadata = {
            'database_info': {
                'name': 'UniMi Vector Database',
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'domain': self.domain,
                'total_pages': len(self.pages_database),
                'embedding_model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                'embedding_dimension': len(self.pages_database[0]['embedding']) if self.pages_database else 0,
                'crawling_stats': self.stats
            },
            'pages': self.pages_database
        }
        
        # Salva database completo
        database_filename = f'unimi_vector_database_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(database_filename, 'w', encoding='utf-8') as f:
            json.dump(database_metadata, f, ensure_ascii=False, indent=2, default=str)
        
        # Salva versione compatta (senza content_preview)
        compact_database = {
            'database_info': database_metadata['database_info'],
            'pages': [
                {
                    'id': page['id'],
                    'url': page['url'],
                    'title': page['title'],
                    'description': page['description'],
                    'keywords': page['keywords'],
                    'content_length': page['content_length'],
                    'embedding': page['embedding'],
                    'crawled_at': page['crawled_at']
                }
                for page in self.pages_database
            ]
        }
        
        compact_filename = f'unimi_vector_database_compact_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(compact_filename, 'w', encoding='utf-8') as f:
            json.dump(compact_database, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ Database salvato in:")
        print(f"  - {database_filename} (versione completa con anteprime)")
        print(f"  - {compact_filename} (versione compatta)")
        
        # Salva anche un indice delle URL per riferimento rapido
        url_index = {
            'database_info': database_metadata['database_info'],
            'url_index': [
                {
                    'id': page['id'],
                    'url': page['url'],
                    'title': page['title']
                }
                for page in self.pages_database
            ]
        }
        
        index_filename = f'unimi_url_index_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(index_filename, 'w', encoding='utf-8') as f:
            json.dump(url_index, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"  - {index_filename} (indice URL)")
        
        return database_filename, compact_filename, index_filename
    
    def print_database_stats(self):
        if not self.pages_database:
            print("❌ Database vuoto")
            return
        
        print(f"\n📊 === STATISTICHE DATABASE ===")
        print(f"Pagine totali: {len(self.pages_database)}")
        print(f"Dimensione embedding: {len(self.pages_database[0]['embedding'])}")
        
        # Statistiche sui contenuti
        avg_content_length = sum(page['content_length'] for page in self.pages_database) / len(self.pages_database)
        pages_with_description = sum(1 for page in self.pages_database if page['description'])
        pages_with_keywords = sum(1 for page in self.pages_database if page['keywords'])
        
        print(f"Lunghezza media contenuto: {avg_content_length:.0f} caratteri")
        print(f"Pagine con descrizione: {pages_with_description} ({pages_with_description/len(self.pages_database)*100:.1f}%)")
        print(f"Pagine con keywords: {pages_with_keywords} ({pages_with_keywords/len(self.pages_database)*100:.1f}%)")
        
        # Top 5 pagine più lunghe
        longest_pages = sorted(self.pages_database, key=lambda x: x['content_length'], reverse=True)[:5]
        print(f"\nTop 5 pagine per contenuto:")
        for i, page in enumerate(longest_pages, 1):
            print(f"  {i}. {page['title'][:50]}... ({page['content_length']} char)")
    
    def run(self):
        print("🚀 === AVVIO CREATORE DATABASE VETTORIALE UNIMI ===")
        print(f"Target: {self.target_pages} pagine da unimi.it")
        
        crawled_count = self.crawl_unimi()
        
        if crawled_count > 0:
            database_files = self.save_vector_database()
            self.print_database_stats()
            self.print_summary()
        else:
            print("❌ Nessuna pagina crawlata con successo")
    
    def print_summary(self):
        duration = datetime.now() - self.stats['start_time']
        
        print("\n" + "=" * 60)
        print("🎯 RIASSUNTO CREAZIONE DATABASE VETTORIALE")
        print("=" * 60)
        print(f"Pagine nel database: {len(self.pages_database)}")
        print(f"Fallimenti: {self.stats['total_failed']}")
        print(f"Durata: {duration}")
        print(f"Velocità: {len(self.pages_database)/(duration.total_seconds()/60):.1f} pagine/minuto")
        print(f"Files generati:")
        print("  - unimi_vector_database_[timestamp].json (completo)")
        print("  - unimi_vector_database_compact_[timestamp].json (compatto)")
        print("  - unimi_url_index_[timestamp].json (indice)")
        print("=" * 60)


def main():
    print("🤖 Crawler UniMi - Creatore Database Vettoriale")
    print("=" * 60)
    
    crawler = UniMiVectorDatabaseCreator(delay=0.2)
    crawler.run()

if __name__ == "__main__":
    main()
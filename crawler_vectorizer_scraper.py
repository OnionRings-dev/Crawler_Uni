#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
import time
import re
import json
from collections import deque
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class UniversalDomainCrawler:
    def __init__(self, base_url, delay=0.1, max_workers=5, target_pages=1000, 
                 qdrant_path="./qdrant_data"):
        self.delay = delay
        self.max_workers = max_workers
        self.target_pages = target_pages
        self.base_url = base_url if base_url.startswith('http') else f'https://{base_url}'
        self.domain = urlparse(self.base_url).netloc
        
        self.pages_count = 0
        self.embedding_model = None
        self.lock = threading.Lock()
        
        self.qdrant_client = None
        self.collection_name = None
        self.init_qdrant(qdrant_path)
        
        self.stats = {
            'total_crawled': 0,
            'total_failed': 0,
            'pdfs_extracted': 0,
            'start_time': datetime.now()
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'it-IT,it;q=0.9,en;q=0.8',
            'Connection': 'keep-alive'
        }
        
        self.init_embedding_model()
    
    def init_qdrant(self, path):
        try:
            print("Inizializzazione Qdrant...")
            self.qdrant_client = QdrantClient(path=path)
            
            domain_clean = self.domain.replace('.', '_').replace('-', '_')
            self.collection_name = f"crawl_{domain_clean}"
            
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if not collection_exists:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Collection creata: {self.collection_name}")
            else:
                print(f"✅ Collection esistente: {self.collection_name}")
                
        except Exception as e:
            print(f"❌ Errore inizializzazione Qdrant: {e}")
            raise
        
    def init_embedding_model(self):
        try:
            print("Caricamento modello embedding...")
            self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            print("✅ Modello caricato")
        except Exception as e:
            print(f"❌ Errore caricamento modello: {e}")
            raise
    
    def create_session(self):
        session = requests.Session()
        session.headers.update(self.headers)
        return session
    
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
            return normalized.rstrip('/')
        except:
            return url
    
    def is_valid_url(self, url):
        if not url:
            return False
            
        try:
            parsed = urlparse(url)
        except:
            return False
        
        if not (parsed.netloc == self.domain or parsed.netloc.endswith('.' + self.domain)):
            return False
        
        skip_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.zip', '.jpg', '.png', '.gif', '.mp4', '.avi'}
        if any(parsed.path.lower().endswith(ext) for ext in skip_extensions):
            return False
        
        return True
    
    def sanitize_content(self, soup):
        for element in soup(['script', 'noscript', 'style', 'nav', 'footer', 'header', 'aside', 'comment']):
            element.decompose()
        
        for element in soup.find_all():
            for attr in list(element.attrs.keys()):
                if attr.startswith('on') or attr == 'style':
                    del element.attrs[attr]
        
        return soup
    
    def extract_fast_content(self, soup, url):
        self.sanitize_content(soup)
        
        content_data = {
            'main_content': '',
            'internal_links': [],
            'internal_pdfs': [],
            'all_links': [],
            'all_pdfs': []
        }
        
        main_selectors = ['main', 'article', '.content', '#content', '.main-content']
        main_content = ""
        
        for selector in main_selectors:
            main_elem = soup.select_one(selector)
            if main_elem:
                main_content = main_elem.get_text(separator=' ', strip=True)
                break
        
        if not main_content:
            body = soup.find('body')
            main_content = body.get_text(separator=' ', strip=True) if body else soup.get_text(separator=' ', strip=True)
        
        main_content = re.sub(r'\s+', ' ', main_content).strip()[:4000]
        content_data['main_content'] = main_content
        
        for link in soup.find_all('a', href=True, limit=100):
            href = link.get('href')
            if href:
                absolute_url = urljoin(url, href)
                link_text = link.get_text(strip=True)
                
                link_data = {
                    'url': absolute_url,
                    'text': link_text,
                    'title': link.get('title', ''),
                    'extracted_at': datetime.now().isoformat()
                }
                
                content_data['all_links'].append(link_data)
                
                if self.is_valid_url(absolute_url):
                    content_data['internal_links'].append(link_data)
                
                if href.lower().endswith('.pdf'):
                    pdf_data = {
                        'url': absolute_url,
                        'text': link_text,
                        'title': link.get('title', ''),
                        'extracted_at': datetime.now().isoformat()
                    }
                    
                    content_data['all_pdfs'].append(pdf_data)
                    
                    if self.is_internal_pdf(absolute_url):
                        content_data['internal_pdfs'].append(pdf_data)
        
        return content_data
    
    def is_internal_pdf(self, pdf_url):
        try:
            parsed = urlparse(pdf_url)
            return parsed.netloc == self.domain or parsed.netloc.endswith('.' + self.domain)
        except:
            return False
    
    def extract_pdf_text_fast(self, pdf_url, session):
        try:
            response = session.get(pdf_url, timeout=10, stream=True)
            if response.status_code == 200 and len(response.content) < 5000000:
                import PyPDF2
                import io
                
                pdf_file = io.BytesIO(response.content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                text = ""
                for page in pdf_reader.pages[:3]:
                    text += page.extract_text()[:1000]
                
                self.stats['pdfs_extracted'] += 1
                return text.strip()
        except:
            pass
        return ""
    
    def create_embedding(self, text):
        start_time = time.time()
        
        try:
            if not text.strip():
                return None, 0.0
            
            embedding = self.embedding_model.encode(text[:2000], normalize_embeddings=True)
            embedding_time = time.time() - start_time
            
            return embedding.astype(np.float32).tolist(), embedding_time
        except:
            return None, time.time() - start_time
    
    def extract_metadata_fast(self, soup):
        title = soup.find('title')
        title = title.get_text().strip() if title else ""
        
        desc_tag = soup.find('meta', attrs={'name': 'description'})
        description = desc_tag.get('content', '').strip() if desc_tag else ""
        
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        keywords = keywords_tag.get('content', '').strip() if keywords_tag else ""
        
        return title, description, keywords
    
    def save_to_qdrant(self, page_record):
        if not self.qdrant_client:
            return False
            
        try:
            payload = {
                'url': page_record['url'],
                'url_hash': page_record['url_hash'],
                'title': page_record['title'],
                'description': page_record['description'],
                'keywords': page_record['keywords'],
                'domain': page_record['domain'],
                'crawled_at': page_record['crawled_at'],
                'content_length': page_record['content']['content_length'],
                'internal_links_count': page_record['content']['internal_links_count'],
                'internal_pdfs_count': page_record['content']['internal_pdfs_count'],
                'links_json': json.dumps(page_record['links']),
                'main_content': page_record['content']['main_content'],
                'scraping_time': page_record['timing']['scraping_time'],
                'embedding_time': page_record['timing']['embedding_time']
            }
            
            point = PointStruct(
                id=page_record['id'],
                vector=page_record['embedding'],
                payload=payload
            )
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            return True
            
        except Exception as e:
            print(f"Errore salvataggio Qdrant: {e}")
            return False
    
    def process_single_url(self, url):
        scraping_start = time.time()
        session = self.create_session()
        
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if 'text/html' not in content_type.lower():
                return None, []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title, description, keywords = self.extract_metadata_fast(soup)
            content_data = self.extract_fast_content(soup, url)
            
            pdf_content = ""
            if content_data['internal_pdfs']:
                pdf_content = self.extract_pdf_text_fast(content_data['internal_pdfs'][0]['url'], session)
            
            scraping_time = time.time() - scraping_start
            
            combined_text = f"{title} {description} {keywords} {content_data['main_content'][:2000]} {pdf_content}".strip()
            
            embedding, embedding_time = self.create_embedding(combined_text)
            
            if embedding:
                with self.lock:
                    self.pages_count += 1
                    record_id = self.pages_count
                
                page_record = {
                    'id': record_id,
                    'url': url,
                    'url_hash': hashlib.md5(url.encode()).hexdigest()[:16],
                    'title': title,
                    'description': description,
                    'keywords': keywords,
                    'content': {
                        'main_content': content_data['main_content'],
                        'content_length': len(content_data['main_content']),
                        'pdf_content': pdf_content,
                        'links_count': len(content_data['all_links']),
                        'pdfs_count': len(content_data['all_pdfs']),
                        'internal_links_count': len(content_data['internal_links']),
                        'internal_pdfs_count': len(content_data['internal_pdfs'])
                    },
                    'links': {
                        'internal_links': content_data['internal_links'],
                        'internal_pdfs': content_data['internal_pdfs'],
                        'all_links': content_data['all_links'][:50],
                        'all_pdfs': content_data['all_pdfs']
                    },
                    'embedding': embedding,
                    'timing': {
                        'scraping_time': round(scraping_time, 3),
                        'embedding_time': round(embedding_time, 3),
                        'total_time': round(scraping_time + embedding_time, 3)
                    },
                    'crawled_at': datetime.now().isoformat(),
                    'domain': self.domain
                }
                
                self.save_to_qdrant(page_record)
                
                valid_links = []
                for link_data in content_data['internal_links'][:20]:
                    normalized = self.normalize_url(link_data['url'])
                    if self.is_valid_url(normalized):
                        valid_links.append(normalized)
                
                return page_record, valid_links
            
        except Exception as e:
            with self.lock:
                self.stats['total_failed'] += 1
        
        finally:
            session.close()
        
        return None, []
    
    def crawl_domain(self):
        print(f"\nCrawling {self.target_pages} pagine da {self.base_url}")
        print(f"Workers paralleli: {self.max_workers}")
        print(f"Collection Qdrant: {self.collection_name}")
        
        visited_urls = set()
        url_queue = deque([self.base_url])
        found_urls = {self.base_url}
        
        pages_processed = 0
        
        while url_queue and pages_processed < self.target_pages:
            current_batch = []
            batch_size = min(self.max_workers * 3, len(url_queue), self.target_pages - pages_processed)
            
            for _ in range(batch_size):
                if url_queue:
                    url = url_queue.popleft()
                    if url not in visited_urls:
                        current_batch.append(url)
                        visited_urls.add(url)
            
            if not current_batch:
                break
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_url = {executor.submit(self.process_single_url, url): url for url in current_batch}
                
                for future in as_completed(future_to_url):
                    try:
                        page_record, new_links = future.result()
                        
                        if page_record:
                            with self.lock:
                                self.stats['total_crawled'] += 1
                                pages_processed += 1
                            
                            for link in new_links[:10]:
                                if link not in found_urls and len(found_urls) < self.target_pages * 3:
                                    url_queue.append(link)
                                    found_urls.add(link)
                        
                    except Exception as e:
                        pass
            
            if pages_processed % 50 == 0:
                print(f"[{pages_processed}/{self.target_pages}] pagine elaborate")
            
            if self.delay > 0:
                time.sleep(self.delay)
        
        print(f"✅ Completato: {self.pages_count} pagine salvate")
        return self.pages_count
    
    def run(self):
        print("Crawler Universale con Qdrant")
        print("=" * 50)
        
        crawled_count = self.crawl_domain()
        
        if crawled_count > 0:
            self.print_summary()
        else:
            print("❌ Nessuna pagina crawlata")
    
    def print_summary(self):
        duration = datetime.now() - self.stats['start_time']
        
        print("\n" + "=" * 50)
        print("RIEPILOGO")
        print("=" * 50)
        print(f"Dominio: {self.domain}")
        print(f"Collection: {self.collection_name}")
        print(f"Pagine crawlate: {self.pages_count}")
        print(f"PDF estratti: {self.stats['pdfs_extracted']}")
        print(f"Richieste fallite: {self.stats['total_failed']}")
        print(f"Durata totale: {duration}")
        print(f"Velocità: {self.pages_count/(duration.total_seconds()/60):.1f} pagine/min")
        print("=" * 50)

def main():
    print("=" * 50)
    print("Universal Domain Crawler con Qdrant")
    print("=" * 50)
    
    domain = input("\nDominio da crawlare (es. www.unimi.it): ").strip()
    
    if not domain:
        print("❌ Dominio non fornito")
        return
    
    try:
        target_pages = int(input("Numero pagine da crawlare (default: 1000): ").strip() or "1000")
    except ValueError:
        target_pages = 1000
    
    try:
        max_workers = int(input("Workers paralleli (default: 5): ").strip() or "5")
    except ValueError:
        max_workers = 5
    
    crawler = UniversalDomainCrawler(
        domain, 
        delay=0.05, 
        max_workers=max_workers, 
        target_pages=target_pages
    )
    crawler.run()

if __name__ == "__main__":
    main()

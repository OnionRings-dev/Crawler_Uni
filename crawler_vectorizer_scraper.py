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
import random
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class FastUniMiCrawler:
    def __init__(self, delay=0.1, max_workers=5, debug=True):
        self.delay = delay
        self.max_workers = max_workers
        self.target_pages = 1000
        self.base_url = 'https://unimi.it'
        self.domain = urlparse(self.base_url).netloc
        self.debug = debug
        
        self.pages_database = []
        self.embedding_model = None
        self.lock = threading.Lock()
        
        self.stats = {
            'total_crawled': 0,
            'total_failed': 0,
            'pdfs_extracted': 0,
            'start_time': datetime.now(),
            'total_scraping_time': 0.0,
            'total_embedding_time': 0.0,
            'average_scraping_time': 0.0,
            'average_embedding_time': 0.0
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'it-IT,it;q=0.9,en;q=0.8',
            'Connection': 'keep-alive'
        }
        
        self.init_embedding_model()
        
    def init_embedding_model(self):
        try:
            print("Loading embedding model...")
            self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def debug_log(self, message):
        """Log debug messages with timestamp if debug mode is enabled"""
        if self.debug:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] {message}")
    
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
        
        # Estrai tutti i link e classifica interno/esterno
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
                
                # Classifica come interno se appartiene al dominio UniMi
                if self.is_valid_url(absolute_url):
                    content_data['internal_links'].append(link_data)
                
                # Gestisci PDF
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
        """Verifica se un PDF appartiene al dominio UniMi"""
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
        """Create embedding with timing"""
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
    
    def process_single_url(self, url):
        scraping_start = time.time()
        session = self.create_session()
        
        self.debug_log(f"🔍 Processing: {url}")
        
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if 'text/html' not in content_type.lower():
                self.debug_log(f"❌ Non-HTML content type: {content_type}")
                return None, []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title, description, keywords = self.extract_metadata_fast(soup)
            content_data = self.extract_fast_content(soup, url)
            
            pdf_content = ""
            if content_data['internal_pdfs']:
                self.debug_log(f"📄 Extracting PDF content from {len(content_data['internal_pdfs'])} PDFs")
                pdf_content = self.extract_pdf_text_fast(content_data['internal_pdfs'][0]['url'], session)
            
            scraping_time = time.time() - scraping_start
            self.debug_log(f"⏱️ Scraping completed in {scraping_time:.3f}s")
            
            # Create embedding with timing
            combined_text = f"{title} {description} {keywords} {content_data['main_content'][:2000]} {pdf_content}".strip()
            
            self.debug_log("🧠 Creating embedding...")
            embedding, embedding_time = self.create_embedding(combined_text)
            self.debug_log(f"⏱️ Embedding completed in {embedding_time:.3f}s")
            
            # Update timing stats
            with self.lock:
                self.stats['total_scraping_time'] += scraping_time
                self.stats['total_embedding_time'] += embedding_time
                
                if self.stats['total_crawled'] > 0:
                    self.stats['average_scraping_time'] = self.stats['total_scraping_time'] / self.stats['total_crawled']
                    self.stats['average_embedding_time'] = self.stats['total_embedding_time'] / self.stats['total_crawled']
            
            if embedding:
                page_record = {
                    'id': 0,
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
                        'all_links': content_data['all_links'][:50],  # Limita per non appesantire il DB
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
                
                valid_links = []
                for link_data in content_data['internal_links'][:20]:
                    normalized = self.normalize_url(link_data['url'])
                    if self.is_valid_url(normalized):
                        valid_links.append(normalized)
                
                total_time = scraping_time + embedding_time
                self.debug_log(f"✅ Completed {url} in {total_time:.3f}s (scrape: {scraping_time:.3f}s, embed: {embedding_time:.3f}s)")
                
                return page_record, valid_links
            
        except Exception as e:
            with self.lock:
                self.stats['total_failed'] += 1
            self.debug_log(f"❌ Error processing {url}: {str(e)}")
        
        finally:
            session.close()
        
        return None, []
    
    def crawl_unimi_fast(self):
        print(f"\n🚀 Fast crawling {self.target_pages} pages from {self.base_url}")
        print(f"Using {self.max_workers} parallel workers")
        print(f"Debug mode: {'ON' if self.debug else 'OFF'}")
        
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
                                page_record['id'] = len(self.pages_database) + 1
                                self.pages_database.append(page_record)
                                self.stats['total_crawled'] += 1
                                pages_processed += 1
                            
                            for link in new_links[:10]:
                                if link not in found_urls and len(found_urls) < self.target_pages * 3:
                                    url_queue.append(link)
                                    found_urls.add(link)
                        
                    except Exception as e:
                        self.debug_log(f"❌ Future error: {str(e)}")
            
            if pages_processed % 20 == 0:
                avg_scrape = self.stats['average_scraping_time']
                avg_embed = self.stats['average_embedding_time']
                print(f"[{pages_processed}/{self.target_pages}] Processed - Avg times: scrape {avg_scrape:.3f}s, embed {avg_embed:.3f}s")
            
            if self.delay > 0:
                time.sleep(self.delay)
        
        print(f"✅ Completed: {len(self.pages_database)} pages in database")
        return len(self.pages_database)
    
    def save_database(self):
        print("\n💾 Saving database...")
        
        database = {
            'database_info': {
                'name': 'UniMi Fast Vector Database Enhanced',
                'version': '2.2',
                'created_at': datetime.now().isoformat(),
                'domain': self.domain,
                'total_pages': len(self.pages_database),
                'embedding_model': 'paraphrase-multilingual-MiniLM-L12-v2',
                'crawling_stats': self.stats,
                'features': [
                    'Internal links tracking',
                    'Internal PDFs tracking', 
                    'Scraping timing',
                    'Embedding timing',
                    'Debug logging',
                    'Enhanced metadata'
                ]
            },
            'pages': self.pages_database
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'unimi_enhanced_database_{timestamp}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(database, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ Database saved: {filename}")
        return filename
    
    def run(self):
        print("🤖 UniMi Enhanced Crawler & Vector Database Creator")
        print("=" * 50)
        
        crawled_count = self.crawl_unimi_fast()
        
        if crawled_count > 0:
            self.save_database()
            self.print_summary()
        else:
            print("❌ No pages crawled")
    
    def print_summary(self):
        duration = datetime.now() - self.stats['start_time']
        
        print("\n" + "=" * 50)
        print("📊 ENHANCED SUMMARY")
        print("=" * 50)
        print(f"Pages crawled: {len(self.pages_database)}")
        print(f"PDFs extracted: {self.stats['pdfs_extracted']}")
        print(f"Failed requests: {self.stats['total_failed']}")
        print(f"Total duration: {duration}")
        print(f"Crawling speed: {len(self.pages_database)/(duration.total_seconds()/60):.1f} pages/min")
        print("\n📈 TIMING ANALYSIS:")
        print(f"Total scraping time: {self.stats['total_scraping_time']:.2f}s")
        print(f"Total embedding time: {self.stats['total_embedding_time']:.2f}s")
        print(f"Average scraping time: {self.stats['average_scraping_time']:.3f}s per page")
        print(f"Average embedding time: {self.stats['average_embedding_time']:.3f}s per page")
        
        if len(self.pages_database) > 0:
            # Calcola statistiche sui link
            total_internal_links = sum(page['content']['internal_links_count'] for page in self.pages_database)
            total_internal_pdfs = sum(page['content']['internal_pdfs_count'] for page in self.pages_database)
            
            print(f"\n🔗 LINK STATISTICS:")
            print(f"Total internal links found: {total_internal_links}")
            print(f"Total internal PDFs found: {total_internal_pdfs}")
            print(f"Average internal links per page: {total_internal_links/len(self.pages_database):.1f}")
            print(f"Average internal PDFs per page: {total_internal_pdfs/len(self.pages_database):.1f}")
        
        print("=" * 50)

def main():
    # Puoi disabilitare il debug impostando debug=False
    crawler = FastUniMiCrawler(delay=0.05, max_workers=5, debug=True)
    crawler.run()

if __name__ == "__main__":
    main()

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
                 qdrant_path="./qdrant_data", deep_crawl=True, max_depth=None):
        self.delay = delay
        self.max_workers = max_workers
        self.target_pages = target_pages
        self.deep_crawl = deep_crawl
        self.max_depth = max_depth
        self.base_url = base_url if base_url.startswith('http') else f'https://{base_url}'
        self.domain = urlparse(self.base_url).netloc
        
        self.pages_count = 0
        self.embedding_model = None
        self.lock = threading.Lock()
        
        self.crawled_urls = set()
        self.saved_urls = set()
        self.failed_urls = set()
        
        self.qdrant_client = None
        self.collection_name = None
        self.init_qdrant(qdrant_path)
        
        self.stats = {
            'total_crawled': 0,
            'total_failed': 0,
            'pdfs_extracted': 0,
            'urls_discovered': 0,
            'start_time': datetime.now()
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'it-IT,it;q=0.9,en;q=0.8',
            'Connection': 'keep-alive',
            'Accept-Encoding': 'gzip, deflate'
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
                self.load_existing_urls()
                
        except Exception as e:
            print(f"❌ Errore inizializzazione Qdrant: {e}")
            raise
    
    def load_existing_urls(self):
        try:
            print("Caricamento URL esistenti da Qdrant...")
            offset = None
            loaded = 0
            
            while True:
                results = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=["url"],
                    with_vectors=False
                )
                
                points, offset = results
                
                if not points:
                    break
                
                for point in points:
                    url = point.payload.get('url')
                    if url:
                        self.saved_urls.add(url)
                        self.crawled_urls.add(url)
                        loaded += 1
                
                if offset is None:
                    break
            
            self.pages_count = loaded
            print(f"✅ Caricati {loaded} URL esistenti")
            
        except Exception as e:
            print(f"⚠️ Errore caricamento URL esistenti: {e}")
        
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
            
            path = parsed.path
            if path and path != '/':
                path = path.rstrip('/')
            
            query_parts = []
            if parsed.query:
                params = parsed.query.split('&')
                skip_params = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
                              'fbclid', 'gclid', 'msclkid', 'ref', 'share', 'source']
                for param in params:
                    param_name = param.split('=')[0].lower()
                    if param_name not in skip_params:
                        query_parts.append(param)
            
            query_string = '&'.join(sorted(query_parts))
            
            normalized = urlunparse((
                parsed.scheme.lower() or 'https',
                parsed.netloc.lower(),
                path or '/',
                '',
                query_string,
                ''
            ))
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
        
        if not (parsed.netloc == self.domain or parsed.netloc.endswith('.' + self.domain)):
            return False
        
        skip_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                          '.zip', '.rar', '.tar', '.gz', '.7z', '.bz2',
                          '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.ico',
                          '.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv',
                          '.mp3', '.wav', '.ogg', '.flac',
                          '.exe', '.dmg', '.iso', '.bin',
                          '.css', '.js', '.json', '.xml', '.rss'}
        
        path_lower = parsed.path.lower()
        if any(path_lower.endswith(ext) for ext in skip_extensions):
            return False
        
        skip_patterns = ['/wp-admin/', '/wp-includes/', '/wp-content/plugins/',
                        '/admin/', '/login', '/logout', '/signin', '/signup',
                        '/cart/', '/checkout/', '/account/',
                        '/feed/', '/rss/', '/atom/',
                        '/print/', '/email/', '/share/',
                        '?replytocom=', '?attachment_id=']
        
        full_url_lower = url.lower()
        if any(pattern in full_url_lower for pattern in skip_patterns):
            return False
            
        return True
    
    def sanitize_content(self, soup):
        for element in soup(['script', 'noscript', 'style', 'nav', 'footer', 'header', 'aside', 'comment', 'iframe']):
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
        
        main_selectors = ['main', 'article', '[role="main"]', '.content', '#content', 
                         '.main-content', '#main-content', '.post-content', '.entry-content',
                         '.article-content', '.page-content']
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
        
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if not href or href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:') or href.startswith('tel:'):
                continue
                
            try:
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
            except:
                continue
        
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
        
        desc_tag = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
        description = desc_tag.get('content', '').strip() if desc_tag else ""
        
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        keywords = keywords_tag.get('content', '').strip() if keywords_tag else ""
        
        return title, description, keywords
    
    def save_to_qdrant(self, page_record):
        if not self.qdrant_client:
            return False
        
        url = page_record['url']
        if url in self.saved_urls:
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
                'embedding_time': page_record['timing']['embedding_time'],
                'depth': page_record.get('depth', 0)
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
            
            with self.lock:
                self.saved_urls.add(url)
            
            return True
            
        except Exception as e:
            print(f"❌ Errore salvataggio Qdrant per {url}: {e}")
            return False
    
    def process_single_url(self, url, depth=0):
        if url in self.crawled_urls or url in self.failed_urls:
            return None, []
        
        with self.lock:
            self.crawled_urls.add(url)
        
        scraping_start = time.time()
        session = self.create_session()
        
        try:
            response = session.get(url, timeout=15, allow_redirects=True)
            
            final_url = self.normalize_url(response.url)
            
            if final_url != url and final_url in self.crawled_urls:
                return None, []
            
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                return None, []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title, description, keywords = self.extract_metadata_fast(soup)
            content_data = self.extract_fast_content(soup, final_url)
            
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
                    'url': final_url,
                    'url_hash': hashlib.md5(final_url.encode()).hexdigest()[:16],
                    'title': title,
                    'description': description,
                    'keywords': keywords,
                    'depth': depth,
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
                
                if self.max_depth is None or depth < self.max_depth:
                    valid_links = []
                    for link_data in content_data['internal_links']:
                        normalized = self.normalize_url(link_data['url'])
                        if (self.is_valid_url(normalized) and 
                            normalized not in self.crawled_urls and 
                            normalized not in self.failed_urls):
                            valid_links.append((normalized, depth + 1))
                    
                    return page_record, valid_links
                else:
                    return page_record, []
            
        except requests.exceptions.RequestException as e:
            with self.lock:
                self.stats['total_failed'] += 1
                self.failed_urls.add(url)
        except Exception as e:
            with self.lock:
                self.stats['total_failed'] += 1
                self.failed_urls.add(url)
        finally:
            session.close()
        
        return None, []
    
    def crawl_domain(self):
        if self.deep_crawl:
            depth_msg = f"Profondità massima {self.max_depth}" if self.max_depth else "Profondità ILLIMITATA"
            print(f"\n🔍 DEEP CRAWL MODE: {depth_msg}")
            print(f"Salverò fino a {self.target_pages} pagine in Qdrant")
        else:
            print(f"\nCrawling {self.target_pages} pagine da {self.base_url}")
        
        print(f"Workers paralleli: {self.max_workers}")
        print(f"Collection Qdrant: {self.collection_name}")
        
        if self.saved_urls:
            print(f"⚠️ Trovati {len(self.saved_urls)} URL già salvati, li skippo")
        
        url_queue = deque([(self.base_url, 0)])
        found_urls = {self.base_url}
        
        pages_processed = 0
        last_report = 0
        
        while url_queue:
            if pages_processed >= self.target_pages:
                print(f"\n✅ Raggiunto target di {self.target_pages} pagine salvate")
                break
            
            batch_size = min(self.max_workers * 10, len(url_queue), 100)
            
            current_batch = []
            for _ in range(batch_size):
                if url_queue:
                    url, depth = url_queue.popleft()
                    if url not in self.crawled_urls and url not in self.failed_urls:
                        current_batch.append((url, depth))
            
            if not current_batch:
                if url_queue:
                    continue
                else:
                    break
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_url = {
                    executor.submit(self.process_single_url, url, depth): (url, depth) 
                    for url, depth in current_batch
                }
                
                for future in as_completed(future_to_url):
                    try:
                        page_record, new_links = future.result()
                        
                        if page_record:
                            with self.lock:
                                self.stats['total_crawled'] += 1
                                pages_processed += 1
                        
                        for link, link_depth in new_links:
                            if (link not in found_urls and 
                                link not in self.crawled_urls and 
                                link not in self.failed_urls):
                                if self.max_depth is None or link_depth <= self.max_depth:
                                    url_queue.append((link, link_depth))
                                    found_urls.add(link)
                                    with self.lock:
                                        self.stats['urls_discovered'] = len(found_urls)
                        
                    except Exception as e:
                        pass
            
            if pages_processed - last_report >= 100:
                last_report = pages_processed
                avg_depth = sum(d for _, d in list(url_queue)[:1000]) / min(len(url_queue), 1000) if url_queue else 0
                print(f"[{pages_processed}/{self.target_pages}] salvate | "
                      f"{len(found_urls)} URL scoperti | "
                      f"{len(url_queue)} in coda | "
                      f"depth avg: {avg_depth:.1f} | "
                      f"failed: {len(self.failed_urls)}")
            
            if self.delay > 0:
                time.sleep(self.delay)
        
        print(f"\n✅ Completato:")
        print(f"   - {self.pages_count} pagine totali salvate in Qdrant")
        print(f"   - {len(found_urls)} URL totali scoperti")
        print(f"   - {len(self.crawled_urls)} URL visitati")
        print(f"   - {len(self.failed_urls)} URL falliti")
        
        return self.pages_count
    
    def run(self):
        print("\n" + "=" * 50)
        print("Crawler Universale con Qdrant - Avvio")
        print("=" * 50)
        
        crawled_count = self.crawl_domain()
        
        if crawled_count > 0:
            self.print_summary()
        else:
            print("❌ Nessuna pagina crawlata")
    
    def print_summary(self):
        duration = datetime.now() - self.stats['start_time']
        
        print("\n" + "=" * 50)
        print("RIEPILOGO FINALE")
        print("=" * 50)
        print(f"Dominio: {self.domain}")
        print(f"Collection Qdrant: {self.collection_name}")
        depth_msg = f"{self.max_depth}" if self.max_depth else "ILLIMITATA"
        print(f"Profondità massima: {depth_msg}")
        print(f"Pagine salvate: {self.pages_count}")
        print(f"URL scoperti: {self.stats['urls_discovered']}")
        print(f"URL crawlati: {len(self.crawled_urls)}")
        print(f"URL falliti: {len(self.failed_urls)}")
        print(f"PDF estratti: {self.stats['pdfs_extracted']}")
        print(f"Durata totale: {duration}")
        if duration.total_seconds() > 0:
            print(f"Velocità: {self.pages_count/(duration.total_seconds()/60):.1f} pagine/min")
        print("=" * 50)
        print(f"\n✅ Dati salvati in Qdrant collection: {self.collection_name}")
        print(f"✅ Path database: ./qdrant_data")

def main():
    print("=" * 50)
    print("Universal Domain Crawler con Qdrant")
    print("=" * 50)
    
    domain = input("\nDominio da crawlare (es. www.unimi.it): ").strip()
    
    if not domain:
        print("❌ Dominio non fornito")
        return
    
    try:
        target_pages = int(input("Numero pagine da SALVARE (0 = illimitato, default: 100000): ").strip() or "100000")
    except ValueError:
        target_pages = 100000
    
    depth_input = input("Profondità massima crawling (vuoto = illimitato, default: illimitato): ").strip()
    max_depth = None if not depth_input else int(depth_input)
    
    try:
        max_workers = int(input("Workers paralleli (default: 10): ").strip() or "10")
    except ValueError:
        max_workers = 10
    
    deep_mode = input("Deep crawl (esplora in profondità)? (s/n, default: s): ").strip().lower()
    deep_crawl = deep_mode != 'n'
    
    print(f"\n🚀 Configurazione:")
    print(f"   - Dominio: {domain}")
    print(f"   - Pagine target: {'ILLIMITATO' if target_pages == 0 else target_pages}")
    depth_msg = "ILLIMITATA" if max_depth is None else str(max_depth)
    print(f"   - Profondità max: {depth_msg}")
    print(f"   - Workers: {max_workers}")
    print(f"   - Deep crawl: {'Sì' if deep_crawl else 'No'}")
    
    confirm = input("\nAvviare il crawling? (s/n): ").strip().lower()
    if confirm != 's':
        print("❌ Operazione annullata")
        return
    
    if target_pages == 0:
        target_pages = 999999999
    
    crawler = UniversalDomainCrawler(
        domain, 
        delay=0.02, 
        max_workers=max_workers, 
        target_pages=target_pages,
        deep_crawl=deep_crawl,
        max_depth=max_depth
    )
    crawler.run()

if __name__ == "__main__":
    main()

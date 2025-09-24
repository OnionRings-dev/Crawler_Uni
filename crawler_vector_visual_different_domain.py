#!/usr/bin/env python3
"""
Crawler Multi-Dominio per Testing Vettoriale
Crawla 30 link da unimi.it, youtube.com e treccani.it
Crea visualizzazione dei vettori con colori distintivi
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
import time
import re
import json
from collections import deque
from datetime import datetime
import hashlib
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
from typing import List, Dict, Tuple, Set
import random

class MultiDomainCrawler:
    def __init__(self, delay=0.5, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.delay = delay
        self.targets_per_domain = 30
        
        self.domains = {
            'unimi': {
                'base_url': 'https://unimi.it',
                'color': 'green',
                'label': 'unimi',
                'pages': [],
                'embeddings': []
            },
            'youtube': {
                'base_url': 'https://youtube.com',
                'color': 'red', 
                'label': 'youtube',
                'pages': [],
                'embeddings': []
            },
            'treccani': {
                'base_url': 'https://treccani.it',
                'color': 'blue',
                'label': 'treccani', 
                'pages': [],
                'embeddings': []
            }
        }
        
        self.test_query = "iscrizione università corsi magistrale triennale erasmus"
        self.query_embedding = None
        
        self.embedding_model = None
        self.all_embeddings = []
        self.all_labels = []
        self.all_colors = []
        
        self.stats = {
            'total_crawled': 0,
            'total_failed': 0,
            'start_time': datetime.now()
        }
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; TestBot/1.0)',
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
    
    def is_valid_url(self, url, domain_key):
        if not url:
            return False
            
        try:
            parsed = urlparse(url)
        except:
            return False
        
        target_domain = urlparse(self.domains[domain_key]['base_url']).netloc
        
        valid_domain = (parsed.netloc == target_domain or 
                       parsed.netloc.endswith('.' + target_domain))
        
        if not valid_domain:
            return False
        
        skip_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.zip', '.jpg', '.png', '.gif', '.mp4', '.avi'}
        if any(parsed.path.lower().endswith(ext) for ext in skip_extensions):
            return False
        
        return True
    
    def extract_text_content(self, soup):
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        main_content = ""
        
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
        
        main_content = re.sub(r'\s+', ' ', main_content)
        main_content = main_content.strip()
        
        return main_content[:3000]
    
    def create_embedding(self, text):
        try:
            if not text.strip():
                return None
            
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            return embedding.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Errore nella creazione embedding: {e}")
            return None
    
    def extract_links(self, url, soup, domain_key):
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
                    
                    if self.is_valid_url(normalized_url, domain_key):
                        new_links.add(normalized_url)
                
                except Exception:
                    continue
        
        except Exception as e:
            self.logger.error(f"Errore nell'estrazione link da {url}: {e}")
        
        return new_links
    
    def crawl_page(self, url, domain_key):
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if 'text/html' not in content_type.lower():
                return None, set()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            description = ""
            desc_tag = soup.find('meta', attrs={'name': 'description'})
            if desc_tag:
                description = desc_tag.get('content', '').strip()
            
            content = self.extract_text_content(soup)
            
            combined_text = f"{title}. {description}. {content}".strip()
            
            embedding = self.create_embedding(combined_text)
            
            if embedding is not None:
                page_data = {
                    'url': url,
                    'title': title,
                    'description': description,
                    'content': content[:500],
                    'embedding': embedding,
                    'domain': domain_key
                }
                
                self.domains[domain_key]['pages'].append(page_data)
                self.domains[domain_key]['embeddings'].append(embedding)
                
                self.stats['total_crawled'] += 1
                
                new_links = self.extract_links(url, soup, domain_key)
                return page_data, new_links
            
            return None, set()
            
        except Exception as e:
            self.logger.error(f"Errore crawling {url}: {e}")
            self.stats['total_failed'] += 1
            return None, set()
    
    def crawl_domain(self, domain_key):
        domain_info = self.domains[domain_key]
        base_url = domain_info['base_url']
        
        print(f"\n🎯 === CRAWLING {domain_key.upper()} ===")
        print(f"Target: {self.targets_per_domain} pagine da {base_url}")
        
        visited_urls = set()
        url_queue = deque([base_url])
        found_urls = set([base_url])
        
        pages_crawled = 0
        attempts = 0
        max_attempts = self.targets_per_domain * 3
        
        while url_queue and pages_crawled < self.targets_per_domain and attempts < max_attempts:
            attempts += 1
            current_url = url_queue.popleft()
            
            if current_url in visited_urls:
                continue
            
            visited_urls.add(current_url)
            
            print(f"[{pages_crawled+1}/{self.targets_per_domain}] Crawling: {current_url}")
            
            page_data, new_links = self.crawl_page(current_url, domain_key)
            
            if page_data:
                pages_crawled += 1
                print(f"✅ Successo: {page_data['title'][:50]}...")
            
            limited_new_links = list(new_links - found_urls)[:10]
            random.shuffle(limited_new_links)
            
            for link in limited_new_links:
                if link not in found_urls:
                    url_queue.append(link)
                    found_urls.add(link)
            
            if self.delay > 0:
                time.sleep(self.delay)
        
        print(f"✅ Completato {domain_key}: {len(domain_info['pages'])} pagine crawlate")
        return len(domain_info['pages'])
    
    def prepare_data_for_visualization(self):
        print("\n📊 === PREPARAZIONE DATI PER VISUALIZZAZIONE ===")
        
        self.all_embeddings = []
        self.all_labels = []
        self.all_colors = []
        
        for domain_key, domain_info in self.domains.items():
            for embedding in domain_info['embeddings']:
                self.all_embeddings.append(embedding)
                self.all_labels.append(domain_info['label'])
                self.all_colors.append(domain_info['color'])
        
        query_embedding = self.create_embedding(self.test_query)
        if query_embedding is not None:
            self.query_embedding = query_embedding
            self.all_embeddings.append(query_embedding)
            self.all_labels.append('domanda')
            self.all_colors.append('purple')
        
        self.all_embeddings = np.array(self.all_embeddings)
        
        print(f"Dati preparati: {len(self.all_embeddings)} vettori totali")
        for domain_key, domain_info in self.domains.items():
            print(f"- {domain_key}: {len(domain_info['embeddings'])} vettori")
        if self.query_embedding is not None:
            print(f"- domanda: 1 vettore")
    
    def create_visualization(self):
        if len(self.all_embeddings) < 2:
            print("❌ Non abbastanza dati per la visualizzazione")
            return
        
        print("\n🎨 === CREAZIONE VISUALIZZAZIONE ===")
        
        print("Applicando PCA...")
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(self.all_embeddings)
        
        plt.figure(figsize=(15, 10))
        
        domain_points = {'unimi': [], 'youtube': [], 'treccani': [], 'domanda': []}
        domain_colors = {'unimi': 'green', 'youtube': 'red', 'treccani': 'blue', 'domanda': 'purple'}
        
        for i, (label, color) in enumerate(zip(self.all_labels, self.all_colors)):
            x, y = embeddings_2d[i]
            domain_points[label].append((x, y))
        
        for domain, points in domain_points.items():
            if points:
                xs, ys = zip(*points)
                plt.scatter(xs, ys, c=domain_colors[domain], label=domain, alpha=0.7, s=50)
        
        plt.xlabel(f'Prima Componente PCA (var: {pca.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'Seconda Componente PCA (var: {pca.explained_variance_ratio_[1]:.3f})')
        plt.title('Visualizzazione Vettori - Confronto Multi-Dominio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizzazione_vettori.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizzazione salvata come 'visualizzazione_vettori.png'")
    
    def save_results(self):
        results = {
            'domains': {k: {'pages': v['pages'], 'count': len(v['pages'])} for k, v in self.domains.items()},
            'test_query': self.test_query,
            'stats': self.stats,
            'total_embeddings': len(self.all_embeddings)
        }
        
        with open('multi_domain_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print("💾 Risultati salvati in 'multi_domain_results.json'")
    
    def run(self):
        print("🚀 === AVVIO CRAWLER MULTI-DOMINIO ===")
        print(f"Target: {self.targets_per_domain} pagine per dominio")
        print(f"Query test: '{self.test_query}'")
        
        total_crawled = 0
        
        for domain_key in self.domains.keys():
            crawled = self.crawl_domain(domain_key)
            total_crawled += crawled
        
        if total_crawled > 0:
            self.prepare_data_for_visualization()
            self.create_visualization() 
            self.save_results()
            self.print_summary()
        else:
            print("❌ Nessuna pagina crawlata con successo")
    
    def print_summary(self):
        duration = datetime.now() - self.stats['start_time']
        
        print("\n" + "=" * 60)
        print("🎯 RIASSUNTO CRAWLING MULTI-DOMINIO")
        print("=" * 60)
        
        for domain_key, domain_info in self.domains.items():
            print(f"{domain_key.upper()}: {len(domain_info['pages'])} pagine")
        
        print(f"TOTALE: {sum(len(d['pages']) for d in self.domains.values())} pagine")
        print(f"Fallimenti: {self.stats['total_failed']}")
        print(f"Durata: {duration}")
        print(f"Query test: '{self.test_query}'")
        print("=" * 60)


def main():
    print("🤖 Crawler Multi-Dominio per Testing Vettoriale")
    print("=" * 50)
    
    crawler = MultiDomainCrawler(delay=0.3)
    crawler.run()

if __name__ == "__main__":
    main()

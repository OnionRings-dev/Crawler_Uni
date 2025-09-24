#!/usr/bin/env python3
"""
Crawler UniMi per Testing Vettoriale e Ricerca Semantica
Crawla 300 link da unimi.it e trova i 10 più vicini alla query
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
from sklearn.metrics.pairwise import cosine_similarity

class UniMiSemanticCrawler:
    def __init__(self, delay=0.3, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.delay = delay
        self.target_pages = 300
        self.base_url = 'https://unimi.it'
        self.domain = urlparse(self.base_url).netloc
        
        self.test_query = "come iscriversi al corso di informatica all'università"
        self.query_embedding = None
        
        self.pages = []
        self.embeddings = []
        self.all_embeddings = []
        self.all_labels = []
        self.all_colors = []
        
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
        
        return main_content[:4000]
    
    def create_embedding(self, text):
        try:
            if not text.strip():
                return None
            
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            return embedding.astype(np.float32)
            
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
                    'content': content[:1000],
                    'embedding': embedding,
                    'combined_text': combined_text[:500]
                }
                
                self.pages.append(page_data)
                self.embeddings.append(embedding)
                
                self.stats['total_crawled'] += 1
                
                new_links = self.extract_links(url, soup)
                return page_data, new_links
            
            return None, set()
            
        except Exception as e:
            self.logger.error(f"Errore crawling {url}: {e}")
            self.stats['total_failed'] += 1
            return None, set()
    
    def crawl_unimi(self):
        print(f"\n🎯 === CRAWLING UNIMI.IT ===")
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
            
            page_data, new_links = self.crawl_page(current_url)
            
            if page_data:
                pages_crawled += 1
                if pages_crawled % 50 == 0:
                    print(f"✅ Crawlate {pages_crawled} pagine")
            
            limited_new_links = list(new_links - found_urls)[:15]
            random.shuffle(limited_new_links)
            
            for link in limited_new_links:
                if link not in found_urls:
                    url_queue.append(link)
                    found_urls.add(link)
            
            if self.delay > 0:
                time.sleep(self.delay)
        
        print(f"✅ Completato crawling: {len(self.pages)} pagine")
        return len(self.pages)
    
    def find_most_similar_pages(self):
        print(f"\n🔍 === RICERCA PAGINE SIMILI ===")
        print(f"Query: '{self.test_query}'")
        
        if not self.embeddings:
            print("❌ Nessun embedding disponibile")
            return []
        
        query_embedding = self.create_embedding(self.test_query)
        if query_embedding is None:
            print("❌ Impossibile creare embedding per la query")
            return []
        
        self.query_embedding = query_embedding
        
        embeddings_matrix = np.array(self.embeddings)
        query_vector = query_embedding.reshape(1, -1)
        
        similarities = cosine_similarity(query_vector, embeddings_matrix)[0]
        
        top_10_indices = np.argsort(similarities)[::-1][:10]
        
        most_similar = []
        for i, idx in enumerate(top_10_indices, 1):
            similarity_score = similarities[idx]
            page = self.pages[idx]
            
            result = {
                'rank': i,
                'url': page['url'],
                'title': page['title'],
                'description': page['description'],
                'similarity': float(similarity_score),
                'content_preview': page['combined_text'][:200] + '...'
            }
            most_similar.append(result)
            
            print(f"{i}. {similarity_score:.4f} - {page['title'][:50]}...")
            print(f"   {page['url']}")
        
        return most_similar
    
    def save_top_results(self, most_similar):
        output = {
            'query': self.test_query,
            'timestamp': datetime.now().isoformat(),
            'total_pages_crawled': len(self.pages),
            'top_10_results': most_similar
        }
        
        with open('top_10_similar_pages.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        with open('top_10_similar_pages.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("🔍 TOP 10 PAGINE PIÙ SIMILI ALLA QUERY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Query: {self.test_query}\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Pagine totali crawlate: {len(self.pages)}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            for result in most_similar:
                f.write(f"{result['rank']}. SIMILARITÀ: {result['similarity']:.4f}\n")
                f.write(f"URL: {result['url']}\n")
                f.write(f"Titolo: {result['title']}\n")
                f.write(f"Descrizione: {result['description']}\n")
                f.write(f"Anteprima: {result['content_preview']}\n")
                f.write("-" * 50 + "\n\n")
        
        print("💾 Top 10 risultati salvati in:")
        print("  - top_10_similar_pages.json")
        print("  - top_10_similar_pages.txt")
    
    def prepare_data_for_visualization(self):
        print("\n📊 === PREPARAZIONE DATI PER VISUALIZZAZIONE ===")
        
        self.all_embeddings = []
        self.all_labels = []
        self.all_colors = []
        
        for embedding in self.embeddings:
            self.all_embeddings.append(embedding)
            self.all_labels.append('unimi')
            self.all_colors.append('green')
        
        if self.query_embedding is not None:
            self.all_embeddings.append(self.query_embedding)
            self.all_labels.append('domanda')
            self.all_colors.append('purple')
        
        self.all_embeddings = np.array(self.all_embeddings)
        
        print(f"Dati preparati: {len(self.all_embeddings)} vettori totali")
        print(f"- unimi: {len(self.embeddings)} vettori")
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
        
        unimi_points = []
        query_point = None
        
        for i, label in enumerate(self.all_labels):
            x, y = embeddings_2d[i]
            if label == 'unimi':
                unimi_points.append((x, y))
            elif label == 'domanda':
                query_point = (x, y)
        
        if unimi_points:
            xs, ys = zip(*unimi_points)
            plt.scatter(xs, ys, c='green', label='unimi', alpha=0.6, s=30)
        
        if query_point:
            plt.scatter(query_point[0], query_point[1], c='purple', label='domanda', s=200, marker='*', edgecolors='black', linewidth=2)
        
        plt.xlabel(f'Prima Componente PCA (var: {pca.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'Seconda Componente PCA (var: {pca.explained_variance_ratio_[1]:.3f})')
        plt.title(f'Visualizzazione Vettori UniMi ({len(self.embeddings)} pagine)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizzazione_unimi_vettori.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizzazione salvata come 'visualizzazione_unimi_vettori.png'")
    
    def save_all_results(self):
        results = {
            'base_url': self.base_url,
            'test_query': self.test_query,
            'total_pages': len(self.pages),
            'stats': self.stats,
            'pages_summary': [
                {
                    'url': page['url'],
                    'title': page['title'],
                    'description': page['description']
                }
                for page in self.pages
            ]
        }
        
        with open('unimi_crawling_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print("💾 Risultati completi salvati in 'unimi_crawling_results.json'")
    
    def run(self):
        print("🚀 === AVVIO CRAWLER UNIMI SEMANTICO ===")
        print(f"Target: {self.target_pages} pagine da unimi.it")
        print(f"Query test: '{self.test_query}'")
        
        crawled_count = self.crawl_unimi()
        
        if crawled_count > 0:
            most_similar = self.find_most_similar_pages()
            
            if most_similar:
                self.save_top_results(most_similar)
            
            self.prepare_data_for_visualization()
            self.create_visualization()
            self.save_all_results()
            self.print_summary()
        else:
            print("❌ Nessuna pagina crawlata con successo")
    
    def print_summary(self):
        duration = datetime.now() - self.stats['start_time']
        
        print("\n" + "=" * 60)
        print("🎯 RIASSUNTO CRAWLING UNIMI")
        print("=" * 60)
        print(f"Pagine crawlate: {len(self.pages)}")
        print(f"Fallimenti: {self.stats['total_failed']}")
        print(f"Durata: {duration}")
        print(f"Query: '{self.test_query}'")
        print(f"Files generati:")
        print("  - visualizzazione_unimi_vettori.png")
        print("  - top_10_similar_pages.json")
        print("  - top_10_similar_pages.txt")
        print("  - unimi_crawling_results.json")
        print("=" * 60)


def main():
    print("🤖 Crawler UniMi per Testing Vettoriale e Ricerca Semantica")
    print("=" * 60)
    
    crawler = UniMiSemanticCrawler(delay=0.2)
    crawler.run()

if __name__ == "__main__":
    main()

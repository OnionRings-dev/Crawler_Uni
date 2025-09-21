import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import threading
from queue import Queue
import time
import sys
import signal

class WebCrawler:
    def __init__(self, domain):
        self.domain = domain
        self.base_url = f"https://{domain}"
        self.visited = set()
        self.all_links = set()
        self.queue = Queue()
        self.lock = threading.Lock()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.running = True
        
    def is_valid_url(self, url):
        parsed = urlparse(url)
        return parsed.netloc.endswith(self.domain)
    
    def extract_links(self, url):
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            links = set()
            for link in soup.find_all('a', href=True):
                full_url = urljoin(url, link['href'])
                parsed = urlparse(full_url)
                if parsed.scheme in ['http', 'https']:
                    clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                    if parsed.query:
                        clean_url += f"?{parsed.query}"
                    links.add(clean_url)
            
            return links
        except Exception:
            return set()
    
    def worker(self):
        while self.running:
            try:
                url = self.queue.get(timeout=1)
                if url is None or not self.running:
                    break
                
                with self.lock:
                    if url in self.visited:
                        self.queue.task_done()
                        continue
                    self.visited.add(url)
                
                print(f"Analizzando: {url}")
                
                links = self.extract_links(url)
                
                with self.lock:
                    for link in links:
                        self.all_links.add(link)
                        if self.is_valid_url(link) and link not in self.visited and self.running:
                            self.queue.put(link)
                
                self.queue.task_done()
            except:
                continue
    
    def crawl(self, num_threads=10):
        self.queue.put(self.base_url)
        
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=self.worker)
            t.daemon = True
            t.start()
            threads.append(t)
        
        self.queue.join()
        
        for _ in range(num_threads):
            self.queue.put(None)
        
        for t in threads:
            t.join()
    
    def save_links(self, filename="links.txt"):
        sorted_links = sorted(self.all_links)
        with open(filename, 'w', encoding='utf-8') as f:
            for link in sorted_links:
                f.write(f"{link}\n")
        
        print(f"\nTrovati {len(sorted_links)} link unici")
        print(f"Salvati in {filename}")

crawler_instance = None

def signal_handler(sig, frame):
    print(f"\n\nInterruzione rilevata! Salvando i link trovati...")
    if crawler_instance:
        crawler_instance.stop()
        crawler_instance.save_links()
    sys.exit(0)

def main():
    global crawler_instance
    
    signal.signal(signal.SIGINT, signal_handler)
    
    domain = input("Inserisci il dominio da analizzare: ").strip()
    if not domain:
        print("Dominio non valido")
        sys.exit(1)
    
    if domain.startswith(('http://', 'https://')):
        domain = urlparse(domain).netloc
    
    crawler_instance = WebCrawler(domain)
    
    print(f"Inizio crawling del dominio: {domain}")
    print("Premi Ctrl+C per interrompere e salvare i link")
    start_time = time.time()
    
    crawler_instance.crawl()
    
    end_time = time.time()
    print(f"Crawling completato in {end_time - start_time:.2f} secondi")
    
    crawler_instance.save_links()

if __name__ == "__main__":
    main()

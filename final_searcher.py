#!/usr/bin/env python3

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import time
from datetime import datetime
import logging
import os
from typing import List, Dict
import argparse
import re
from groq import Groq
import tiktoken
import sys

class QdrantMultiDomainSearchBot:
    def __init__(self, groq_api_key=None, model_name="openai/gpt-oss-20b", 
                 embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 qdrant_path="./qdrant_data"):
        self.groq_api_key = groq_api_key or 'API'
        self.model_name = model_name
        self.groq_client = None
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        
        self.qdrant_path = qdrant_path
        self.qdrant_client = None
        self.current_collection = None
        self.current_domain = None
        
        self.current_query = ""
        self.cleaned_query = ""
        self.search_results = []
        self.scraped_pages = []
        
        logging.basicConfig(level=logging.ERROR)
        self.logger = logging.getLogger(__name__)
        
        self.init_qdrant_client()
        self.init_embedding_model()
        self.init_groq_client()
        self.init_tokenizer()
    
    def init_qdrant_client(self):
        try:
            print("Connessione a Qdrant...")
            self.qdrant_client = QdrantClient(path=self.qdrant_path)
            print("✅ Qdrant connesso")
        except Exception as e:
            print(f"❌ Errore connessione Qdrant: {e}")
            raise
    
    def init_embedding_model(self):
        try:
            print("Caricamento modello embedding...")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print("✅ Modello embedding caricato")
        except Exception as e:
            print(f"❌ Errore caricamento embedding: {e}")
            raise
    
    def init_groq_client(self):
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY richiesta")
        
        try:
            self.groq_client = Groq(api_key=self.groq_api_key)
            print("✅ Groq inizializzato")
        except Exception as e:
            print(f"❌ Errore inizializzazione Groq: {e}")
            raise
    
    def init_tokenizer(self):
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def get_available_domains(self):
        try:
            collections = self.qdrant_client.get_collections().collections
            domains = []
            
            for collection in collections:
                if collection.name.startswith('crawl_'):
                    domain = collection.name.replace('crawl_', '').replace('_', '.')
                    
                    try:
                        collection_info = self.qdrant_client.get_collection(collection.name)
                        points_count = collection_info.points_count
                    except:
                        points_count = 0
                    
                    domains.append({
                        'domain': domain,
                        'collection': collection.name,
                        'points_count': points_count
                    })
            
            return sorted(domains, key=lambda x: x['domain'])
            
        except Exception as e:
            print(f"Errore recupero domini: {e}")
            return []
    
    def load_domain(self, domain):
        try:
            domain_clean = domain.replace('.', '_').replace('-', '_')
            collection_name = f"crawl_{domain_clean}"
            
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(c.name == collection_name for c in collections)
            
            if not collection_exists:
                print(f"❌ Collection non trovata: {collection_name}")
                return False
            
            collection_info = self.qdrant_client.get_collection(collection_name)
            
            self.current_collection = collection_name
            self.current_domain = domain
            
            print(f"✅ Dominio caricato: {domain} ({collection_info.points_count} documenti)")
            
            return True
            
        except Exception as e:
            print(f"❌ Errore caricamento dominio {domain}: {e}")
            return False
    
    def clean_user_query(self, user_query):
        stop_words = {
            'il', 'la', 'lo', 'le', 'gli', 'i', 'un', 'una', 'uno',
            'del', 'della', 'dei', 'delle', 'degli', 'al', 'alla', 'alle', 'agli',
            'dal', 'dalla', 'dalle', 'dagli', 'nel', 'nella', 'nelle', 'negli',
            'sul', 'sulla', 'sulle', 'sugli', 'per', 'con', 'tra', 'fra',
            'di', 'da', 'in', 'su', 'a', 'e', 'o', 'ma', 'però', 'quindi',
            'potresti', 'riusciresti', 'puoi', 'riesci', 'per favore', 'grazie',
            'vorrei', 'volevo', 'mi servirebbe', 'mi serve', 'ho bisogno',
            'come faccio', 'dove posso', 'è possibile', 'si può', 'anche',
            'ancora', 'già', 'che', 'chi', 'cui', 'quale', 'quali'
        }
        
        text = user_query.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        words = text.split()
        filtered_words = [
            word for word in words 
            if word and word not in stop_words and len(word) > 2
        ]
        
        cleaned = ' '.join(filtered_words[:12])
        self.cleaned_query = cleaned
        
        return cleaned
    
    def create_query_embedding(self, query):
        try:
            if not query.strip():
                return None
            
            embedding = self.embedding_model.encode(query, normalize_embeddings=True)
            return embedding.astype(np.float32).tolist()
            
        except Exception as e:
            print(f"Errore creazione embedding: {e}")
            return None
    
    def semantic_search_qdrant(self, query, top_k=15):
        if not self.current_collection:
            raise ValueError("Nessun dominio caricato. Usa load_domain() prima")
        
        cleaned_query = self.clean_user_query(query)
        self.current_query = query
        
        query_embedding = self.create_query_embedding(cleaned_query)
        if query_embedding is None:
            raise ValueError("Impossibile creare embedding della query")
        
        search_results = self.qdrant_client.search(
            collection_name=self.current_collection,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        
        results = []
        for i, hit in enumerate(search_results, 1):
            payload = hit.payload
            
            links = {}
            if 'links_json' in payload:
                try:
                    links = json.loads(payload['links_json'])
                except:
                    pass
            
            result = {
                'rank': i,
                'id': hit.id,
                'url': payload.get('url', ''),
                'title': payload.get('title', 'No title'),
                'description': payload.get('description', ''),
                'keywords': payload.get('keywords', ''),
                'similarity_score': float(hit.score),
                'content_length': payload.get('content_length', 0),
                'crawled_at': payload.get('crawled_at', ''),
                'content_preview': payload.get('main_content', '')[:300] + '...',
                'internal_links_count': payload.get('internal_links_count', 0),
                'internal_pdfs_count': payload.get('internal_pdfs_count', 0),
                'timing': {
                    'scraping_time': payload.get('scraping_time', 0),
                    'embedding_time': payload.get('embedding_time', 0)
                },
                'links': links,
                'full_content': payload.get('main_content', ''),
                'source': 'qdrant'
            }
            results.append(result)
        
        print(f"✅ Trovate {len(results)} pagine rilevanti")
        
        self.search_results = results
        return results
    
    def prepare_pages_for_llm_streaming(self):
        """Prepara le pagine con output progressivo"""
        self.scraped_pages = []
        
        for result in self.search_results:
            page_data = {
                'rank': result['rank'],
                'url': result['url'],
                'title': result['title'],
                'description': result['description'],
                'content': result['full_content'],
                'content_length': result['content_length'],
                'similarity_score': result['similarity_score'],
                'internal_links_count': result['internal_links_count'],
                'internal_pdfs_count': result['internal_pdfs_count'],
                'links': result['links'],
                'timing': result['timing'],
                'source': 'qdrant',
                'crawled_at': result['crawled_at']
            }
            self.scraped_pages.append(page_data)
        
        return len(self.scraped_pages)
    
    def count_tokens(self, text):
        try:
            return len(self.tokenizer.encode(text))
        except:
            return int(len(text.split()) * 1.3)
    
    def create_latex_ai_prompt_streaming(self):
        """Crea il prompt mostrando progressivamente cosa viene aggiunto"""
        sorted_pages = sorted(self.scraped_pages, 
                            key=lambda x: x.get('similarity_score', 0), 
                            reverse=True)
        
        base_prompt = f"""Sei un assistente esperto del dominio {self.current_domain}. 
Un utente ha fatto questa domanda: "{self.current_query}"

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
        
        total_tokens = self.count_tokens(base_prompt)
        max_tokens = 7000
        
        prompt = base_prompt
        pages_added = 0
        
        for page in sorted_pages:
            if pages_added >= 12:
                break
            
            page_content = f"""
--- PAGINA {page['rank']} (Score: {page['similarity_score']:.4f}) ---
URL: {page['url']}
Titolo: {page['title']}
Descrizione: {page['description']}
Contenuto: {page['content'][:3000]}{'...' if len(page['content']) > 3000 else ''}

"""
            
            page_tokens = self.count_tokens(page_content)
            
            if total_tokens + page_tokens > max_tokens:
                break
            
            prompt += page_content
            total_tokens += page_tokens
            pages_added += 1
        
        prompt += f"""
GENERA UNA RISPOSTA COMPLETA IN FORMATO LaTeX che aiuti concretamente l'utente a risolvere la sua domanda.
La risposta deve essere pronta per la compilazione LaTeX e ben strutturata.
Ricorda di includere tutti i dettagli importanti trovati nelle pagine analizzate.
"""
        
        return prompt
    
    def generate_ai_response_streaming(self, prompt):
        """Genera la risposta con streaming dei token"""
        try:
            print("\n" + "="*60)
            print("🤖 RISPOSTA DELL'AI")
            print("="*60 + "\n")
            
            response_stream = self.groq_client.chat.completions.create(
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
                stream=True  # ABILITATO STREAMING
            )
            
            full_response = ""
            
            # Stream dei token in tempo reale
            for chunk in response_stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    print(content, end='', flush=True)
            
            print("\n\n" + "="*60)
            print("✅ Risposta completata")
            print("="*60 + "\n")
            
            return full_response
            
        except Exception as e:
            print(f"\n❌ Errore generazione risposta: {e}")
            raise
    
    def save_results(self, ai_response):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        clean_query = re.sub(r'[^\w\s-]', '', self.current_query)[:50]
        clean_query = re.sub(r'\s+', '_', clean_query)
        domain_clean = self.current_domain.replace('.', '_')
        
        latex_filename = f'{domain_clean}_response_{clean_query}_{timestamp}.tex'
        with open(latex_filename, 'w', encoding='utf-8') as f:
            f.write(ai_response)
        
        print(f"💾 Risposta salvata in: {latex_filename}\n")
        
        return latex_filename
    
    def process_query(self, query, domain, top_k=15):
        try:
            if not self.current_collection or self.current_domain != domain:
                if not self.load_domain(domain):
                    raise ValueError(f"Impossibile caricare dominio: {domain}")
            
            search_results = self.semantic_search_qdrant(query, top_k)
            
            if not search_results:
                print("❌ Nessuna pagina rilevante trovata")
                return None
            
            self.prepare_pages_for_llm_streaming()
            prompt = self.create_latex_ai_prompt_streaming()
            ai_response = self.generate_ai_response_streaming(prompt)
            files = self.save_results(ai_response)
            
            print(f"✅ Query elaborata con successo!\n")
            
            return ai_response
            
        except Exception as e:
            print(f"❌ Errore elaborazione query: {e}")
            return None
    
    def interactive_mode(self):
        print("\n" + "="*60)
        print("Qdrant Multi-Domain Search Bot (STREAMING MODE)")
        print("="*60)
        
        available_domains = self.get_available_domains()
        
        if not available_domains:
            print("❌ Nessuna collection trovata. Esegui prima il crawler.")
            return
        
        print(f"\n📚 Domini disponibili:")
        for info in available_domains:
            print(f"   - {info['domain']} ({info['points_count']} documenti)")
        
        print("\n💡 Comandi:")
        print("  - Scrivi la tua domanda")
        print("  - 'domains' per lista domini")
        print("  - 'switch <domain>' per cambiare dominio")
        print("  - 'quit' per uscire")
        print("="*60)
        
        current_domain_info = available_domains[0]
        if not self.load_domain(current_domain_info['domain']):
            print("❌ Impossibile caricare il database")
            return
        
        while True:
            try:
                query = input(f"\n[{self.current_domain}] Domanda: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Arrivederci!")
                    break
                
                if query.lower() == 'domains':
                    print(f"\n📚 Domini disponibili:")
                    for info in available_domains:
                        marker = " ← attuale" if info['domain'] == self.current_domain else ""
                        print(f"   - {info['domain']} ({info['points_count']} documenti){marker}")
                    continue
                
                if query.lower().startswith('switch '):
                    new_domain = query[7:].strip()
                    domain_names = [d['domain'] for d in available_domains]
                    if new_domain in domain_names:
                        if self.load_domain(new_domain):
                            print(f"✅ Dominio attivo: {self.current_domain}")
                        else:
                            print(f"❌ Impossibile cambiare dominio: {new_domain}")
                    else:
                        print(f"❌ Dominio non trovato: {new_domain}")
                        print(f"Disponibili: {', '.join(domain_names)}")
                    continue
                
                if not query:
                    print("Inserisci una domanda")
                    continue
                
                self.search_results = []
                self.scraped_pages = []
                self.cleaned_query = ""
                
                result = self.process_query(query, self.current_domain, top_k=15)
                
                if not result:
                    print("❌ Nessun risultato trovato")
                
            except KeyboardInterrupt:
                print("\n👋 Arrivederci!")
                break
            except Exception as e:
                print(f"❌ Errore: {e}")

def main():
    parser = argparse.ArgumentParser(description='Qdrant Multi-Domain Search Bot with Streaming')
    parser.add_argument('--query', '-q', help='Domanda specifica')
    parser.add_argument('--domain', '-d', help='Dominio da cercare')
    parser.add_argument('--groq-key', '-k', help='Groq API key')
    parser.add_argument('--ai-model', '-m', default='openai/gpt-oss-20b')
    parser.add_argument('--top-k', '-t', type=int, default=15)
    parser.add_argument('--interactive', '-i', action='store_true')
    parser.add_argument('--qdrant-path', '-p', default='./qdrant_data', help='Path to Qdrant data')
    
    args = parser.parse_args()
    
    try:
        print("🚀 Inizializzazione Qdrant Multi-Domain Search Bot (STREAMING)...")
        
        bot = QdrantMultiDomainSearchBot(
            groq_api_key=args.groq_key,
            model_name=args.ai_model,
            qdrant_path=args.qdrant_path
        )
        
        if args.interactive or not args.query:
            bot.interactive_mode()
        else:
            if not args.domain:
                available_domains = bot.get_available_domains()
                if not available_domains:
                    print("❌ Nessun dominio trovato in Qdrant")
                    return 1
                domain = available_domains[0]['domain']
                print(f"Uso dominio di default: {domain}")
            else:
                domain = args.domain
            
            result = bot.process_query(args.query, domain, args.top_k)
            if result:
                print("✅ Query elaborata con successo")
            else:
                print("❌ Elaborazione query fallita")
            
    except KeyboardInterrupt:
        print("\n👋 Interrotto dall'utente")
    except Exception as e:
        print(f"💥 Errore critico: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

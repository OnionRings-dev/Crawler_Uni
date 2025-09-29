let currentRequestId = null;
        let statusCheckInterval = null;

        function showSection(section) {
            document.getElementById('landingSection').style.display = 'none';
            document.getElementById('searchSection').style.display = 'none';
            
            if (section === 'landing') {
                document.getElementById('landingSection').style.display = 'block';
            } else if (section === 'search') {
                document.getElementById('searchSection').style.display = 'block';
            }
            
            // Update navigation
            document.querySelectorAll('.nav-links a').forEach(link => {
                link.classList.remove('active');
            });
            
            if (section === 'landing') {
                document.querySelector('.nav-links a[onclick*="landing"]').classList.add('active');
            } else if (section === 'search') {
                document.querySelector('.nav-links a[onclick*="search"]').classList.add('active');
            }
            
            document.getElementById('navLinks').classList.remove('active');
        }

        function toggleMobileMenu() {
            document.getElementById('navLinks').classList.toggle('active');
        }

        function showStatus(message, isError = false, duration = 3000) {
            const statusBar = document.getElementById('statusBar');
            const statusText = document.getElementById('statusText');
            
            statusBar.className = isError ? 'status-bar show error' : 'status-bar show';
            statusText.textContent = message;
            
            setTimeout(() => {
                statusBar.classList.remove('show');
            }, duration);
        }

        function performSearch() {
            const input = document.getElementById('searchInput');
            const searchButton = document.getElementById('searchButton');
            const query = input.value.trim();
            
            if (!query) {
                showStatus('Inserisci una domanda', true);
                return;
            }

            // Show thinking indicator
            document.getElementById('thinkingIndicator').style.display = 'block';
            document.getElementById('resultsContainer').style.display = 'none';
            searchButton.disabled = true;
            searchButton.textContent = 'Elaborazione...';

            // Send to server
            fetch('/api/query', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: query})
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    hideThinking();
                    showStatus('Errore: ' + data.error, true);
                    resetSearchButton();
                } else {
                    currentRequestId = data.request_id;
                    statusCheckInterval = setInterval(checkStatus, 1000);
                }
            })
            .catch(error => {
                hideThinking();
                showStatus('Errore di connessione', true);
                resetSearchButton();
            });
        }

        function checkStatus() {
            fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'completed' && data.request_id === currentRequestId) {
                    clearInterval(statusCheckInterval);
                    hideThinking();
                    
                    if (data.result && data.result.latex_content) {
                        showResults(data.result.latex_content);
                    } else {
                        showStatus('Nessun risultato trovato', true);
                    }
                    
                    resetSearchButton();
                } else if (data.status === 'error') {
                    clearInterval(statusCheckInterval);
                    hideThinking();
                    showStatus('Errore durante l\'elaborazione', true);
                    resetSearchButton();
                }
            })
            .catch(error => {
                clearInterval(statusCheckInterval);
                hideThinking();
                showStatus('Errore di connessione', true);
                resetSearchButton();
            });
        }

        function hideThinking() {
            document.getElementById('thinkingIndicator').style.display = 'none';
        }

        function showResults(latexContent) {
            const resultsContent = document.getElementById('resultsContent');
            const resultsContainer = document.getElementById('resultsContainer');
            
            resultsContent.innerHTML = processLatexForDisplay(latexContent);
            resultsContainer.style.display = 'block';
            
            // Scroll to results
            resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        function processLatexForDisplay(latex) {
            if (!latex) return '<p>Contenuto vuoto</p>';
            
            let html = latex;
            
            // Remove LaTeX document structure
            html = html.replace(/\\documentclass\{[^}]+\}/g, '');
            html = html.replace(/\\usepackage\[[^\]]*\]\{[^}]+\}/g, '');
            html = html.replace(/\\usepackage\{[^}]+\}/g, '');
            html = html.replace(/\\begin\{document\}/g, '');
            html = html.replace(/\\end\{document\}/g, '');
            html = html.replace(/\\maketitle/g, '');
            html = html.replace(/\\title\{[^}]*\}/g, '');
            html = html.replace(/\\author\{[^}]*\}/g, '');
            html = html.replace(/\\date\{[^}]*\}/g, '');
            
            // Convert sections
            html = html.replace(/\\section\{([^}]+)\}/g, '<h2>$1</h2>');
            html = html.replace(/\\subsection\{([^}]+)\}/g, '<h3>$1</h3>');
            html = html.replace(/\\subsubsection\{([^}]+)\}/g, '<h4>$1</h4>');
            
            // Convert formatting
            html = html.replace(/\\textbf\{([^}]+)\}/g, '<strong>$1</strong>');
            html = html.replace(/\\textit\{([^}]+)\}/g, '<em>$1</em>');
            html = html.replace(/\\emph\{([^}]+)\}/g, '<em>$1</em>');
            
            // Convert lists
            html = html.replace(/\\begin\{itemize\}/g, '<ul>');
            html = html.replace(/\\end\{itemize\}/g, '</ul>');
            html = html.replace(/\\begin\{enumerate\}/g, '<ol>');
            html = html.replace(/\\end\{enumerate\}/g, '</ol>');
            html = html.replace(/\\item\s*/g, '<li>');
            
            // Convert links
            html = html.replace(/\\href\{([^}]+)\}\{([^}]+)\}/g, '<a href="$1" target="_blank">$2</a>');
            html = html.replace(/\\url\{([^}]+)\}/g, '<a href="$1" target="_blank">$1</a>');
            
            // Convert line breaks
            html = html.replace(/\\\\\s*/g, '<br>');
            html = html.replace(/\\par\s*/g, '</p><p>');
            html = html.replace(/\n\s*\n/g, '</p><p>');
            
            // Clean up
            html = html.replace(/\s+/g, ' ').trim();
            if (html && !html.startsWith('<')) html = '<p>' + html + '</p>';
            html = html.replace(/<p>\s*<\/p>/g, '');
            
            return html || '<p>Errore nella conversione</p>';
        }

        function resetSearchButton() {
            const searchButton = document.getElementById('searchButton');
            searchButton.disabled = false;
            searchButton.textContent = 'Cerca';
        }

        function clearSearch() {
            document.getElementById('searchInput').value = '';
            document.getElementById('resultsContainer').style.display = 'none';
            document.getElementById('thinkingIndicator').style.display = 'none';
            
            if (statusCheckInterval) {
                clearInterval(statusCheckInterval);
            }
            
            resetSearchButton();
            
            // Scroll back to input
            document.querySelector('.search-input-section').scrollIntoView({ 
                behavior: 'smooth', 
                block: 'center' 
            });
        }

        // Handle Enter key
        document.getElementById('searchInput').addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                performSearch();
            }
        });

        // Check server health
        window.onload = function() {
            fetch('/api/health')
            .then(response => response.json())
            .then(data => {
                if (data.status !== 'healthy') {
                    showStatus('Server non disponibile', true, 5000);
                }
            })
            .catch(error => {
                showStatus('Impossibile connettersi al server', true, 5000);
            });
        };

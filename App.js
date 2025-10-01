import React, { useState, useEffect, useRef } from 'react';

const API_BASE = 'http://localhost:5000';

export default function MultiDomainSearchApp() {
  const [domains, setDomains] = useState([]);
  const [selectedDomain, setSelectedDomain] = useState('');
  const [query, setQuery] = useState('');
  const [status, setStatus] = useState({ status: 'idle', message: '' });
  const [result, setResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentRequestId, setCurrentRequestId] = useState(null);
  const statusIntervalRef = useRef(null);

  useEffect(() => {
    loadDomains();
    return () => {
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
      }
    };
  }, []);

  const loadDomains = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/domains`);
      const data = await response.json();
      if (data.domains && data.domains.length > 0) {
        setDomains(data.domains);
        setSelectedDomain(data.domains[0]);
      }
    } catch (error) {
      console.error('Error loading domains:', error);
    }
  };

  const checkStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/status`);
      const data = await response.json();
      
      setStatus(data);

      if (data.status === 'completed') {
        if (statusIntervalRef.current) {
          clearInterval(statusIntervalRef.current);
          statusIntervalRef.current = null;
        }
        setIsProcessing(false);
        
        if (data.result && data.result.latex_content) {
          setResult(data.result);
        } else {
          setStatus({ 
            status: 'error', 
            message: 'Response completed but no content received' 
          });
        }
      } else if (data.status === 'error') {
        if (statusIntervalRef.current) {
          clearInterval(statusIntervalRef.current);
          statusIntervalRef.current = null;
        }
        setIsProcessing(false);
      }
    } catch (error) {
      console.error('Error checking status:', error);
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
        statusIntervalRef.current = null;
      }
      setIsProcessing(false);
      setStatus({ status: 'error', message: 'Failed to check status' });
    }
  };

  const handleSearch = async () => {
    if (!query.trim() || !selectedDomain) {
      alert('Please enter a query and select a domain');
      return;
    }

    setIsProcessing(true);
    setResult(null);
    setStatus({ status: 'processing', message: 'Sending query...' });

    try {
      const response = await fetch(`${API_BASE}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, domain: selectedDomain })
      });

      const data = await response.json();

      if (data.error) {
        setStatus({ status: 'error', message: data.error });
        setIsProcessing(false);
      } else {
        setCurrentRequestId(data.request_id);
        statusIntervalRef.current = setInterval(checkStatus, 1000);
      }
    } catch (error) {
      setStatus({ status: 'error', message: 'Connection error' });
      setIsProcessing(false);
    }
  };

  const handleClear = () => {
    setQuery('');
    setResult(null);
    setStatus({ status: 'idle', message: '' });
    if (statusIntervalRef.current) {
      clearInterval(statusIntervalRef.current);
      statusIntervalRef.current = null;
    }
    setIsProcessing(false);
    setCurrentRequestId(null);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !isProcessing && query.trim() && selectedDomain) {
      handleSearch();
    }
  };

  const processLatex = (latex) => {
    if (!latex) return '<p>Empty content</p>';
    
    let html = latex;
    
    // Remove LaTeX preamble and document class
    html = html.replace(/\\documentclass(\[[^\]]*\])?\{[^}]+\}/g, '');
    html = html.replace(/\\usepackage(\[[^\]]*\])?\{[^}]+\}/g, '');
    html = html.replace(/\\geometry\{[^}]+\}/g, '');
    html = html.replace(/\\begin\{document\}/g, '');
    html = html.replace(/\\end\{document\}/g, '');
    html = html.replace(/\\maketitle/g, '');
    html = html.replace(/\\title\{[^}]*\}/g, '');
    html = html.replace(/\\author\{[^}]*\}/g, '');
    html = html.replace(/\\date\{[^}]*\}/g, '');
    
    // Remove standalone braces and numbers with LaTeX commands
    html = html.replace(/^\s*\d+\.\s*\[label=\\[^\]]+\]\s*$/gm, '');
    html = html.replace(/^\s*\}\s*$/gm, '');
    html = html.replace(/\[label=\\[^\]]+\]/g, '');
    
    // Remove introduction sections
    html = html.replace(/\\section\*?\{Introduzione\}[\s\S]*?(?=\\section|$)/gi, '');
    html = html.replace(/\\section\*?\{Introduction\}[\s\S]*?(?=\\section|$)/gi, '');
    
    // Convert sections (with or without *)
    html = html.replace(/\\section\*\{([^}]+)\}/g, '<h2 style="margin-top: 30px; margin-bottom: 15px; font-size: 24px; font-weight: bold; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">$1</h2>');
    html = html.replace(/\\section\{([^}]+)\}/g, '<h2 style="margin-top: 30px; margin-bottom: 15px; font-size: 24px; font-weight: bold; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">$1</h2>');
    html = html.replace(/\\subsection\*\{([^}]+)\}/g, '<h3 style="margin-top: 20px; margin-bottom: 10px; font-size: 20px; font-weight: bold; color: #34495e;">$1</h3>');
    html = html.replace(/\\subsection\{([^}]+)\}/g, '<h3 style="margin-top: 20px; margin-bottom: 10px; font-size: 20px; font-weight: bold; color: #34495e;">$1</h3>');
    html = html.replace(/\\subsubsection\*\{([^}]+)\}/g, '<h4 style="margin-top: 15px; margin-bottom: 8px; font-size: 18px; font-weight: bold; color: #4a5568;">$1</h4>');
    html = html.replace(/\\subsubsection\{([^}]+)\}/g, '<h4 style="margin-top: 15px; margin-bottom: 8px; font-size: 18px; font-weight: bold; color: #4a5568;">$1</h4>');
    
    // Convert text formatting
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\\textbf\{([^}]+)\}/g, '<strong>$1</strong>');
    html = html.replace(/\\textit\{([^}]+)\}/g, '<em>$1</em>');
    html = html.replace(/\\emph\{([^}]+)\}/g, '<em>$1</em>');
    
    // Convert lists - improved handling
    html = html.replace(/\\begin\{itemize\}/g, '<ul style="margin: 15px 0; padding-left: 30px;">');
    html = html.replace(/\\end\{itemize\}/g, '</ul>');
    html = html.replace(/\\begin\{enumerate\}/g, '<ol style="margin: 15px 0; padding-left: 30px;">');
    html = html.replace(/\\end\{enumerate\}/g, '</ol>');
    
    // Handle list items with numbering patterns like [label=\arabic*.]
    html = html.replace(/\d+\.\s*\[label=[^\]]+\]/g, '');
    html = html.replace(/\\item\s*/g, '<li style="margin: 8px 0; line-height: 1.6;">');
    
    // Convert links
    html = html.replace(/\\href\{([^}]+)\}\{([^}]+)\}/g, '<a href="$1" target="_blank" style="color: #3498db; text-decoration: none; border-bottom: 1px solid #3498db;">$2</a>');
    html = html.replace(/\\url\{([^}]+)\}/g, '<a href="$1" target="_blank" style="color: #3498db; text-decoration: none; border-bottom: 1px solid #3498db;">$1</a>');
    
    // Remove any remaining LaTeX commands
    html = html.replace(/\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?/g, '');
    html = html.replace(/^\s*\}\s*$/gm, '');
    
    // Handle line breaks and paragraphs
    html = html.replace(/\\\\\s*/g, '<br>');
    html = html.replace(/\\par\s*/g, '</p><p style="margin: 15px 0; line-height: 1.6;">');
    
    // Convert double newlines to paragraphs
    html = html.replace(/\n\s*\n/g, '</p><p style="margin: 15px 0; line-height: 1.6;">');
    
    // Handle bullet points with * or -
    html = html.replace(/^\s*[\*\-]\s+(.+)$/gm, '<li style="margin: 8px 0; line-height: 1.6;">$1</li>');
    
    // Clean up whitespace
    html = html.replace(/\s+/g, ' ').trim();
    
    // Wrap content in paragraph if it doesn't start with a tag
    if (html && !html.startsWith('<')) {
      html = '<p style="margin: 15px 0; line-height: 1.6;">' + html + '</p>';
    }
    
    // Remove empty paragraphs
    html = html.replace(/<p[^>]*>\s*<\/p>/g, '');
    
    // Close any unclosed list items
    html = html.replace(/<li([^>]*)>([^<]*?)(?=<li|<\/ul>|<\/ol>|$)/g, '<li$1>$2</li>');
    
    return html || '<p>Error in conversion</p>';
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '20px', fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      <h1 style={{ fontSize: '32px', marginBottom: '10px' }}>Multi-Domain Search Bot</h1>
      <p style={{ color: '#666', marginBottom: '30px' }}>Search across multiple university domains</p>

      <div style={{ marginBottom: '30px', padding: '20px', border: '1px solid #ddd', borderRadius: '8px', backgroundColor: '#ffffff' }}>
        <div style={{ marginBottom: '15px' }}>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
            Select Domain:
          </label>
          <select
            value={selectedDomain}
            onChange={(e) => setSelectedDomain(e.target.value)}
            disabled={isProcessing}
            style={{ width: '100%', padding: '10px', fontSize: '16px', borderRadius: '4px', border: '1px solid #ccc' }}
          >
            {domains.map(domain => (
              <option key={domain} value={domain}>{domain}</option>
            ))}
          </select>
        </div>

        <div style={{ marginBottom: '15px' }}>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
            Your Question:
          </label>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Es. Come faccio a iscrivermi?"
            disabled={isProcessing}
            style={{ width: '100%', padding: '10px', fontSize: '16px', borderRadius: '4px', border: '1px solid #ccc' }}
          />
        </div>

        <button
          onClick={handleSearch}
          disabled={isProcessing || !query.trim() || !selectedDomain}
          style={{
            padding: '12px 24px',
            fontSize: '16px',
            backgroundColor: isProcessing ? '#ccc' : '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: isProcessing ? 'not-allowed' : 'pointer',
            marginRight: '10px'
          }}
        >
          {isProcessing ? 'Processing...' : 'Search'}
        </button>

        {result && (
          <button
            onClick={handleClear}
            style={{
              padding: '12px 24px',
              fontSize: '16px',
              backgroundColor: '#6c757d',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            New Search
          </button>
        )}
      </div>
    
      {status.status !== 'idle' && (
        <div style={{ 
          padding: '15px', 
          marginBottom: '20px', 
          backgroundColor: status.status === 'error' ? '#f8d7da' : '#d1ecf1', 
          border: '1px solid ' + (status.status === 'error' ? '#f5c6cb' : '#bee5eb'), 
          borderRadius: '4px' 
        }}>
          <strong>Status:</strong> {status.message || status.status}
          {status.progress > 0 && ` (${status.progress}%)`}
        </div>
      )}

      {result && (
        <div style={{ padding: '25px', border: '1px solid #ddd', borderRadius: '8px', backgroundColor: '#ffffff', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
          <div 
            style={{ 
              lineHeight: '1.7',
              fontSize: '16px',
              color: '#333'
            }}
            dangerouslySetInnerHTML={{ __html: processLatex(result.latex_content) }}
          />
        </div>
      )}
    </div>
  );
}
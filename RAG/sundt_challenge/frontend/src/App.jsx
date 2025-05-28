import React, { useState, useEffect, useRef } from 'react';
import './App.css';

// Function to convert markdown formatting to HTML
const convertMarkdownToHtml = (text) => {
  if (!text) return text;
  
  let html = text;
  
  // Convert bold (**text** or __text__)
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/__(.*?)__/g, '<strong>$1</strong>');
  
  // Convert italic (*text* or _text_)
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
  html = html.replace(/_(.*?)_/g, '<em>$1</em>');
  
  // Convert bullet points (simple approach)
  // Split by lines and process
  const lines = html.split('\n');
  let htmlLines = [];
  let inList = false;
  
  for (let line of lines) {
    if (line.trim().startsWith('* ') || line.trim().startsWith('- ')) {
      if (!inList) {
        htmlLines.push('<ul>');
        inList = true;
      }
      // Remove the bullet point marker and create list item
      const listContent = line.trim().substring(2).trim();
      htmlLines.push(`<li>${listContent}</li>`);
    } else {
      if (inList) {
        htmlLines.push('</ul>');
        inList = false;
      }
      // Convert line breaks to <br> for non-list items
      if (line.trim() === '') {
        htmlLines.push('<br>');
      } else {
        htmlLines.push(line);
      }
    }
  }
  
  // Close any open list
  if (inList) {
    htmlLines.push('</ul>');
  }
  
  html = htmlLines.join('\n');
  
  // Convert double line breaks to paragraph breaks
  html = html.replace(/\n\n/g, '<br><br>');
  // Convert single line breaks to <br>
  html = html.replace(/\n/g, '<br>');
  
  return html;
};

function App() {
  const [messages, setMessages] = useState([
    { sender: 'ai', text: "Hello! Ask me about Sundt's projects or awards.", sources: [] }
  ]);
  const [inputText, setInputText] = useState('');
  const [selectedAI, setSelectedAI] = useState('projects'); // 'projects' or 'awards'
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const backendBaseUrl = 'http://localhost:8000/api/v1/rag/ask'; // Updated base URL

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleInputChange = (e) => {
    setInputText(e.target.value);
  };

  const handleSelectChange = (e) => {
    setSelectedAI(e.target.value);
    setMessages([
      { sender: 'ai', text: `Switched to ${e.target.value} AI. How can I help?`, sources: [] }
    ]);
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    const userMessage = { sender: 'user', text: inputText };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      const endpoint = selectedAI === 'projects' ? `${backendBaseUrl}/projects` : `${backendBaseUrl}/awards`; // Construct full endpoint
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: userMessage.text }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Unknown error occurred" }));
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorData.detail}`);
      }

      const data = await response.json();
      const aiMessage = { 
        sender: 'ai', 
        text: data.answer,
        sources: data.source_documents || [] // Updated to match backend response structure
      };
      setMessages(prevMessages => [...prevMessages, aiMessage]);

    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = { sender: 'ai', text: `Sorry, I encountered an error: ${error.message}`, sources: [] };
      setMessages(prevMessages => [...prevMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        <h2>Sundt Company Information AI</h2>
        <div className="ai-selector">
          <label htmlFor="ai-select">Chat with: </label>
          <select id="ai-select" value={selectedAI} onChange={handleSelectChange}>
            <option value="projects">Projects AI</option>
            <option value="awards">Awards AI</option>
          </select>
        </div>
      </header>
      <div className="chat-messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}-message`}>
            <div 
              dangerouslySetInnerHTML={{ 
                __html: msg.sender === 'ai' ? convertMarkdownToHtml(msg.text) : msg.text 
              }} 
            />
            {msg.sources && msg.sources.length > 0 && (
              <div className="sources">
                <strong>Sources:</strong>
                <ul>
                  {msg.sources.map((source, i) => (
                    <li key={i}>
                      <p><strong>Content:</strong> {source.page_content.substring(0,150)}...</p>
                      <p><em>Source: {source.metadata.source} (Page: {source.metadata.page_number !== undefined ? source.metadata.page_number : 'N/A'}, Type: {source.metadata.type})</em></p>
                      {source.url && (
                        <p><a href={source.url} target="_blank" rel="noopener noreferrer">View Source</a></p>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <form className="chat-input-form" onSubmit={handleSendMessage}>
        <input
          type="text"
          id="chat-input"
          placeholder="Type your message..."
          value={inputText}
          onChange={handleInputChange}
          disabled={isLoading}
        />
        <button type="submit" id="send-button" disabled={isLoading}>
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
}

export default App;

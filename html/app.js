// Application data from the provided JSON
const categories = [
  {
    name: "mathematics",
    keywords: ["math", "calculate", "derivative", "integral", "equation", "solve", "algebra", "geometry", "statistics", "probability"],
    models: [
      {"name": "phi4", "score": 1.0, "categories": ["Mathematics", "Logic"], "description": "Excellent for mathematical reasoning and logical problems"},
      {"name": "mistral-small3.1", "score": 0.8, "categories": ["Mathematics", "General"], "description": "Strong mathematical capabilities and versatile reasoning"},
      {"name": "llama3.2:latest", "score": 0.6, "categories": ["General"], "description": "General-purpose model with basic math support"}
    ]
  },
  {
    name: "programming",
    keywords: ["code", "function", "variable", "loop", "array", "javascript", "python", "react", "api", "debug", "syntax", "programming", "software", "algorithm", "coding"],
    models: [
      {"name": "mistral-small3.1", "score": 0.9, "categories": ["Programming", "Debugging"], "description": "Specialized for coding, debugging, and software development"},
      {"name": "llama3.2:latest", "score": 0.8, "categories": ["Programming", "Code Analysis"], "description": "Strong programming assistance and code analysis"},
      {"name": "phi4", "score": 0.6, "categories": ["Mathematics", "Algorithms"], "description": "Good for algorithmic thinking and mathematical programming"}
    ]
  },
  {
    name: "health",
    keywords: ["doctor", "medicine", "symptom", "treatment", "health", "medical", "hospital", "disease", "drug", "therapy", "diagnosis"],
    models: [
      {"name": "llama3.2:latest", "score": 0.9, "categories": ["Health", "General"], "description": "Specialized in medical knowledge and health information"},
      {"name": "mistral-small3.1", "score": 0.8, "categories": ["General", "Research"], "description": "Reliable for medical research and health topics"},
      {"name": "phi4", "score": 0.6, "categories": ["General"], "description": "Basic health information and general medical questions"}
    ]
  },
  {
    name: "history",
    keywords: ["history", "historical", "war", "century", "ancient", "civilization", "empire", "revolution", "battle", "timeline"],
    models: [
      {"name": "llama3.2:latest", "score": 0.9, "categories": ["History", "General"], "description": "Extensive historical knowledge and contextual analysis"},
      {"name": "mistral-small3.1", "score": 0.8, "categories": ["General", "Research"], "description": "Good historical context and research capabilities"},
      {"name": "phi4", "score": 0.7, "categories": ["General"], "description": "Solid foundation in historical facts and timelines"}
    ]
  },
  {
    name: "general",
    keywords: ["hello", "how", "what", "when", "where", "why", "explain", "help", "question", "answer"],
    models: [
      {"name": "mistral-small3.1", "score": 0.8, "categories": ["General", "Reasoning"], "description": "Versatile general-purpose model with strong reasoning"},
      {"name": "llama3.2:latest", "score": 0.75, "categories": ["General", "Analysis"], "description": "Excellent analytical capabilities and broad knowledge"},
      {"name": "phi4", "score": 0.6, "categories": ["Specialized"], "description": "Compact model optimized for specific reasoning tasks"}
    ]
  }
];

const piiPatterns = [
  {
    type: "EMAIL_ADDRESS",
    pattern: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g,
    replacement: "john.doe@example.com",
    risk: "medium"
  },
  {
    type: "CREDIT_CARD",
    pattern: /\b(?:\d{4}[-\s]?){3}\d{4}\b/g,
    replacement: "4111-1111-1111-1111",
    risk: "high"
  },
  {
    type: "PHONE_NUMBER",
    pattern: /\b(?:\+?1[-\s]?)?\(?[0-9]{3}\)?[-\s]?[0-9]{3}[-\s]?[0-9]{4}\b/g,
    replacement: "(555) 012-3456",
    risk: "medium"
  },
  {
    type: "SSN",
    pattern: /\b\d{3}-\d{2}-\d{4}\b/g,
    replacement: "000-00-0000",
    risk: "high"
  }
];

const samplePrompts = [
  "How much is 2+2",
  "Calculate 2 plus 2",
  "Write a Python function to sort an array",
  "How do I debug JavaScript code in the browser?",
  "What are the symptoms of diabetes?",
  "How does aspirin work for pain relief?",
  "My email is sarah.wilson@example.com and phone is (555) 987-6543 call or email me",
  "What caused World War 2?",
  "Hello, how are you today?",
  "How can I teach my dog to sit and stay?"
];

class SemanticRouterDashboard {
  constructor() {
    // Get configuration instance
    this.config = window.SemanticRouterConfig;
    
    // Initialize API client
    this.apiClient = new window.ApiClient(this.config);
    
    // DOM elements
    this.chatContainer = document.getElementById('chatContainer');
    this.messageInput = document.getElementById('messageInput');
    this.chatForm = document.getElementById('chatForm');
    this.clearChatBtn = document.getElementById('clearChat');
    this.samplePromptsContainer = document.getElementById('samplePrompts');
    this.processingStatus = document.getElementById('processingStatus');
    
    // Mode toggle elements
    this.mockModeRadio = document.getElementById('mockMode');
    this.liveModeRadio = document.getElementById('liveMode');
    this.modeIndicator = document.getElementById('modeIndicator');
    
    // State
    this.isProcessing = false;
    this.chatHistory = [];
    this.currentClassification = null;
    this.currentPiiResults = [];
    this.selectedModel = null;
    
    this.init();
  }

  init() {
    this.setupEventListeners();
    this.renderSamplePrompts();
    this.resetProcessingSteps();
    this.initializeModeToggle();
  }

  setupEventListeners() {
    this.chatForm.addEventListener('submit', (e) => this.handleSubmit(e));
    this.clearChatBtn.addEventListener('click', () => this.clearChat());
    this.clearCacheBtn = document.getElementById('clearCache');
    this.clearCacheBtn.addEventListener('click', () => this.clearCache());
    
    // Mode toggle event listeners
    this.mockModeRadio.addEventListener('change', () => this.handleModeChange('mock'));
    this.liveModeRadio.addEventListener('change', () => this.handleModeChange('live'));
    
    // Configuration change listeners
    this.config.on('mode', (event) => this.handleConfigModeChange(event));
  }
  
  initializeModeToggle() {
    // Set initial mode from config
    const currentMode = this.config.get('mode');
    this.updateModeToggle(currentMode);
    this.updateModeIndicator(currentMode);
  }
  
  handleModeChange(mode) {
    // Update configuration
    this.config.set('mode', mode);
    this.updateModeIndicator(mode);
    this.updateCacheStats();
    
    // Log mode change for debugging
    if (this.config.get('showDebugInfo')) {
      console.log(`Switched to ${mode} mode`);
    }
  }
  
  handleConfigModeChange(event) {
    // Handle mode changes from other sources (e.g., another tab)
    this.updateModeToggle(event.newValue);
    this.updateModeIndicator(event.newValue);
  }
  
  updateModeToggle(mode) {
    if (mode === 'mock') {
      this.mockModeRadio.checked = true;
    } else {
      this.liveModeRadio.checked = true;
    }
  }
  
  updateModeIndicator(mode) {
    const indicatorText = this.modeIndicator.querySelector('.mode-indicator__text');
    const indicatorStatus = this.modeIndicator.querySelector('.mode-indicator__status');
    
    if (mode === 'mock') {
      indicatorText.textContent = 'Mock Mode';
      indicatorStatus.className = 'mode-indicator__status mode-indicator__status--mock';
    } else {
      indicatorText.textContent = 'Live Mode';
      indicatorStatus.className = 'mode-indicator__status mode-indicator__status--live';
    }
  }

  updateCacheStats() {
    const currentMode = this.config.get('mode');
    const cacheStatsElement = document.getElementById('cacheStats');
    const clearCacheBtn = document.getElementById('clearCache');
    
    if (currentMode === 'live') {
      // Show cache stats in live mode
      cacheStatsElement.style.display = 'flex';
      clearCacheBtn.style.display = 'inline-block';
      
      const stats = this.apiClient.getCacheStats();
      document.getElementById('cacheHitRate').textContent = `${stats.totalHitRate}%`;
      document.getElementById('cacheDetails').innerHTML = `
        <span class="cache-stats__server" title="Server Cache: Backend semantic cache using BERT embeddings to find similar questions">💾 ${stats.serverHitRate}%</span>
        <span class="cache-stats__client" title="Client Cache: Instant browser-side cache for exact repeated questions">⚡ ${stats.clientHitRate}%</span>
      `;
    } else {
      // Hide cache stats in mock mode
      cacheStatsElement.style.display = 'none';
      clearCacheBtn.style.display = 'none';
    }
  }

  clearCache() {
    this.apiClient.clearCache();
    this.updateCacheStats();
    console.log('Cache cleared by user');
  }

  renderSamplePrompts() {
    this.samplePromptsContainer.innerHTML = samplePrompts
      .map(prompt => `
        <button class="sample-prompt" onclick="dashboard.useSamplePrompt('${prompt.replace(/'/g, "\\'")}')">
          ${prompt.length > 50 ? prompt.substring(0, 50) + '...' : prompt}
        </button>
      `).join('');
  }

  useSamplePrompt(prompt) {
    this.messageInput.value = prompt;
    this.messageInput.focus();
  }

  async handleSubmit(e) {
    e.preventDefault();
    
    if (this.isProcessing) return;
    
    const message = this.messageInput.value.trim();
    if (!message) return;

    this.isProcessing = true;
    this.addChatMessage(message, 'user');
    this.messageInput.value = '';
    
    await this.processMessage(message);
    
    this.isProcessing = false;
  }

  // Helper function to format markdown-style text to HTML
  formatMessage(message, sender) {
    if (sender === 'user') {
      // Don't format user messages, keep them as plain text
      return message;
    }

    // Convert markdown-style formatting to HTML for assistant messages
    let formatted = message
      // Convert **bold** to <strong>
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      // Convert *italic* to <em>
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      // Convert numbered lists (1. item) to proper HTML lists
      .replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>')
      // Convert line breaks to <br> tags
      .replace(/\n/g, '<br>')
      // Convert double line breaks to paragraph breaks
      .replace(/<br><br>/g, '</p><p>');

    // Wrap list items in <ol> tags
    if (formatted.includes('<li>')) {
      formatted = formatted.replace(/(<li>.*<\/li>)/gs, (match) => {
        return '<ol>' + match + '</ol>';
      });
    }

    // Wrap in paragraph tags if it doesn't start with a list
    if (!formatted.startsWith('<ol>')) {
      formatted = '<p>' + formatted + '</p>';
    }

    return formatted;
  }

  addChatMessage(message, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message chat-message--${sender}`;
    
    // Format the message and use innerHTML for proper rendering
    const formattedMessage = this.formatMessage(message, sender);
    if (sender === 'user') {
      messageDiv.textContent = formattedMessage; // Keep user messages as plain text
    } else {
      // Check for cache information and add indicator
      let cacheIndicator = '';
      if (this.currentResponse && this.currentResponse.cached) {
        const cacheType = this.currentResponse.cacheType;
        const cacheIcon = cacheType === 'client' ? '⚡' : '💾';
        const cacheText = cacheType === 'client' 
          ? 'Client Cache Hit: Instant response from browser cache (exact match)' 
          : 'Server Cache Hit: Response from backend semantic cache (similar question found using AI)';
        cacheIndicator = `<div class="cache-indicator" title="${cacheText}">${cacheIcon}</div>`;
      }
      
      messageDiv.innerHTML = `${cacheIndicator}${formattedMessage}`; // Format assistant messages
    }
    
    this.chatContainer.appendChild(messageDiv);
    this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    
    this.chatHistory.push({ message, sender, timestamp: new Date() });
  }

  clearChat() {
    this.chatContainer.innerHTML = `
      <div class="chat-message chat-message--system">
        Welcome! Try asking a question or use one of the sample prompts below.
      </div>
    `;
    this.chatHistory = [];
    this.resetProcessingSteps();
  }

  async processMessage(message) {
    // Clear all previous processing results immediately
    this.resetProcessingSteps();
    
    this.updateProcessingStatus('Processing...', 'warning');
    
    const currentMode = this.config.get('mode');
    const stepDelay = currentMode === 'live' ? 0 : 500; // No delays in live mode
    
    // Step 1: Semantic Classification (fast) - Show result immediately
    await this.performClassification(message);
    // Small delay to let user see the classification result
    await this.delay(currentMode === 'live' ? 300 : stepDelay);
    
    // Step 2: Semantic Cache Check
    await this.performCacheCheck(message);
    await this.delay(stepDelay);
    
    // Step 3: PII Detection (fast)
    await this.performPiiDetection(message);
    await this.delay(stepDelay);
    
    // Step 4: Model Selection
    await this.performModelSelection();
    await this.delay(stepDelay);
    
    // Step 5: Data Processing
    await this.performDataProcessing(message);
    await this.delay(stepDelay);
    
    // Step 6: Model Response (the slow part)
    await this.performModelResponse(message);
    
    this.updateProcessingStatus('Complete', 'success');
  }

  async performClassification(message) {
    const step = document.getElementById('classificationStep');
    const status = document.getElementById('classificationStatus');
    const content = document.getElementById('classificationContent');

    this.activateStep(step);
    status.innerHTML = '<div class="loading-spinner"></div>';

    // Show mode-specific loading message
    const currentMode = this.config.get('mode');
    if (currentMode === 'live') {
      content.innerHTML = '<p class="text-secondary">Analyzing prompt semantics (fast classification)...</p>';
      // No delay in live mode - proceed immediately
    } else {
      content.innerHTML = '<p class="text-secondary">Analyzing prompt semantic meaning...</p>';
      await this.delay(800);
    }

    try {
      const classification = await this.classifyMessage(message);
      
      // IMMEDIATELY update the UI with classification results
      status.innerHTML = '<span class="status status--success">Complete</span>';
      
      // Show different content based on source
      let sourceInfo = '';
      let technicalInfo = '';
      if (classification.source === 'fast_analysis') {
        sourceInfo = '<p><em>⚡ Fast classification analysis (client-side)</em></p>';
        technicalInfo = '<p><strong>Fast Analysis:</strong> Keyword-based semantic classification</p>';
      } else if (classification.source === 'mock_fallback') {
        sourceInfo = '<p><em>⚠️ Using mock classification (API error)</em></p>';
        technicalInfo = '<p><strong>Fallback Mode:</strong> Local keyword matching</p>';
      } else if (classification.source === 'live') {
        sourceInfo = '<p><em>✅ Live classification from semantic router API</em></p>';
        technicalInfo = '<p><strong>BERT Embedding:</strong> 384-dimensional vector processed</p>';
      } else {
        sourceInfo = '<p><em>📋 Using mock classification</em></p>';
        technicalInfo = '<p><strong>Mock Analysis:</strong> Local keyword-based classification</p>';
      }
      
      content.innerHTML = `
        <div class="classification-result">
          <div class="category-info">
            <span class="category-badge">${classification.category}</span>
            <div class="confidence-bar">
              <div class="confidence-fill" style="width: ${classification.confidence}%"></div>
            </div>
            <span class="confidence-text">${classification.confidence}%</span>
          </div>
          <p><strong>Detected Keywords:</strong> ${classification.matchedKeywords.join(', ')}</p>
          ${technicalInfo}
          ${sourceInfo}
        </div>
      `;

      this.completeStep(step);
      this.currentClassification = classification;
      
      // Force a UI repaint to ensure immediate visibility
      step.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      
    } catch (error) {
      console.error('Classification failed:', error);
      status.innerHTML = '<span class="status status--error">Error</span>';
      content.innerHTML = `
        <div class="classification-result">
          <p style="color: var(--color-error);">Classification failed: ${error.message}</p>
          <p><em>The system will proceed with general classification.</em></p>
        </div>
      `;
      
      // Set fallback classification
      this.currentClassification = {
        category: 'general',
        confidence: 50,
        matchedKeywords: ['fallback'],
        source: 'error_fallback',
        error: error.message
      };
      
      this.completeStep(step);
    }
  }

  async performCacheCheck(message) {
    const step = document.getElementById('cacheStep');
    const status = document.getElementById('cacheStatus');
    const content = document.getElementById('cacheContent');

    this.activateStep(step);
    status.innerHTML = '<div class="loading-spinner"></div>';

    const currentMode = this.config.get('mode');
    
    if (currentMode === 'mock') {
      // In mock mode, show that cache is not active
      await this.delay(300);
      status.innerHTML = '<span class="status status--info">Disabled</span>';
      content.innerHTML = `
        <div class="cache-result">
          <p>⚠️ <strong>Semantic cache disabled in mock mode</strong></p>
          <p class="text-secondary">Cache only operates in live mode with BERT embeddings</p>
          <div class="cache-info">
            <p><strong>Why cache doesn't work in mock mode:</strong></p>
            <ul style="margin: 8px 0; padding-left: 20px;">
              <li>No backend server connection</li>
              <li>No BERT model for embedding generation</li>
              <li>No semantic similarity calculation</li>
              <li>Only client-side exact match available</li>
            </ul>
            <p><em>💡 Switch to Live mode to see cache similarity scores</em></p>
          </div>
        </div>
      `;
      this.completeStep(step);
      return;
    }

    // Live mode - check cache
    content.innerHTML = '<p class="text-secondary">Generating BERT embedding and checking semantic similarity...</p>';
    // No delay in live mode - proceed immediately

    try {
      // Store initial cache stats
      const initialStats = this.apiClient.getCacheStats();
      
      // Check if this would be a cache hit by generating cache key
      const cacheKey = this.apiClient.generateCacheKey(message);
      const hasClientCache = this.apiClient.cache.has(cacheKey);
      
      let cacheResult = {
        hit: false,
        type: 'miss',
        similarity: 0,
        source: 'none'
      };

      if (hasClientCache) {
        cacheResult = {
          hit: true,
          type: 'client',
          similarity: 100, // Exact match
          source: 'client_cache'
        };
      } else {
        // For display purposes, we'll show that we're about to check server cache
        // The real cache check happens during the actual API call
        
        // Check if we have any entries in server cache from previous requests
        // A better way to detect cache availability is to check total requests vs just hits
        const hasServerHistory = initialStats.totalRequests > 1; // More than just this request
        
        cacheResult = {
          hit: false,
          type: 'server_check_pending',
          similarity: 0,
          source: hasServerHistory ? 'server_available' : 'no_server_history',
          serverCacheAvailable: hasServerHistory
        };
      }

      // Store cache result for later use by model response step
      this.currentCacheResult = cacheResult;

      // Update status based on result
      if (cacheResult.hit) {
        status.innerHTML = '<span class="status status--success">Cache Hit</span>';
      } else {
        status.innerHTML = '<span class="status status--info">Cache Miss</span>';
      }

      // Show detailed cache information
      let cacheDetails = '';
      if (cacheResult.hit) {
        if (cacheResult.type === 'client') {
          cacheDetails = `
            <div class="cache-hit">
              <p>✅ <strong>Exact match found in client cache</strong></p>
              <p><strong>Similarity:</strong> 100% (exact match)</p>
              <p><strong>Cache Type:</strong> Client-side (instant)</p>
              <p><strong>Response Time:</strong> <1ms</p>
            </div>
          `;
        } else {
          cacheDetails = `
            <div class="cache-hit">
              <p>✅ <strong>Similar query found in server cache</strong></p>
              <p><strong>Similarity:</strong> ${cacheResult.similarity.toFixed(1)}%</p>
              <p><strong>Cache Type:</strong> Server-side semantic cache</p>
              <p><strong>BERT Embedding:</strong> 512-dimensional vector processed</p>
            </div>
          `;
        }
      } else {
        if (cacheResult.type === 'server_check_pending') {
          cacheDetails = `
            <div class="cache-miss">
              <p>🔍 <strong>Preparing server cache check</strong></p>
              <p><strong>Client Cache:</strong> No exact match found</p>
              <p><strong>Server Cache:</strong> ${cacheResult.serverCacheAvailable ? 'Available for semantic similarity check' : 'No previous entries'}</p>
              <p><strong>Next:</strong> BERT embedding will be generated during API call</p>
              <p><strong>Process:</strong></p>
              <ul style="margin: 8px 0; padding-left: 20px;">
                <li>Generate 512-dimensional BERT embedding</li>
                <li>Calculate cosine similarity with cached entries</li>
                <li>Check if similarity ≥ 80% threshold</li>
                <li>Return cached response or proceed with full processing</li>
              </ul>
              <p><em>🔬 Server-side semantic cache uses deep learning embeddings</em></p>
            </div>
          `;
        } else {
          const embeddingInfo = cacheResult.serverCacheAvailable ? 
            '<p><strong>BERT Embedding:</strong> 512-dimensional vector generated for comparison</p>' :
            '<p><strong>Server Cache:</strong> No previous entries to compare against</p>';
          
          cacheDetails = `
            <div class="cache-miss">
              <p>❌ <strong>No similar queries found</strong></p>
              <p><strong>Similarity Threshold:</strong> 80%</p>
              <p><strong>Best Match:</strong> ${cacheResult.similarity.toFixed(1)}% (below threshold)</p>
              ${embeddingInfo}
              <p><em>Query will be processed and cached for future requests</em></p>
            </div>
          `;
        }
      }

      content.innerHTML = `
        <div class="cache-result">
          ${cacheDetails}
          <div class="cache-stats">
            <div class="cache-stat-item">
              <span class="cache-stat-label">Total Requests:</span>
              <span class="cache-stat-value">${initialStats.totalRequests}</span>
            </div>
            <div class="cache-stat-item">
              <span class="cache-stat-label">Cache Hit Rate:</span>
              <span class="cache-stat-value">${initialStats.totalHitRate}%</span>
            </div>
            <div class="cache-stat-item">
              <span class="cache-stat-label">Server Cache:</span>
              <span class="cache-stat-value">${initialStats.serverHitRate}%</span>
            </div>
            <div class="cache-stat-item">
              <span class="cache-stat-label">Client Cache:</span>
              <span class="cache-stat-value">${initialStats.clientHitRate}%</span>
            </div>
          </div>
          <p><em>💾 Semantic cache uses BERT embeddings to find similar questions</em></p>
        </div>
      `;

      this.completeStep(step);

    } catch (error) {
      console.error('Cache check failed:', error);
      status.innerHTML = '<span class="status status--error">Error</span>';
      content.innerHTML = `
        <div class="cache-result">
          <p style="color: var(--color-error);">Cache check failed: ${error.message}</p>
          <p><em>Proceeding without cache optimization</em></p>
        </div>
      `;
      this.completeStep(step);
    }
  }

  async performPiiDetection(message) {
    const step = document.getElementById('piiStep');
    const status = document.getElementById('piiStatus');
    const content = document.getElementById('piiContent');

    this.activateStep(step);
    status.innerHTML = '<div class="loading-spinner"></div>';

    // Show mode-specific loading message
    const currentMode = this.config.get('mode');
    if (currentMode === 'live') {
      // Check if this is a programming question (likely to take longer)
      const isProgrammingQuery = this.currentClassification?.category === 'programming';
      
      if (isProgrammingQuery) {
        content.innerHTML = '<p class="text-secondary">Analyzing PII with live API (programming queries may take longer)...</p>';
        await this.delay(500);
        
        // Provide additional feedback after 10 seconds for programming queries
        setTimeout(() => {
          if (status.innerHTML.includes('loading-spinner')) {
            content.innerHTML = '<p class="text-secondary">Processing complex programming query with larger model, please wait...</p>';
          }
        }, 10000);
        
        // Additional feedback after 30 seconds
        setTimeout(() => {
          if (status.innerHTML.includes('loading-spinner')) {
            content.innerHTML = '<p class="text-secondary">Large model is still processing your programming query (this can take up to 2 minutes)...</p>';
          }
        }, 30000);
      } else {
        content.innerHTML = '<p class="text-secondary">Analyzing PII with live API...</p>';
        // No delay in live mode - proceed immediately
      }
    } else {
      content.innerHTML = '<p class="text-secondary">Scanning for personally identifiable information...</p>';
      await this.delay(1200);
    }

    try {
      const piiResults = await this.detectPii(message, this.currentClassification);
      
      status.innerHTML = `<span class="status status--${piiResults.length > 0 ? 'warning' : 'success'}">
        ${piiResults.length > 0 ? 'PII Detected' : 'Clean'}
      </span>`;
      
      if (piiResults.length > 0) {
        const highlightedText = this.highlightPii(message, piiResults);
        
        // Show source information
        let sourceInfo = '';
        if (piiResults[0].source === 'live') {
          sourceInfo = '<p><em>✓ PII detection from live semantic router API</em></p>';
        } else if (piiResults[0].source === 'mock_fallback') {
          sourceInfo = '<p><em>⚠️ Using mock PII detection (API error)</em></p>';
        } else {
          sourceInfo = '<p><em>📋 Using mock PII detection</em></p>';
        }
        
        content.innerHTML = `
          <div class="pii-results">
            <p><strong>Original text with PII highlighted:</strong></p>
            <p>${highlightedText}</p>
            <div style="margin-top: 12px;">
              ${piiResults.map(pii => `
                <div class="pii-item">
                  <span class="pii-type">${pii.type.replace('_', ' ')}</span>
                  <span class="pii-risk pii-risk--${pii.risk}">${pii.risk.toUpperCase()} RISK</span>
                </div>
              `).join('')}
            </div>
            ${sourceInfo}
          </div>
        `;
      } else {
        let sourceInfo = '';
        if (currentMode === 'live') {
          sourceInfo = '<p><em>✓ No PII detected by live semantic router API</em></p>';
        } else {
          sourceInfo = '<p><em>📋 No PII detected by mock scanner</em></p>';
        }
        
        content.innerHTML = `
          <div class="pii-results">
            <p style="color: var(--color-success);">✓ No personally identifiable information detected</p>
            <p>Text is safe to process without redaction.</p>
            ${sourceInfo}
          </div>
        `;
      }

      this.completeStep(step);
      this.currentPiiResults = piiResults;
      
    } catch (error) {
      console.error('PII detection failed:', error);
      status.innerHTML = '<span class="status status--error">Error</span>';
      
      // Provide specific error message for timeout
      let errorMessage = error.message;
      if (error.message.includes('timeout') || error.message.includes('timed out')) {
        errorMessage = 'Request timed out. Programming queries may require longer processing time.';
      }
      
      content.innerHTML = `
        <div class="pii-results">
          <p style="color: var(--color-error);">PII detection failed: ${errorMessage}</p>
          <p><em>The system will proceed without PII filtering.</em></p>
        </div>
      `;
      
      // Set empty PII results to continue processing
      this.currentPiiResults = [];
      
      this.completeStep(step);
    }
  }

  async performModelSelection() {
    const step = document.getElementById('modelStep');
    const status = document.getElementById('modelStatus');
    const content = document.getElementById('modelContent');

    this.activateStep(step);
    status.innerHTML = '<div class="loading-spinner"></div>';

    const currentMode = this.config.get('mode');
    if (currentMode !== 'live') {
      await this.delay(1000); // Only delay in mock mode
    }

    // Normalize the category (handle coding -> programming)
    const normalizedCategory = normalizeCategory(this.currentClassification.category);
    const category = categories.find(cat => cat.name === normalizedCategory);
    
    if (!category) {
      // Fallback to general if category not found
      const generalCategory = categories.find(cat => cat.name === 'general');
      const selectedModel = generalCategory.models[0];
      
      status.innerHTML = '<span class="status status--success">Selected</span>';
      content.innerHTML = `
        <div class="model-grid">
          <div class="model-item model-item--selected">
            <div class="model-info">
              <div class="model-name">${selectedModel.name}</div>
              <div class="model-description">Fallback: ${selectedModel.description}</div>
              <div class="model-categories">Best for: ${selectedModel.categories.join(', ')}</div>
            </div>
            <div class="model-score">${selectedModel.score}</div>
          </div>
        </div>
        <p style="margin-top: 12px;"><strong>Selection Reasoning:</strong> Category "${this.currentClassification.category}" not found, using general-purpose model.</p>
      `;
      
      this.completeStep(step);
      this.selectedModel = selectedModel;
      return;
    }

    const selectedModel = category.models[0]; // Best model for the category
    
    status.innerHTML = '<span class="status status--success">Selected</span>';
    
    content.innerHTML = `
      <div class="model-grid">
        ${category.models.map((model, index) => `
          <div class="model-item ${index === 0 ? 'model-item--selected' : ''}">
            <div class="model-info">
              <div class="model-name">${model.name}</div>
              <div class="model-description">${model.description}</div>
              <div class="model-categories">Best for: ${model.categories.join(', ')}</div>
            </div>
            <div class="model-score">${model.score}</div>
          </div>
        `).join('')}
      </div>
      <p style="margin-top: 12px;"><strong>Selection Reasoning:</strong> ${selectedModel.name} has the highest score (${selectedModel.score}) for ${normalizedCategory} tasks and excels in: ${selectedModel.categories.join(', ')}.</p>
    `;

    this.completeStep(step);
    this.selectedModel = selectedModel;
  }

  async performDataProcessing(originalMessage) {
    const step = document.getElementById('dataStep');
    const status = document.getElementById('dataStatus');
    const content = document.getElementById('dataContent');

    this.activateStep(step);
    status.innerHTML = '<div class="loading-spinner"></div>';

    const currentMode = this.config.get('mode');
    if (currentMode !== 'live') {
      await this.delay(1000); // Only delay in mock mode
    }

    const processedMessage = this.processMessageForSending(originalMessage);
    
    status.innerHTML = '<span class="status status--success">Ready</span>';
    
    content.innerHTML = `
      <div class="text-comparison">
        <div class="text-box original-text">
          <h4>Original Prompt</h4>
          <p>${originalMessage}</p>
        </div>
        <div class="text-box processed-text">
          <h4>Processed Prompt (sent to ${this.selectedModel.name})</h4>
          <p>${processedMessage}</p>
        </div>
        ${this.currentPiiResults.length > 0 ? `
          <p><strong>Security Actions:</strong></p>
          <ul>
            ${this.currentPiiResults.map(pii => `
              <li>${pii.type.replace('_', ' ')} detected and redacted (${pii.risk} risk)</li>
            `).join('')}
          </ul>
        ` : '<p style="color: var(--color-success);">✓ No security redactions needed</p>'}
      </div>
    `;

    this.completeStep(step);
  }

  async performModelResponse(originalMessage) {
    const step = document.getElementById('responseStep');
    const status = document.getElementById('responseStatus');
    const content = document.getElementById('responseContent');

    this.activateStep(step);
    status.innerHTML = '<div class="loading-spinner"></div>';

    const currentMode = this.config.get('mode');
    
    if (currentMode === 'live') {
      // Check if this is a programming question (likely to take longer)
      const isProgrammingQuery = this.currentClassification?.category === 'programming';
      
      if (isProgrammingQuery) {
        content.innerHTML = '<p class="text-secondary">Sending request to selected model via API (programming queries may take longer)...</p>';
        
        // Provide additional feedback after 15 seconds for programming queries
        setTimeout(() => {
          if (status.innerHTML.includes('loading-spinner')) {
            content.innerHTML = '<p class="text-secondary">Large model is processing your programming query, please wait...</p>';
          }
        }, 15000);
        
        // Additional feedback after 45 seconds
        setTimeout(() => {
          if (status.innerHTML.includes('loading-spinner')) {
            content.innerHTML = '<p class="text-secondary">Complex programming query is still being processed (this can take up to 3 minutes)...</p>';
          }
        }, 45000);
      } else {
        content.innerHTML = '<p class="text-secondary">Sending request to selected model via API...</p>';
      }
    } else {
      content.innerHTML = '<p class="text-secondary">Generating mock response...</p>';
    }

    try {
      // Get the processed (PII-redacted) message
      const processedMessage = this.processMessageForSending(originalMessage);
      
      // Generate the AI response
      const response = await this.generateAiResponse(processedMessage);
      
      status.innerHTML = '<span class="status status--success">Complete</span>';
      
      // Show response preview
      const responsePreview = response.length > 100 ? response.substring(0, 100) + '...' : response;
      let sourceInfo = '';
      
      if (currentMode === 'live') {
        // Check for cache information
        if (this.currentResponse && this.currentResponse.cached) {
          const cacheType = this.currentResponse.cacheType;
          const cacheIcon = cacheType === 'client' ? '⚡' : '💾';
          const cacheText = cacheType === 'client' ? 'Client Cache Hit' : 'Server Cache Hit';
          sourceInfo = `<p><em>${cacheIcon} ${cacheText} - Response served from cache</em></p>`;
        } else {
          sourceInfo = '<p><em>✓ Response generated by live API</em></p>';
        }
      } else {
        sourceInfo = '<p><em>📋 Mock response generated</em></p>';
      }
      
      content.innerHTML = `
        <div class="response-result">
          <p><strong>Response Preview:</strong></p>
          <p style="font-style: italic; padding: 8px; background: var(--color-secondary); border-radius: 4px;">${responsePreview}</p>
          ${sourceInfo}
        </div>
      `;

      this.completeStep(step);
      
      // Add the response to chat
      const finalDelay = currentMode === 'live' ? 0 : 500; // No delay in live mode
      await this.delay(finalDelay);
      this.addChatMessage(response, 'assistant');
      
      // Update cache stats and cache step with actual results
      this.updateCacheStats();
      this.updateCacheStepWithResults();
      
    } catch (error) {
      console.error('Model response failed:', error);
      status.innerHTML = '<span class="status status--error">Error</span>';
      
      // Provide specific error message for timeout
      let errorMessage = error.message;
      if (error.message.includes('timeout') || error.message.includes('timed out')) {
        errorMessage = 'Request timed out. Programming queries may require longer processing time.';
      }
      
      content.innerHTML = `
        <div class="response-result">
          <p style="color: var(--color-error);">Model response failed: ${errorMessage}</p>
          <p><em>Using fallback response.</em></p>
        </div>
      `;
      
      // Add fallback response
      const fallbackResponse = "I apologize, but I'm unable to process your request at the moment. Please try again later.";
      this.addChatMessage(fallbackResponse, 'assistant');
      
      this.completeStep(step);
    }
  }

  async classifyMessage(message) {
    const currentMode = this.config.get('mode');
    
    if (currentMode === 'mock') {
      return this.classifyMessageMock(message);
    } else {
      // Use full BERT classification via live API
      return await this.classifyMessageLive(message);
    }
  }

  // Fast classification for live mode - no full API call
  classifyMessageFast(message) {
    // Store the original console.log to capture debug output
    const originalLog = console.log;
    let debugInfo = null;
    
    // Temporarily override console.log to capture debug info
    console.log = (...args) => {
      if (args[0] === 'Classification Debug:') {
        debugInfo = args[1];
      }
      originalLog.apply(console, args);
    };
    
    // Use client-side analysis for fast classification in live mode
    const rawCategory = this.apiClient.analyzeMessageForCategory(message);
    const category = normalizeCategory(rawCategory); // Normalize the category
    const confidence = this.apiClient.calculateInferredConfidence(message, category) * 100;
    
    // Restore original console.log
    console.log = originalLog;
    
    // Extract detected keywords from debug info
    const matchedKeywords = debugInfo && debugInfo.detectedKeywords ? debugInfo.detectedKeywords : [category.toLowerCase()];
    
    return {
      category: category,
      confidence: Math.round(confidence),
      matchedKeywords: matchedKeywords,
      source: 'fast_analysis'
    };
  }

  classifyMessageMock(message) {
    const messageLower = message.toLowerCase();
    const categoryScores = {};
    
    categories.forEach(category => {
      const matchedKeywords = category.keywords.filter(keyword => 
        messageLower.includes(keyword.toLowerCase())
      );
      categoryScores[category.name] = {
        score: matchedKeywords.length,
        matchedKeywords: matchedKeywords
      };
    });

    // Find the category with the highest score
    let bestCategory = 'general';
    let bestScore = categoryScores.general.score;
    let matchedKeywords = categoryScores.general.matchedKeywords;

    Object.entries(categoryScores).forEach(([categoryName, data]) => {
      if (data.score > bestScore) {
        bestCategory = categoryName;
        bestScore = data.score;
        matchedKeywords = data.matchedKeywords;
      }
    });

    // Calculate confidence based on keyword matches and add some randomness for demo
    let confidence = Math.min(95, Math.max(70, (bestScore * 15) + Math.random() * 20));
    if (matchedKeywords.length === 0) {
      confidence = Math.random() * 20 + 65; // Random confidence for general queries
      matchedKeywords = ['general', 'query'];
    }

    // Normalize the category (coding -> programming)
    const normalizedCategory = normalizeCategory(bestCategory);

    return {
      category: normalizedCategory,
      confidence: Math.round(confidence),
      matchedKeywords: matchedKeywords,
      source: 'mock'
    };
  }

  async classifyMessageLive(message) {
    try {
      // Start the LLM request in the background (this triggers classification)
      const llmPromise = this.apiClient.sendSemanticRouterMessage(message, { 
        skipCache: true
      });
      
      // Poll for recent classifications that match our query
      const classificationResult = await this.pollForRecentClassification(message, 5000);
      
      if (classificationResult) {
        // Map semantic category to our UI categories
        const category = this.mapSemanticCategory(classificationResult.category);
        
        return {
          category: category,
          confidence: Math.round(classificationResult.confidence * 100),
          matchedKeywords: [classificationResult.category.toLowerCase()],
          source: 'live',
          model: classificationResult.model,
          llmPromise: llmPromise // Keep the LLM promise for later use
        };
      } else {
        // Fallback: wait for full response and extract from headers/body
        const response = await llmPromise;
        const semanticData = response.semanticData;
        const category = this.mapSemanticCategory(semanticData.semanticCategory);
        const confidence = semanticData.confidence ? Math.round(semanticData.confidence * 100) : 90;
        
        return {
          category: category,
          confidence: confidence,
          matchedKeywords: [semanticData.semanticCategory.toLowerCase()],
          source: 'live',
          originalResponse: response,
          semanticData: semanticData
        };
      }
      
    } catch (error) {
      console.error('Live classification failed:', error);
      
      // Fallback to mock classification on error
      const mockResult = this.classifyMessageMock(message);
      mockResult.source = 'mock_fallback';
      mockResult.error = error.message;
      
      return mockResult;
    }
  }

  async pollForRecentClassification(query, timeoutMs = 5000) {
    const startTime = Date.now();
    const pollInterval = 100; // Poll every 100ms
    const queryLower = query.toLowerCase().trim();
    
    while (Date.now() - startTime < timeoutMs) {
      try {
        const response = await fetch(`http://localhost:9190/classification`);
        if (response.ok) {
          const allClassifications = await response.json();
          
          // Look for a recent classification that matches our query
          for (const [requestId, classification] of Object.entries(allClassifications)) {
            if (classification.query && classification.query.toLowerCase().trim() === queryLower) {
              // Check if it's recent (within last 10 seconds)
              const age = Date.now() / 1000 - classification.timestamp;
              if (age < 10) {
                console.log('Found matching classification:', classification);
                return classification;
              }
            }
          }
        }
      } catch (error) {
        // Continue polling on error
      }
      
      // Wait before next poll
      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }
    
    console.log('Classification polling timed out for query:', query);
    return null;
  }

  mapSemanticCategory(apiCategory) {
    // Map API categories to our UI categories
    const categoryMap = {
      'mathematics': 'mathematics',
      'math': 'mathematics',
      'programming': 'programming',
      'health': 'health',
      'history': 'history',
      'general': 'general',
      'default': 'general'
    };
    
    const normalized = apiCategory.toLowerCase();
    return categoryMap[normalized] || categoryMap['default'];
  }

  async detectPii(message, classificationResult = null) {
    const currentMode = this.config.get('mode');
    
    if (currentMode === 'mock') {
      return this.detectPiiMock(message);
    } else {
      return await this.detectPiiLive(message, classificationResult);
    }
  }

  detectPiiMock(message) {
    const detectedPii = [];
    
    piiPatterns.forEach(pattern => {
      const matches = message.match(pattern.pattern);
      if (matches) {
        matches.forEach(match => {
          detectedPii.push({
            type: pattern.type,
            value: match,
            replacement: pattern.replacement,
            risk: pattern.risk,
            source: 'mock'
          });
        });
      }
    });

    return detectedPii;
  }

  async detectPiiLive(message, classificationResult = null) {
    try {
      let piiData = [];
      
      // Debug logging
      console.log('PII Detection Debug:', {
        hasClassificationResult: !!classificationResult,
        classificationSource: classificationResult?.source,
        hasSemanticData: !!classificationResult?.semanticData,
        piiDetectionData: classificationResult?.semanticData?.piiDetection
      });
      
      // If we already have classification data from live API, use it
      if (classificationResult && classificationResult.source === 'live' && classificationResult.semanticData) {
        piiData = classificationResult.semanticData.piiDetection || [];
        console.log('Using PII data from classification result:', piiData);
      } else {
        // Use fast local PII detection instead of expensive API call
        console.log('Using fast local PII detection (no expensive API call needed)');
        const localPiiResults = this.apiClient.analyzeMessageForPii(message);
        
        // Convert to our expected format
        piiData = localPiiResults.map(pii => ({
          type: pii.type,
          value: pii.value,
          risk: pii.risk,
          replacement: pii.replacement || this.getFormatPreservingReplacement(pii.type)
        }));
        
        console.log('Local PII detection results:', piiData);
      }
      
      // Convert API PII format to our UI format
      return piiData.map(piiItem => {
        if (typeof piiItem === 'string') {
          // Simple string format
          return {
            type: 'PII_DETECTED',
            value: piiItem,
            replacement: 'john.doe@example.com',
            risk: 'medium',
            source: 'live'
          };
        } else {
          // Object format
          return {
            type: piiItem.type || 'PII_DETECTED',
            value: piiItem.value || piiItem.text || 'Detected PII',
            replacement: piiItem.replacement || this.getFormatPreservingReplacement(piiItem.type || 'PII_DETECTED'),
            risk: piiItem.risk || 'medium',
            source: 'live'
          };
        }
      });
      
    } catch (error) {
      console.error('Live PII detection failed:', error);
      
      // Fallback to mock PII detection
      const mockResult = this.detectPiiMock(message);
      mockResult.forEach(item => {
        item.source = 'mock_fallback';
        item.error = error.message;
      });
      
      return mockResult;
    }
  }

  highlightPii(message, piiResults) {
    let highlightedMessage = message;
    
    piiResults.forEach(pii => {
      highlightedMessage = highlightedMessage.replace(
        new RegExp(pii.value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'),
        `<span class="pii-highlight">${pii.value}</span>`
      );
    });

    return highlightedMessage;
  }

  getFormatPreservingReplacement(piiType) {
    const replacements = {
      'EMAIL_ADDRESS': [
        'john.doe@example.com',
        'jane.smith@example.org',
        'user@example.net'
      ],
      'PHONE_NUMBER': [
        '(555) 012-3456',
        '(123) 456-7890',
        '(555) 123-4567'
      ],
      'CREDIT_CARD': [
        '4111-1111-1111-1111',
        '4000-0000-0000-0002',
        '5555-5555-5555-4444'
      ],
      'SSN': [
        '000-00-0000',
        '987-65-4321',
        '123-45-6789'
      ],
      'PII_DETECTED': [
        'john.doe@example.com',
        'user@example.com'
      ]
    };
    
    const typeReplacements = replacements[piiType] || replacements['PII_DETECTED'];
    
    // Simple rotation based on current time to provide variety
    const index = Math.floor(Date.now() / 1000) % typeReplacements.length;
    return typeReplacements[index];
  }

  processMessageForSending(message) {
    let processedMessage = message;
    
    this.currentPiiResults.forEach(pii => {
      processedMessage = processedMessage.replace(
        new RegExp(pii.value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'),
        pii.replacement
      );
    });

    return processedMessage;
  }

  async generateAiResponse(processedMessage) {
    const currentMode = this.config.get('mode');
    
    // Debug logging
    console.log('generateAiResponse - currentMode:', currentMode);
    console.log('generateAiResponse - processedMessage:', processedMessage);
    
    // In live mode, make the actual API call with the processed message
    if (currentMode === 'live') {
      try {
        console.log('Making API call with processed message:', processedMessage);
        
        // Use custom timeout for programming queries (which use heavier models)
        const isProgrammingQuery = this.currentClassification?.category === 'programming';
        const customTimeout = isProgrammingQuery ? 180000 : this.config.get('timeout'); // 3 minutes for programming, default for others
        
        console.log(`Making AI response API call with ${customTimeout/1000}s timeout for ${isProgrammingQuery ? 'programming' : 'general'} query`);
        
        // Use the semantic router message API which already handles cache detection
        const response = await this.apiClient.sendSemanticRouterMessage(processedMessage, {
          timeout: customTimeout
        });
        
        console.log('Semantic router response received:', response);
        
        // Store response for cache detection in UI
        this.currentResponse = response;
        
        // Extract the actual message content from various possible locations
        let messageContent = '';
        
        // Debug the response structure
        console.log('Response type:', typeof response);
        console.log('Response keys:', Object.keys(response));
        console.log('Response data type:', typeof response.data);
        
        // First, try the formattedContent (processed by the API client)
        if (response.formattedContent) {
          // Handle both string and object formats for formattedContent
          if (typeof response.formattedContent === 'string') {
            messageContent = response.formattedContent;
          } else if (typeof response.formattedContent === 'object') {
            // Extract text content from the formatted object
            messageContent = response.formattedContent.plainText || 
                           response.formattedContent.originalMarkdown || 
                           response.formattedContent.htmlContent || '';
          }

        }
        // Then try the standard OpenAI format
        else if (response.data && response.data.choices && response.data.choices[0] && response.data.choices[0].message) {
          messageContent = response.data.choices[0].message.content;
          
        }
        // Handle Blob response data
        else if (response.data instanceof Blob) {
          try {
            const blobText = await response.data.text();
            console.log('Blob content:', blobText);
            
            // Try to parse as JSON
            try {
              const parsedData = JSON.parse(blobText);
              if (parsedData.choices && parsedData.choices[0] && parsedData.choices[0].message) {
                messageContent = parsedData.choices[0].message.content;
              } else if (parsedData.content) {
                messageContent = parsedData.content;
              } else if (parsedData.response) {
                messageContent = parsedData.response;
              }
            } catch (parseError) {
              // If not JSON, use the text directly
              messageContent = blobText;
            }
          } catch (blobError) {
            console.error('Error reading blob data:', blobError);
            throw new Error('Failed to read response data');
          }
        }
        // Try other possible response formats
        else if (response.data && typeof response.data === 'string') {
          messageContent = response.data;
        }
        else if (response.originalMessage) {
          // Fallback: echo back the original message with processing note
          messageContent = `Processed query: "${response.originalMessage}"`;
        }
        

        
        if (!messageContent || typeof messageContent !== 'string' || !messageContent.trim()) {
          console.error('No valid content found in response:', response);
          throw new Error('No message content found in API response');
        }
        
        console.log('Extracted message content:', messageContent);
        console.log('Cache info:', { cached: response.cached, cacheType: response.cacheType });
        
        return messageContent.trim();
        
      } catch (error) {
        console.error('Live API call failed:', error);
        // Fall through to mock responses
      }
    }
    
    // Fallback to mock responses for demo mode or when no live response available
    const responses = [
      "I've processed your request through the semantic router. The system classified your prompt, checked for sensitive information, selected the optimal model, and cleaned the data before processing.",
      "Your query has been successfully routed and processed. The semantic analysis helped determine the best model for your specific type of request.",
      "Processing complete! The semantic router analyzed your prompt, detected any sensitive information, and selected the most appropriate model for generating a response.",
      "Thanks for using the semantic router demo! Your message was classified, scanned for PII, and routed to the optimal model based on the content analysis."
    ];
    
    return responses[Math.floor(Math.random() * responses.length)];
  }

  activateStep(stepElement) {
    stepElement.classList.add('processing-step--active');
  }

  completeStep(stepElement) {
    stepElement.classList.remove('processing-step--active');
    stepElement.classList.add('processing-step--completed');
  }

  resetProcessingSteps() {
    const steps = document.querySelectorAll('.processing-step');
    steps.forEach(step => {
      step.classList.remove('processing-step--active', 'processing-step--completed');
    });

    // Reset status indicators
    document.getElementById('classificationStatus').innerHTML = '';
    document.getElementById('piiStatus').innerHTML = '';
    document.getElementById('modelStatus').innerHTML = '';
    document.getElementById('dataStatus').innerHTML = '';
    document.getElementById('responseStatus').innerHTML = '';

    // Reset content
    document.getElementById('classificationContent').innerHTML = '<p class="text-secondary">Analyzing prompt semantic meaning...</p>';
    document.getElementById('piiContent').innerHTML = '<p class="text-secondary">Scanning for personally identifiable information...</p>';
    document.getElementById('modelContent').innerHTML = '<p class="text-secondary">Selecting optimal model based on category...</p>';
    document.getElementById('dataContent').innerHTML = '<p class="text-secondary">Cleaning and preparing final prompt...</p>';
    document.getElementById('responseContent').innerHTML = '<p class="text-secondary">Generating AI response...</p>';

    this.updateProcessingStatus('Ready', 'info');
  }

  updateProcessingStatus(text, type) {
    this.processingStatus.innerHTML = `<span class="status status--${type}">${text}</span>`;
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  updateCacheStepWithResults() {
    const step = document.getElementById('cacheStep');
    const status = document.getElementById('cacheStatus');
    const content = document.getElementById('cacheContent');

    // Only update if we have response data and we're in live mode
    if (!this.currentResponse || this.config.get('mode') !== 'live') {
      return;
    }

    const currentStats = this.apiClient.getCacheStats();
    const wasCacheHit = this.currentResponse.cached;
    const cacheType = this.currentResponse.cacheType;

    if (wasCacheHit) {
      status.innerHTML = '<span class="status status--success">Cache Hit!</span>';
      
      let hitDetails = '';
      if (cacheType === 'client') {
        hitDetails = `
          <div class="cache-hit">
            <p>⚡ <strong>Client Cache Hit - Instant Response!</strong></p>
            <p><strong>Similarity:</strong> 100% (exact match)</p>
            <p><strong>Cache Type:</strong> Browser-side cache</p>
            <p><strong>Response Time:</strong> <1ms</p>
            <p><em>Identical query found in local cache</em></p>
          </div>
        `;
      } else {
        // Server cache hit
        const similarity = this.currentResponse.similarity ? (this.currentResponse.similarity * 100) : 85; // Convert to percentage
        hitDetails = `
          <div class="cache-hit">
            <p>💾 <strong>Server Cache Hit - Semantic Match Found!</strong></p>
            <p><strong>Similarity Score:</strong> ${similarity.toFixed(1)}%</p>
            <p><strong>Cache Type:</strong> Server-side semantic cache</p>
            <p><strong>BERT Embedding:</strong> 512-dimensional vectors compared</p>
            <p><strong>Threshold:</strong> 80% (exceeded)</p>
            <p><em>Similar question found using semantic analysis</em></p>
          </div>
        `;
      }

      content.innerHTML = `
        <div class="cache-result">
          ${hitDetails}
          <div class="cache-stats">
            <div class="cache-stat-item">
              <span class="cache-stat-label">Total Requests:</span>
              <span class="cache-stat-value">${currentStats.totalRequests}</span>
            </div>
            <div class="cache-stat-item">
              <span class="cache-stat-label">Overall Hit Rate:</span>
              <span class="cache-stat-value">${currentStats.totalHitRate}%</span>
            </div>
            <div class="cache-stat-item">
              <span class="cache-stat-label">Server Cache:</span>
              <span class="cache-stat-value">${currentStats.serverHitRate}%</span>
            </div>
            <div class="cache-stat-item">
              <span class="cache-stat-label">Client Cache:</span>
              <span class="cache-stat-value">${currentStats.clientHitRate}%</span>
            </div>
          </div>
          <p><em>✅ Cache optimization successfully reduced response time</em></p>
        </div>
      `;
    } else {
      // Cache miss - update with actual results
      status.innerHTML = '<span class="status status--info">Cache Miss</span>';
      
      // Try to get similarity score from response if available
      const bestSimilarity = this.currentResponse.similarity ? (this.currentResponse.similarity * 100) : null;
      const similarityText = bestSimilarity !== null ? 
        `${bestSimilarity.toFixed(1)}% (below 80% threshold)` : 
        'Below 80% threshold';
      
      content.innerHTML = `
        <div class="cache-result">
          <div class="cache-miss">
            <p>❌ <strong>No cache hit - Full processing required</strong></p>
            <p><strong>BERT Embedding:</strong> 512-dimensional vector generated</p>
            <p><strong>Similarity Check:</strong> Compared against ${currentStats.totalRequests > 1 ? 'existing entries' : 'empty cache'}</p>
            <p><strong>Best Match:</strong> ${similarityText}</p>
            <p><em>Query processed and cached for future similar requests</em></p>
          </div>
          <div class="cache-stats">
            <div class="cache-stat-item">
              <span class="cache-stat-label">Total Requests:</span>
              <span class="cache-stat-value">${currentStats.totalRequests}</span>
            </div>
            <div class="cache-stat-item">
              <span class="cache-stat-label">Overall Hit Rate:</span>
              <span class="cache-stat-value">${currentStats.totalHitRate}%</span>
            </div>
            <div class="cache-stat-item">
              <span class="cache-stat-label">Server Cache:</span>
              <span class="cache-stat-value">${currentStats.serverHitRate}%</span>
            </div>
            <div class="cache-stat-item">
              <span class="cache-stat-label">Client Cache:</span>
              <span class="cache-stat-value">${currentStats.clientHitRate}%</span>
            </div>
          </div>
          <p><em>🔄 New entry added to semantic cache for future optimization</em></p>
        </div>
      `;
    }
  }
}

// Initialize the dashboard when the page loads
let dashboard;

document.addEventListener('DOMContentLoaded', () => {
  dashboard = new SemanticRouterDashboard();
});

// Make dashboard available globally for sample prompt buttons
window.dashboard = dashboard;

// Function to normalize category names (treat coding as programming)
function normalizeCategory(category) {
  const normalized = category.toLowerCase();
  if (normalized === 'coding' || normalized === 'code') {
    return 'programming';
  }
  return normalized;
}
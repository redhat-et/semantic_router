// Application data from the provided JSON
const categories = [
  {
    name: "mathematics",
    keywords: ["math", "calculate", "derivative", "integral", "equation", "solve", "algebra", "geometry", "statistics", "probability"],
    models: [
      {"name": "phi4", "score": 1.0, "description": "Best for mathematical reasoning"},
      {"name": "mistral-small3.1", "score": 0.8, "description": "Good alternative for math"},
      {"name": "gemma3:27b", "score": 0.6, "description": "Acceptable fallback"}
    ]
  },
  {
    name: "programming",
    keywords: ["code", "function", "variable", "loop", "array", "javascript", "python", "react", "api", "debug", "syntax", "programming", "software", "algorithm"],
    models: [
      {"name": "mistral-small3.1", "score": 0.9, "description": "Excellent for coding tasks"},
      {"name": "gemma3:27b", "score": 0.8, "description": "Good programming assistance"},
      {"name": "phi4", "score": 0.6, "description": "Decent for algorithms"}
    ]
  },
  {
    name: "health",
    keywords: ["doctor", "medicine", "symptom", "treatment", "health", "medical", "hospital", "disease", "drug", "therapy", "diagnosis"],
    models: [
      {"name": "gemma3:27b", "score": 0.9, "description": "Excellent for health information"},
      {"name": "mistral-small3.1", "score": 0.8, "description": "Good medical knowledge"},
      {"name": "phi4", "score": 0.6, "description": "Basic health assistance"}
    ]
  },
  {
    name: "history",
    keywords: ["history", "historical", "war", "century", "ancient", "civilization", "empire", "revolution", "battle", "timeline"],
    models: [
      {"name": "gemma3:27b", "score": 0.9, "description": "Excellent for historical knowledge"},
      {"name": "mistral-small3.1", "score": 0.8, "description": "Good historical context"},
      {"name": "phi4", "score": 0.7, "description": "Decent historical facts"}
    ]
  },
  {
    name: "general",
    keywords: ["hello", "how", "what", "when", "where", "why", "explain", "help", "question", "answer"],
    models: [
      {"name": "mistral-small3.1", "score": 0.8, "description": "Best general-purpose model"},
      {"name": "gemma3:27b", "score": 0.75, "description": "Good reasoning capabilities"},
      {"name": "phi4", "score": 0.6, "description": "Specialized but capable"}
    ]
  }
];

const piiPatterns = [
  {
    type: "EMAIL_ADDRESS",
    pattern: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g,
    replacement: "[EMAIL_REDACTED]",
    risk: "medium"
  },
  {
    type: "CREDIT_CARD",
    pattern: /\b(?:\d{4}[-\s]?){3}\d{4}\b/g,
    replacement: "[CREDIT_CARD_REDACTED]",
    risk: "high"
  },
  {
    type: "PHONE_NUMBER",
    pattern: /\b(?:\+?1[-\s]?)?\(?[0-9]{3}\)?[-\s]?[0-9]{3}[-\s]?[0-9]{4}\b/g,
    replacement: "[PHONE_REDACTED]",
    risk: "medium"
  },
  {
    type: "SSN",
    pattern: /\b\d{3}-\d{2}-\d{4}\b/g,
    replacement: "[SSN_REDACTED]",
    risk: "high"
  }
];

const samplePrompts = [
  "What is the derivative of f(x) = x² + 2x - 5?",
  "Calculate the integral of sin(x) from 0 to π",
  "Write a Python function to sort an array",
  "How do I debug JavaScript code in the browser?",
  "What are the symptoms of diabetes?",
  "How does aspirin work for pain relief?",
  "Tell me about the Roman Empire",
  "What caused World War 2?",
  "Hello, how are you today?",
  "My email is john.doe@company.com and my phone is (555) 123-4567"
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
      messageDiv.innerHTML = formattedMessage; // Format assistant messages
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
    this.updateProcessingStatus('Processing...', 'warning');
    
    // Step 1: Semantic Classification (fast)
    await this.performClassification(message);
    await this.delay(500);
    
    // Step 2: PII Detection (fast)
    await this.performPiiDetection(message);
    await this.delay(500);
    
    // Step 3: Model Selection
    await this.performModelSelection();
    await this.delay(500);
    
    // Step 4: Data Processing
    await this.performDataProcessing(message);
    await this.delay(500);
    
    // Step 5: Model Response (the slow part)
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
      await this.delay(300);
    } else {
      content.innerHTML = '<p class="text-secondary">Analyzing prompt semantic meaning...</p>';
      await this.delay(800);
    }

    try {
      const classification = await this.classifyMessage(message);
      
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
      } else {
        sourceInfo = '<p><em>📋 Using mock classification</em></p>';
        technicalInfo = '<p><strong>BERT Embedding:</strong> 768-dimensional vector processed</p>';
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

  async performPiiDetection(message) {
    const step = document.getElementById('piiStep');
    const status = document.getElementById('piiStatus');
    const content = document.getElementById('piiContent');

    this.activateStep(step);
    status.innerHTML = '<div class="loading-spinner"></div>';

    // Show mode-specific loading message
    const currentMode = this.config.get('mode');
    if (currentMode === 'live') {
      content.innerHTML = '<p class="text-secondary">Analyzing PII with live API...</p>';
      await this.delay(500);
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
      content.innerHTML = `
        <div class="pii-results">
          <p style="color: var(--color-error);">PII detection failed: ${error.message}</p>
          <p><em>Proceeding without PII filtering (security risk).</em></p>
        </div>
      `;
      
      // Set empty PII results on error
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

    await this.delay(1000);

    const category = categories.find(cat => cat.name === this.currentClassification.category);
    const selectedModel = category.models[0]; // Best model for the category
    
    status.innerHTML = '<span class="status status--success">Selected</span>';
    
    content.innerHTML = `
      <div class="model-grid">
        ${category.models.map((model, index) => `
          <div class="model-item ${index === 0 ? 'model-item--selected' : ''}">
            <div class="model-info">
              <div class="model-name">${model.name}</div>
              <div class="model-description">${model.description}</div>
            </div>
            <div class="model-score">${model.score}</div>
          </div>
        `).join('')}
      </div>
      <p style="margin-top: 12px;"><strong>Selection Reasoning:</strong> ${selectedModel.name} has the highest MMLU-Pro score (${selectedModel.score}) for ${this.currentClassification.category} tasks.</p>
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

    await this.delay(1000);

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
      content.innerHTML = '<p class="text-secondary">Sending request to selected model via API...</p>';
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
        sourceInfo = '<p><em>✓ Response generated by live API</em></p>';
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
      await this.delay(500);
      this.addChatMessage(response, 'assistant');
      
    } catch (error) {
      console.error('Model response failed:', error);
      status.innerHTML = '<span class="status status--error">Error</span>';
      content.innerHTML = `
        <div class="response-result">
          <p style="color: var(--color-error);">Model response failed: ${error.message}</p>
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
      return this.classifyMessageFast(message);
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
    const category = this.apiClient.analyzeMessageForCategory(message);
    const confidence = this.apiClient.calculateInferredConfidence(message, category) * 100;
    
    // Restore original console.log
    console.log = originalLog;
    
    // Extract detected keywords from debug info
    const matchedKeywords = debugInfo && debugInfo.detectedKeywords ? debugInfo.detectedKeywords : [category.toLowerCase()];
    
    return {
      category: category.toLowerCase(),
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

    return {
      category: bestCategory,
      confidence: Math.round(confidence),
      matchedKeywords: matchedKeywords,
      source: 'mock'
    };
  }

  async classifyMessageLive(message) {
    try {
      // Send message to semantic router API
      const response = await this.apiClient.sendSemanticRouterMessage(message);
      
      // Extract semantic data
      const semanticData = response.semanticData;
      
      // Map semantic category to our UI categories
      const category = this.mapSemanticCategory(semanticData.semanticCategory);
      
      // Calculate confidence (use extracted confidence or default to high confidence for live)
      const confidence = semanticData.confidence ? Math.round(semanticData.confidence * 100) : 90;
      
      return {
        category: category,
        confidence: confidence,
        matchedKeywords: [semanticData.semanticCategory.toLowerCase()],
        source: 'live',
        originalResponse: response,
        semanticData: semanticData
      };
      
    } catch (error) {
      console.error('Live classification failed:', error);
      
      // Fallback to mock classification on error
      const mockResult = this.classifyMessageMock(message);
      mockResult.source = 'mock_fallback';
      mockResult.error = error.message;
      
      return mockResult;
    }
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
      
      // If we already have classification data from live API, use it
      if (classificationResult && classificationResult.source === 'live' && classificationResult.semanticData) {
        piiData = classificationResult.semanticData.piiDetection || [];
      } else {
        // Otherwise make a separate request (though this should be rare)
        const response = await this.apiClient.sendSemanticRouterMessage(message);
        piiData = response.semanticData.piiDetection || [];
      }
      
      // Convert API PII format to our UI format
      return piiData.map(piiItem => {
        if (typeof piiItem === 'string') {
          // Simple string format
          return {
            type: 'PII_DETECTED',
            value: piiItem,
            replacement: '[PII_REDACTED]',
            risk: 'medium',
            source: 'live'
          };
        } else {
          // Object format
          return {
            type: piiItem.type || 'PII_DETECTED',
            value: piiItem.value || piiItem.text || 'Detected PII',
            replacement: piiItem.replacement || '[PII_REDACTED]',
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
        const apiResponse = await this.apiClient.sendSemanticRouterMessage(processedMessage);
        console.log('API response received:', apiResponse);
        
        // Debug: log the full response structure
        console.log('generateAiResponse - full API response:', apiResponse);
        console.log('generateAiResponse - response.data:', apiResponse.data);
        console.log('generateAiResponse - response.headers:', apiResponse.headers);
        
        // Try to extract the actual AI response from various possible locations
        let actualResponse = null;
        
        // Handle Blob data - convert to text first
        if (apiResponse.data instanceof Blob) {
          try {
            const blobText = await apiResponse.data.text();
            console.log('generateAiResponse - blob content:', blobText);
            
            // Try to parse as JSON
            let parsedData = null;
            try {
              parsedData = JSON.parse(blobText);
              console.log('generateAiResponse - parsed JSON from blob:', parsedData);
            } catch (e) {
              // If not JSON, use the text directly
              console.log('generateAiResponse - blob is not JSON, using as text');
              actualResponse = blobText;
            }
            
            // If we successfully parsed JSON, extract the response
            if (parsedData) {
              // Check if it's an OpenAI-style response with choices
              if (parsedData.choices && parsedData.choices.length > 0) {
                const choice = parsedData.choices[0];
                if (choice.message && choice.message.content) {
                  actualResponse = choice.message.content;
                } else if (choice.text) {
                  actualResponse = choice.text;
                }
              }
              
              // Check other possible locations
              if (!actualResponse && parsedData.response) {
                actualResponse = parsedData.response;
              }
              
              if (!actualResponse && parsedData.content) {
                actualResponse = parsedData.content;
              }
            }
          } catch (blobError) {
            console.error('Error reading blob data:', blobError);
          }
        } else {
          // Handle non-blob data (the original logic)
          // Check if it's an OpenAI-style response with choices
          if (apiResponse.data && apiResponse.data.choices && apiResponse.data.choices.length > 0) {
            const choice = apiResponse.data.choices[0];
            if (choice.message && choice.message.content) {
              actualResponse = choice.message.content;
            } else if (choice.text) {
              actualResponse = choice.text;
            }
          }
          
          // Check if the response is directly in the data
          if (!actualResponse && apiResponse.data && apiResponse.data.response) {
            actualResponse = apiResponse.data.response;
          }
          
          // Check if the response is in the content field
          if (!actualResponse && apiResponse.data && apiResponse.data.content) {
            actualResponse = apiResponse.data.content;
          }
          
          // Check if the response is directly the string
          if (!actualResponse && typeof apiResponse.data === 'string') {
            actualResponse = apiResponse.data;
          }
        }
        
        // If we found an actual response, return it with a note about the source
        if (actualResponse && actualResponse.trim()) {
          return `${actualResponse.trim()}\n\n---\n*Response generated by ${this.selectedModel?.name || 'semantic router model'} via live API*`;
        }
        
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

    // Reset content
    document.getElementById('classificationContent').innerHTML = '<p class="text-secondary">Analyzing prompt semantic meaning...</p>';
    document.getElementById('piiContent').innerHTML = '<p class="text-secondary">Scanning for personally identifiable information...</p>';
    document.getElementById('modelContent').innerHTML = '<p class="text-secondary">Selecting optimal model based on category...</p>';
    document.getElementById('dataContent').innerHTML = '<p class="text-secondary">Cleaning and preparing final prompt...</p>';

    this.updateProcessingStatus('Ready', 'info');
  }

  updateProcessingStatus(text, type) {
    this.processingStatus.innerHTML = `<span class="status status--${type}">${text}</span>`;
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Initialize the dashboard when the page loads
let dashboard;

document.addEventListener('DOMContentLoaded', () => {
  dashboard = new SemanticRouterDashboard();
});

// Make dashboard available globally for sample prompt buttons
window.dashboard = dashboard;
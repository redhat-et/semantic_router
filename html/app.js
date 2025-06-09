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
    name: "creative_writing",
    keywords: ["write", "story", "poem", "creative", "narrative", "character", "plot", "fiction", "novel", "script"],
    models: [
      {"name": "gemma3:27b", "score": 0.9, "description": "Excellent for creative tasks"},
      {"name": "claude-3", "score": 0.85, "description": "Strong creative capabilities"},
      {"name": "mistral-small3.1", "score": 0.7, "description": "Decent for creative writing"}
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
  "Write a short story about a robot discovering emotions",
  "My email is john.doe@company.com and my phone is (555) 123-4567. Can you help me?",
  "My credit card number is 1234-5678-9012-3456. Process my payment.",
  "Hello, how are you today?",
  "Calculate the area of a circle with radius 5",
  "Create a poem about the ocean"
];

class SemanticRouterDashboard {
  constructor() {
    this.chatContainer = document.getElementById('chatContainer');
    this.messageInput = document.getElementById('messageInput');
    this.chatForm = document.getElementById('chatForm');
    this.clearChatBtn = document.getElementById('clearChat');
    this.samplePromptsContainer = document.getElementById('samplePrompts');
    this.processingStatus = document.getElementById('processingStatus');
    
    this.isProcessing = false;
    this.chatHistory = [];
    
    this.init();
  }

  init() {
    this.setupEventListeners();
    this.renderSamplePrompts();
    this.resetProcessingSteps();
  }

  setupEventListeners() {
    this.chatForm.addEventListener('submit', (e) => this.handleSubmit(e));
    this.clearChatBtn.addEventListener('click', () => this.clearChat());
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

  addChatMessage(message, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message chat-message--${sender}`;
    messageDiv.textContent = message;
    
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
    
    // Step 1: Semantic Classification
    await this.performClassification(message);
    await this.delay(800);
    
    // Step 2: PII Detection
    await this.performPiiDetection(message);
    await this.delay(800);
    
    // Step 3: Model Selection
    await this.performModelSelection();
    await this.delay(800);
    
    // Step 4: Data Processing
    await this.performDataProcessing(message);
    await this.delay(500);
    
    this.updateProcessingStatus('Complete', 'success');
    
    // Add AI response
    const response = this.generateAiResponse();
    await this.delay(1000);
    this.addChatMessage(response, 'assistant');
  }

  async performClassification(message) {
    const step = document.getElementById('classificationStep');
    const status = document.getElementById('classificationStatus');
    const content = document.getElementById('classificationContent');

    this.activateStep(step);
    status.innerHTML = '<div class="loading-spinner"></div>';

    await this.delay(1500);

    const classification = this.classifyMessage(message);
    
    status.innerHTML = '<span class="status status--success">Complete</span>';
    
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
        <p><strong>BERT Embedding:</strong> 768-dimensional vector processed</p>
      </div>
    `;

    this.completeStep(step);
    this.currentClassification = classification;
  }

  async performPiiDetection(message) {
    const step = document.getElementById('piiStep');
    const status = document.getElementById('piiStatus');
    const content = document.getElementById('piiContent');

    this.activateStep(step);
    status.innerHTML = '<div class="loading-spinner"></div>';

    await this.delay(1200);

    const piiResults = this.detectPii(message);
    
    status.innerHTML = `<span class="status status--${piiResults.length > 0 ? 'warning' : 'success'}">
      ${piiResults.length > 0 ? 'PII Detected' : 'Clean'}
    </span>`;
    
    if (piiResults.length > 0) {
      const highlightedText = this.highlightPii(message, piiResults);
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
        </div>
      `;
    } else {
      content.innerHTML = `
        <div class="pii-results">
          <p style="color: var(--color-success);">✓ No personally identifiable information detected</p>
          <p>Text is safe to process without redaction.</p>
        </div>
      `;
    }

    this.completeStep(step);
    this.currentPiiResults = piiResults;
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

  classifyMessage(message) {
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
      matchedKeywords: matchedKeywords
    };
  }

  detectPii(message) {
    const detectedPii = [];
    
    piiPatterns.forEach(pattern => {
      const matches = message.match(pattern.pattern);
      if (matches) {
        matches.forEach(match => {
          detectedPii.push({
            type: pattern.type,
            value: match,
            replacement: pattern.replacement,
            risk: pattern.risk
          });
        });
      }
    });

    return detectedPii;
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

  generateAiResponse() {
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
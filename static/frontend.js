// State management class
class AppState {
    constructor() {
        this.isRunning = false;
        this.isProcessing = false;
        this.isLogVisible = false;
        this.totalTokens = 0;
        this.memories = [];
        this.systemStats = { cpu: 0, ram: 0, gpu: null };
        this.settings = this.loadSettings();
        this.loadMemories();
    }

    loadSettings() {
        return {
            model: localStorage.getItem('lastUsedModel') || 'Models/DarkIdol-Llama-3_1.gguf',
            prompts: {
                main: localStorage.getItem('mainSystemPrompt') || 'You are a helpful AI assistant with autonomous capabilities. You can think internally and respond externally.',
                internal: localStorage.getItem('internalPrompt') || 'This is your internal thought process. Use this to analyze and plan your responses.',
                external: localStorage.getItem('externalPrompt') || 'This is your external response to the user. Be clear, helpful, and engaging.'
            },
            tts: JSON.parse(localStorage.getItem('ttsSettings')) || {
                internalEnabled: true,
                externalEnabled: true,
                rate: 150,
                volume: 1.0,
                voice: null
            }
        };
    }

    // loadSettings() {
    //     return {
    //         model: localStorage.getItem('lastUsedModel') || 'darkidol.gguf',
    //         prompts: {
    //             main: localStorage.getItem('mainPrompt') || '',
    //             internal: localStorage.getItem('internalPrompt') || '',
    //             external: localStorage.getItem('externalPrompt') || ''
    //         },
    //         temperature: localStorage.getItem('temperature') || 0.7,
    //         maxTokens: localStorage.getItem('maxTokens') || 1000,
    //         topP: localStorage.getItem('topP') || 0.9
    //     };
    //  }loadSettings() {
        // return {
        //     model: localStorage.getItem('lastUsedModel') || 'F:/lm-studio/models/QuantFactory/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-GGUF/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q8_0.gguf',
        //     prompts: {
        //         main: localStorage.getItem('mainSystemPrompt') || 'You are a helpful AI assistant.',
        //         internal: localStorage.getItem('internalPrompt') || 'This is your internal thought process.',
        //         external: localStorage.getItem('externalPrompt') || 'This is your external response.'
        //     }
        // };

    saveSettings() {
        localStorage.setItem('lastUsedModel', this.settings.model);
        localStorage.setItem('mainPrompt', this.settings.prompts.main);
        localStorage.setItem('internalPrompt', this.settings.prompts.internal);
        localStorage.setItem('externalPrompt', this.settings.prompts.external);
        localStorage.setItem('temperature', this.settings.temperature);
        localStorage.setItem('maxTokens', this.settings.maxTokens);
        localStorage.setItem('topP', this.settings.topP);
    }

    loadMemories() {
        const savedMemories = localStorage.getItem('memories');
        this.memories = savedMemories ? JSON.parse(savedMemories) : [];
    }

    saveMemories() {
        localStorage.setItem('memories', JSON.stringify(this.memories));
    }

    addMemory(content, type = 'user') {
        const memory = {
            id: Date.now(),
            content,
            type,
            timestamp: new Date().toISOString()
        };
        this.memories.push(memory);
        this.saveMemories();
        return memory;
    }

    deleteMemory(id) {
        this.memories = this.memories.filter(m => m.id !== id);
        this.saveMemories();
    }

    clearMemories() {
        this.memories = [];
        this.saveMemories();
    }
}

// UI Controller class
class UIController {
    constructor(state, api) {
        this.state = state;
        this.api = api;
        this.elements = {};
        this.intervals = {};
    }

    initialize() {
        this.bindElements();
        this.loadSavedSettings();
        this.setupEventListeners();
        this.startPolling();
        this.setupModelSelection();
        this.updateStatus('offline');
        this.updateControls();
    }

    bindElements() {
        // Main controls
        this.elements = {
            startBtn: document.getElementById('startBtn'),
            stopBtn: document.getElementById('stopBtn'),
            submitBtn: document.getElementById('submitBtn'),
            promptInput: document.getElementById('promptInput'),
            cancelBtn: document.getElementById('cancelBtn'),
            statusIndicator: document.getElementById('statusIndicator'),
            messageContainer: document.getElementById('messageContainer'),
            spinner: document.getElementById('spinner'),
            modelSelect: document.getElementById('modelSelect'),
            loadedModel: document.getElementById('loadedModel'),
            // Progress elements
            loadingBar: document.getElementById('loadingBar'),
            loadingBarProgress: document.getElementById('loadingBarProgress'),
            processTime: document.getElementById('processTime'),
            // Stats elements
            cpuBar: document.getElementById('cpuBar'),
            ramBar: document.getElementById('ramBar'),
            gpuBar: document.getElementById('gpuBar'),
            cpuText: document.getElementById('cpuText'),
            ramText: document.getElementById('ramText'),
            gpuText: document.getElementById('gpuText'),
            // Settings elements
            mainPrompt: document.getElementById('mainSystemPrompt'),
            internalPrompt: document.getElementById('internalPrompt'),
            externalPrompt: document.getElementById('externalPrompt'),
            toggleLogBtn: document.getElementById('toggleLogBtn'),
            clearLogBtn: document.getElementById('clearLogBtn'),
            logContainer: document.getElementById('logContainer'),
            logContent: document.getElementById('logContent'),
            toggleSettingsBtn: document.getElementById('toggleSettingsBtn'),
            settingsPanel: document.getElementById('settingsPanel'),
            updatePromptsBtn: document.getElementById('updatePromptsBtn')
        };

        // Verify all elements exist
        Object.entries(this.elements).forEach(([key, element]) => {
            if (!element) {
                console.error(`Missing UI element: ${key}`);
            }
        });
    }

    loadSavedSettings() {
        // Load prompts
        this.elements.mainPrompt.value = this.state.settings.prompts.main;
        this.elements.internalPrompt.value = this.state.settings.prompts.internal;
        this.elements.externalPrompt.value = this.state.settings.prompts.external;
        
        // Load other settings if they exist
        if (this.elements.temperature) {
            this.elements.temperature.value = this.state.settings.temperature;
        }
        if (this.elements.maxTokens) {
            this.elements.maxTokens.value = this.state.settings.maxTokens;
        }
        if (this.elements.topP) {
            this.elements.topP.value = this.state.settings.topP;
        }
    }
    
    setupSettingsPanel() {
        if (!this.elements.updatePromptsBtn) return;
        
        this.elements.updatePromptsBtn.addEventListener('click', () => {
            // Save to localStorage
            localStorage.setItem('mainSystemPrompt', this.elements.mainPrompt.value);
            localStorage.setItem('internalPrompt', this.elements.internalPrompt.value);
            localStorage.setItem('externalPrompt', this.elements.externalPrompt.value);
            
            // Update state
            this.state.settings.prompts = {
                main: this.elements.mainPrompt.value,
                internal: this.elements.internalPrompt.value,
                external: this.elements.externalPrompt.value
            };
        });
    }

    updateChat(message, type = 'user') {
        const chatLog = document.getElementById('chatLog');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message-card ${type}-message`;
        
        if (type === 'user') {
            messageDiv.innerHTML = `
                <div class="message-content user">
                    <p>${message}</p>
                </div>
            `;
        } else if (type === 'internal') {
            messageDiv.innerHTML = `
                <div class="message-content internal">
                    <p class="internal-thought">${message}</p>
                </div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="message-content ai">
                    <p>${message}</p>
                </div>
            `;
        }
        
        chatLog.appendChild(messageDiv);
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    setupEventListeners() {
        // Start system
        this.elements.startBtn.addEventListener('click', async () => {
            if (!this.state.settings.model) {
                this.showError('Please select a model first');
                return;
            }
            
            try {
                this.updateStatus('loading');
                this.simulateProgress(10000);
                
                const response = await this.api.startSystem(this.state.settings.model);
                if (response.status === 'started') {
                    this.state.isRunning = true;
                    this.updateStatus('loaded');
                    this.elements.loadedModel.textContent = this.state.settings.model;
                    this.elements.loadedModel.style.display = 'inline-flex';
                }
                
            } catch (error) {
                this.showError(`Failed to start system: ${error.message}`);
                this.updateStatus('offline');
            } finally {
                this.completeProgress();
                this.updateControls();
            }
        });

        // Stop system
        this.elements.stopBtn.addEventListener('click', async () => {
            try {
                const response = await this.api.stopSystem();
                if (response.status === 'stopped') {
                    this.state.isRunning = false;
                    this.updateStatus('offline');
                }
            } catch (error) {
                this.showError(`Failed to stop system: ${error.message}`);
            } finally {
                this.updateControls();
            }
        });

        // Submit prompt
        this.elements.submitBtn.addEventListener('click', async () => {
            if (this.state.isProcessing) return;
            
            try {
                this.state.isProcessing = true;
                this.updateStatus('thinking');
                this.updateControls();
                
                const prompt = this.elements.promptInput.value;
                const response = await this.api.submitPrompt(prompt);
                
                if (response.error) throw new Error(response.error);
                
                // Clear input after successful submission
                this.elements.promptInput.value = '';
                
            } catch (error) {
                this.showError(`Error processing request: ${error.message}`);
            } finally {
                this.state.isProcessing = false;
                this.updateStatus('loaded');
                this.updateControls();
            }
        });

        // Enter key handler for prompt input
        this.elements.promptInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter' && !event.shiftKey && this.elements.promptInput.value.trim()) {
                event.preventDefault();
                if (this.state.isRunning && !this.state.isProcessing) {
                    this.elements.submitBtn.click();
                }
            }
        });

        // Model selection handler
        this.elements.modelSelect.addEventListener('change', (e) => {
            this.state.settings.model = e.target.value;
            this.state.saveSettings();
        });

        this.setupLogHandlers();
        this.setupSettingsPanel();
    }

    setupLogHandlers() {
        this.elements.toggleLogBtn.addEventListener('click', () => {
            this.elements.logContainer.classList.toggle('show');
        });
    
        this.elements.clearLogBtn.addEventListener('click', async () => {
            try {
                await this.api.clearLogs();
                this.elements.logContent.innerHTML = '';
            } catch (error) {
                console.error('Error clearing logs:', error);
            }
        });
    
        // Add polling for logs
        setInterval(async () => {
            try {
                const response = await fetch('/get_logs');
                const data = await response.json();
                if (data.logs) {
                    this.updateLogs(data.logs);
                }
            } catch (error) {
                console.error('Error fetching logs:', error);
            }
        }, 1000);
    }
    
    updateLogs(logs) {
        const logContent = this.elements.logContent;
        if (!Array.isArray(logs)) return;
        
        logs.forEach(log => {
            // Skip if log entry is malformed
            if (!log || !log.message) return;
            
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry ${log.level || 'info'}`;
            const timestamp = log.timestamp || new Date().toISOString();
            logEntry.textContent = `[${timestamp}] ${log.message}`;
            logContent.appendChild(logEntry);
        });
        logContent.scrollTop = logContent.scrollHeight;
    }
    // setupLogHandlers() {
    //     const toggleLogBtn = this.elements.toggleLogBtn;
    //     const clearLogBtn = this.elements.clearLogBtn;
    //     const logContainer = this.elements.logContainer;

    //     toggleLogBtn.addEventListener('click', () => {
    //         logContainer.classList.toggle('show');
    //     });

    //     clearLogBtn.addEventListener('click', async () => {
    //         try {
    //             const response = await this.api.clearLogs();
    //             if (response.status === 'success') {
    //                 this.elements.logContent.innerHTML = '';
    //             }
    //         } catch (error) {
    //             console.error('Error clearing logs:', error);
    //         }
    //     });
    // }

    // updateLogs(logs) {
    //     const logContent = this.elements.logContent;
    //     logs.forEach(log => {
    //         const logEntry = document.createElement('div');
    //         logEntry.className = `log-entry log-${log.level}`;
    //         logEntry.textContent = log.message;
    //         logContent.appendChild(logEntry);
    //     });
    //     logContent.scrollTop = logContent.scrollHeight;
    // }

    startPolling() {
        // Poll messages
        this.intervals.messages = setInterval(() => this.pollMessages(), 1000);
        
        // Poll system stats
        this.intervals.stats = setInterval(() => this.updateSystemStats(), 1000);
    }

    async pollMessages() {
        if (!this.state.isRunning) return;

        try {
            const messages = await this.api.getMessages();
            if (!messages || !messages.length) return;

            let completedResponse = false;
            messages.forEach(msg => this.appendMessage(msg));

            // Check for completion
            if (messages.some(msg => msg.type === 'external')) {
                this.state.isProcessing = false;
                this.updateControls();
            }

            // Scroll to bottom
            this.elements.messageContainer.scrollTop = 
                this.elements.messageContainer.scrollHeight;

        } catch (error) {
            console.error('Error polling messages:', error);
            this.state.isProcessing = false;
            this.updateControls();
        }
    }

    appendMessage(msg) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${msg.type}`;
        messageDiv.innerHTML = `
            <span class="timestamp">[${msg.timestamp}]</span>
            ${this.getMessageIcon(msg.type)}
            ${msg.content}
        `;
        this.elements.messageContainer.appendChild(messageDiv);

        // Handle memory-related content
        if (this.shouldAddToMemory(msg.content)) {
            this.addMemory(msg.content, 'ai');
        }
    }

    getMessageIcon(type) {
        const icons = {
            thinking: '🤔',
            internal: '💭',
            external: '🗣️'
        };
        return icons[type] || '';
    }

    shouldAddToMemory(content) {
        const keywords = ['remember', 'memory', 'note', 'important'];
        return keywords.some(keyword => 
            content.toLowerCase().includes(keyword));
    }

    updateSystemStats() {
        this.api.getSystemStats()
            .then(stats => {
                this.updateStatBar('cpu', stats.cpu);
                this.updateStatBar('ram', stats.ram);
                if (stats.gpu) {
                    this.updateStatBar('gpu', stats.gpu.load);
                    this.elements.gpuStats.style.display = 'flex';
                }
            })
            .catch(error => console.error('Error updating stats:', error));
    }

    updateStatBar(type, value) {
        const bar = this.elements[`${type}Bar`];
        const text = this.elements[`${type}Text`];
        if (!bar || !text) return;

        bar.style.width = `${value}%`;
        text.textContent = `${Math.round(value)}%`;
        this.updateBarColor(bar, value);
    }

    updateBarColor(bar, value) {
        bar.classList.remove('low', 'medium', 'high');
        if (value < 60) bar.classList.add('low');
        else if (value < 85) bar.classList.add('medium');
        else bar.classList.add('high');
    }

    showError(message) {
        // Implement your preferred error display mechanism
        console.error(message);
        alert(message); // Replace with better UI feedback
    }

    updateStatus(status) {
        if (!this.elements.statusIndicator) return;

        const statusConfig = {
            loaded: { class: 'loaded', text: 'Loaded' },
            offline: { class: 'offline', text: 'Offline' },
            loading: { class: 'loading', text: 'Loading' },
            thinking: { class: 'thinking', text: 'Thinking' }
        };

        const config = statusConfig[status];
        if (!config) return;

        this.elements.statusIndicator.className = 'status-badge ' + config.class;
        this.elements.statusIndicator.textContent = config.text;
    }

    updateControls() {
        if (!this.elements.startBtn) return;

        const controls = {
            startBtn: { disabled: this.state.isRunning },
            stopBtn: { disabled: !this.state.isRunning },
            promptInput: { 
                disabled: !this.state.isRunning || this.state.isProcessing 
            },
            submitBtn: { 
                disabled: !this.state.isRunning || this.state.isProcessing 
            },
            spinner: { 
                style: { display: this.state.isProcessing ? 'block' : 'none' }
            },
            cancelBtn: { 
                style: { display: this.state.isProcessing ? 'inline-block' : 'none' }
            }
        };

        Object.entries(controls).forEach(([elementName, properties]) => {
            const element = this.elements[elementName];
            if (!element) return;

            Object.entries(properties).forEach(([prop, value]) => {
                if (prop === 'style') {
                    Object.assign(element.style, value);
                } else {
                    element[prop] = value;
                }
            });
        });
    }

    simulateProgress(duration) {
        if (!this.elements.loadingBar || !this.elements.loadingBarProgress) return;

        this.elements.loadingBar.style.display = 'block';
        this.elements.loadingBarProgress.style.width = '0%';

        const startTime = Date.now();
        const update = () => {
            const elapsed = Date.now() - startTime;
            const percentage = Math.min((elapsed / duration) * 100, 95);
            this.elements.loadingBarProgress.style.width = percentage + '%';

            if (percentage < 95) {
                requestAnimationFrame(update);
            }
        };

        requestAnimationFrame(update);
    }

    completeProgress() {
        if (!this.elements.loadingBar || !this.elements.loadingBarProgress) return;

        this.elements.loadingBarProgress.style.width = '100%';
        setTimeout(() => {
            this.elements.loadingBar.style.display = 'none';
            this.elements.loadingBarProgress.style.width = '0%';
        }, 300);
    }

    cleanup() {
        // Clear all intervals
        Object.values(this.intervals).forEach(interval => clearInterval(interval));
    }

    // setupModelSelection() {
    //     const modelFolderBtn = document.getElementById('modelFolderBtn');
    //     const modelSelect = document.getElementById('modelSelect');

    //     if (modelFolderBtn && modelSelect) {
    //         modelFolderBtn.addEventListener('click', async () => {
    //             try {
    //                 const response = await fetch('/select_model_folder');
    //                 const data = await response.json();
                    
    //                 if (data.models) {
    //                     modelSelect.innerHTML = data.models
    //                         .map(model => `<option value="${model}">${model}</option>`)
    //                         .join('');
    //                 }
    //             } catch (error) {
    //                 console.error('Error loading models:', error);
    //             }
    //         });

    //         modelSelect.addEventListener('change', (e) => {
    //             localStorage.setItem('lastUsedModel', e.target.value);
    //         });
    //     }
    // }

    setupModelSelection() {
        const modelFolderBtn = document.getElementById('modelFolderBtn');
        const modelSelect = document.getElementById('modelSelect');
    
        modelFolderBtn.addEventListener('click', async () => {
            try {
                const response = await fetch('/select_model_folder');
                const data = await response.json();
                console.log('Models found:', data.models); // Debug log
                
                if (data.models && data.models.length > 0) {
                    modelSelect.innerHTML = '';
                    data.models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = `models/${model}`;
                        option.textContent = model;
                        modelSelect.appendChild(option);
                    });
                }
            } catch (error) {
                console.error('Error loading models:', error);
            }
        });
    
        // Trigger model folder load on startup
        modelFolderBtn.click();
    }

    setupSettingsPanel() {
        const toggleSettingsBtn = this.elements.toggleSettingsBtn;
        const settingsPanel = this.elements.settingsPanel;
        const updatePromptsBtn = this.elements.updatePromptsBtn;

        toggleSettingsBtn.addEventListener('click', () => {
            settingsPanel.classList.toggle('collapsed');
        });

        updatePromptsBtn.addEventListener('click', async () => {
            const prompts = {
                main: this.elements.mainPrompt.value,
                internal: this.elements.internalPrompt.value,
                external: this.elements.externalPrompt.value
            };

            try {
                await this.api.updatePrompts(prompts);
                this.state.settings.prompts = prompts;
                this.state.saveSettings();
            } catch (error) {
                console.error('Error updating prompts:', error);
            }
        });
    }
}

// API class for backend communication
class API {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
    }

    async request(endpoint, options = {}) {
        try {
            const response = await fetch(this.baseUrl + endpoint, {
                ...options,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                }
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'API request failed');
            return data;

        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            throw error;
        }
    }

    setupModelSelection() {
        const modelFolderBtn = document.getElementById('modelFolderBtn');
        const modelSelect = document.getElementById('modelSelect');
    
        modelFolderBtn.addEventListener('click', async () => {
            try {
                const response = await fetch('/select_model_folder');
                const data = await response.json();
                
                if (data.models && data.models.length > 0) {
                    modelSelect.innerHTML = '';
                    data.models.forEach(model => {
                        const fullPath = `models/${model}`;
                        const option = document.createElement('option');
                        option.value = fullPath;
                        option.textContent = model;
                        modelSelect.appendChild(option);
                    });
                }
            } catch (error) {
                console.error('Error loading models:', error);
            }
        });
    }

    // setupModelSelection() {
    //     const modelFolderBtn = document.getElementById('modelFolderBtn');
    //     const modelSelect = document.getElementById('modelSelect');
    
    //     modelFolderBtn.addEventListener('click', async () => {
    //         try {
    //             const response = await fetch('/select_model_folder');
    //             const data = await response.json();
    //             console.log('Models found:', data.models); // Debug log
                
    //             if (data.models && data.models.length > 0) {
    //                 modelSelect.innerHTML = '';
    //                 data.models.forEach(model => {
    //                     const option = document.createElement('option');
    //                     option.value = `F:/lm-studio/models/QuantFactory/${model}`;
    //                     option.textContent = model;
    //                     modelSelect.appendChild(option);
    //                 });
    //             }
    //         } catch (error) {
    //             console.error('Error loading models:', error);
    //         }
    //     });
    
    //     modelSelect.addEventListener('change', (e) => {
    //         console.log('Selected model:', e.target.value); // Debug log
    //         this.state.settings.model = e.target.value;
    //         localStorage.setItem('lastUsedModel', e.target.value);
    //     });
    // }
    
    async startSystem(model) {
        console.log('Starting system with model:', model); // Debug log
        const response = await fetch('/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                model: model 
            })
        }).then(res => res.json());
    }

    setupEventListeners() {
        this.elements.startBtn.addEventListener('click', async () => {
            try {
                const selectedModel = this.elements.modelSelect.value;
                if (!selectedModel) {
                    throw new Error('No model selected');
                }
    
                // Show loading UI
                this.elements.spinner.style.display = 'block';
                this.elements.loadingBar.style.display = 'block';
                this.elements.loadingBarProgress.style.width = '0%';
    
                const response = await this.api.startSystem(selectedModel);
                
                // Update progress bar
                this.elements.loadingBarProgress.style.width = '100%';
                
                if (response.status === 'started') {
                    this.state.isRunning = true;
                    this.updateStatus('loaded');
                }
            } catch (error) {
                console.error('Error starting system:', error);
                this.updateStatus('error');
            } finally {
                // Hide loading UI
                this.elements.spinner.style.display = 'none';
                this.elements.loadingBar.style.display = 'none';
            }
        });
    }
    
    startSystem(model) {
        return fetch('/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                model_name: model,
                selected_model: model  // Add this line
            })
        }).then(res => res.json());
    }

    stopSystem() {
        return this.request('/stop', { method: 'POST' });
    }

    // submitPrompt(prompt) {
    //     return this.request('/submit', {
    //         method: 'POST',
    //         body: JSON.stringify({ prompt })
    //     });
    // }

    async submitPrompt(prompt) {
        try {
            // Show user message
            this.updateChat(prompt, 'user');
            
            const response = await this.api.submitPrompt(prompt);
            if (response.internal) {
                this.updateChat(response.internal, 'internal');
            }
            if (response.response) {
                this.updateChat(response.response, 'ai');
            }
        } catch (error) {
            console.error('Error:', error);
        }
    }

    getMessages() {
        return this.request('/get_messages');
    }

    getSystemStats() {
        return this.request('/system_stats');
    }

    updateSettings(settings) {
        return this.request('/update_settings', {
            method: 'POST',
            body: JSON.stringify(settings)
        });
    }

    updatePrompts(prompts) {
        return this.request('/update_prompts', {
            method: 'POST',
            body: JSON.stringify(prompts)
        });
    }

    clearLogs() {
        return this.request('/clear_logs', { method: 'POST' });
    }

    getLogs() {
        return this.request('/get_logs');
    }

    async getModelList() {
        const response = await this.request('/select_model_folder');
        return response.models.map(model => ({
            name: typeof model === 'string' ? model : model.name,
            path: typeof model === 'string' ? model : model.path
        }));
    }
}

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    const state = new AppState();
    const api = new API();
    const ui = new UIController(state, api);
    ui.initialize();

    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        ui.cleanup()});
    }
);
// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const http = require('http');
const https = require('https');

const app = express();
const PORT = process.env.PORT || 3000;

// Configuração de agentes HTTP/HTTPS para manter conexões vivas (Keep-Alive)
// Isso previne erros de socket e 502 Bad Gateway por exaustão de portas
const httpAgent = new http.Agent({ keepAlive: true, maxSockets: 100 });
const httpsAgent = new https.Agent({ keepAlive: true, maxSockets: 100 });

// Middleware
app.use(cors());

app.use(express.json({
  limit: process.env.JSON_LIMIT || '50mb'
}));

app.use(express.urlencoded({
  limit: process.env.JSON_LIMIT || '50mb',
  extended: true
}));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'z-ai/glm-5.2',
  'gpt-4': 'qwen/qwen3.5-397b-a17b',
  'gpt-4-turbo': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro',
  'claude-3-opus': 'minimaxai/minimax-m3',
  'claude-3-sonnet': 'moonshotai/kimi-k2.6',
  'gemini-pro': 'stepfun-ai/step-3.7-flash'
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  // Cria um AbortController para cancelar a requisição se o cliente desconectar
  const abortController = new AbortController();
  
  // Se o cliente fechar a conexão antecipadamente, abortamos o Axios
  req.on('close', () => {
    abortController.abort();
  });

  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    
    if (!nimModel) {
      const modelLower = (model || '').toLowerCase();
      if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
        nimModel = 'meta/llama-3.1-405b-instruct';
      } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
        nimModel = 'meta/llama-3.1-70b-instruct';
      } else if (modelLower.includes('/') || modelLower.includes('-')) {
        nimModel = model;
      } else {
        nimModel = 'meta/llama-3.1-8b-instruct';
      }
    }
    
    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: stream || false
    };
    
    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 120000,
      httpAgent,
      httpsAgent,
      signal: abortController.signal,
      validateStatus: function (status) {
        return status >= 200 && status < 300;
      }
    });
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        
        // Mantém a última linha no buffer caso ela esteja incompleta
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          const trimmedLine = line.trim();
          if (trimmedLine.startsWith('data: ')) {
            if (trimmedLine.includes('[DONE]')) {
              res.write(trimmedLine + '\n\n');
              return;
            }
            
            try {
              const dataStr = trimmedLine.slice(6).trim();
              if (!dataStr) return; // Proteção contra linhas vazias
              
              const data = JSON.parse(dataStr);
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              // Em caso de falha no parse do JSON, ignoramos a linha silenciosamente
              // ou repassamos de forma bruta para evitar travar o fluxo
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      
      response.data.on('error', (err) => {
        console.error('Erro no stream de dados da NVIDIA:', err.message);
        if (!res.headersSent) {
          res.status(502).json({ error: { message: 'Stream interrompido', type: 'bad_gateway' } });
        } else {
          res.end();
        }
      });
      
    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    // Se o erro foi causado pelo cliente fechando a conexão (AbortError), apenas ignoramos
    if (axios.isCancel(error)) {
      console.log('Requisição cancelada pelo cliente.');
      return;
    }

    if (error.response) {
      console.error('NVIDIA NIM Rejeitou com:', error.response.status, JSON.stringify(error.response.data));
    } else {
      console.error('Erro de Proxy:', error.message);
    }
    
    if (!res.headersSent) {
      res.status(error.response?.status || 502).json({
        error: {
          message: error.response?.data?.detail || error.message || 'Erro de conexão ou falha interna do servidor',
          type: 'invalid_request_error',
          code: error.response?.status || 502
        }
      });
    }
  }
});

// Catch-all para rotas não encontradas
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// Proteções globais contra falhas que poderiam parar o servidor
process.on('uncaughtException', (err) => {
  console.error('Exceção não tratada capturada globalmente:', err);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Promessa rejeitada não tratada:', reason);
});

// Inicialização do Servidor
const server = app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});

// Ajustes essenciais de timeout para evitar erros 502 em balanceadores de carga
// O timeout do Node deve ser maior que o timeout do proxy reverso (ex: Nginx/AWS)
server.keepAliveTimeout = 65000; // 65 segundos
server.headersTimeout = 66000;   // Deve ser um pouco maior que keepAliveTimeout

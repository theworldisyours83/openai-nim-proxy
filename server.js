// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const http = require('http');
const https = require('https');

const app = express();
const PORT = process.env.PORT || 3000;

// Configuração de agentes HTTP/HTTPS para reutilização de conexões TCP.
const httpAgent = new http.Agent({ keepAlive: true, maxSockets: 100 });
const httpsAgent = new https.Agent({ keepAlive: true, maxSockets: 100 });

app.use(cors());
app.use(express.json({ limit: process.env.JSON_LIMIT || '50mb' }));
app.use(express.urlencoded({ limit: process.env.JSON_LIMIT || '50mb', extended: true }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'z-ai/glm-5.2',
  'gpt-4': 'qwen/qwen3.5-397b-a17b',
  'gpt-4-turbo': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro',
  'claude-3-opus': 'minimaxai/minimax-m3',
  'claude-3-sonnet': 'moonshotai/kimi-k2.6',
  'gemini-pro': 'stepfun-ai/step-3.7-flash'
};

app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'NVIDIA NIM Proxy (Stable)', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  const abortController = new AbortController();
  let streamClosedByClient = false;
  let heartbeatInterval = null;
  
  req.on('close', () => {
    streamClosedByClient = true;
    abortController.abort();
    if (heartbeatInterval) clearInterval(heartbeatInterval);
  });

  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
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
    
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: stream || false
    };

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache, no-transform');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');
      res.flushHeaders(); 

      heartbeatInterval = setInterval(() => {
        if (!streamClosedByClient) {
          res.write(': keepalive ping\n\n'); 
          if (res.flush) res.flush();
        }
      }, 15000);
    }
    
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 180000, 
      httpAgent,
      httpsAgent,
      signal: abortController.signal,
      validateStatus: status => status >= 200 && status < 300
    });
    
    if (heartbeatInterval) {
      clearInterval(heartbeatInterval);
      heartbeatInterval = null;
    }
    
    if (stream) {
      let buffer = '';
      let reasoningStarted = false;
      let isDoneSent = false;
      
      response.data.on('data', (chunk) => {
        if (streamClosedByClient) return;

        buffer += chunk.toString('utf8').replace(/\r\n/g, '\n');
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';
        
        for (const event of events) {
          const trimmedEvent = event.trim();
          if (!trimmedEvent || trimmedEvent.startsWith(':')) continue;

          if (trimmedEvent.startsWith('data:')) {
            const dataStr = trimmedEvent.slice(5).trim();
            
            if (dataStr === '[DONE]') {
              if (!isDoneSent) {
                res.write('data: [DONE]\n\n');
                isDoneSent = true;
              }
              continue;
            }
            
            try {
              const data = JSON.parse(dataStr);
              if (data.choices && data.choices[0] && data.choices[0].delta) {
                const delta = data.choices[0].delta;
                const reasoning = delta.reasoning_content;
                const content = delta.content;
                
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
                  
                  if (combinedContent) delta.content = combinedContent;
                  delete delta.reasoning_content;
                } else {
                  if (content) delta.content = content;
                  else delta.content = '';
                  delete delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
              if (res.flush) res.flush();
            } catch (e) {
              // Falha silenciosa para pacotes incompletos
            }
          }
        }
      });
      
      response.data.on('end', () => {
        if (!isDoneSent && !streamClosedByClient) {
          res.write('data: [DONE]\n\n');
          isDoneSent = true;
        }
        res.end();
      });
      
      response.data.on('error', (err) => {
        console.error('Erro no stream da NVIDIA:', err.message);
        if (!isDoneSent && !streamClosedByClient) {
          res.write('data: [DONE]\n\n');
          isDoneSent = true;
        }
        res.end();
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
            message: { role: choice.message.role, content: fullContent },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      res.json(openaiResponse);
    }
    
  } catch (error) {
    if (heartbeatInterval) clearInterval(heartbeatInterval);

    if (axios.isCancel(error)) return; 

    // Análise detalhada do erro retornado
    let errorDetail = 'Erro de conexão upstream.';
    if (error.response) {
      console.error('API da NVIDIA retornou erro:', error.response.status, JSON.stringify(error.response.data));
      errorDetail = error.response.data?.detail || error.response.data?.message || `Erro HTTP ${error.response.status} da NVIDIA.`;
    } else {
      console.error('Falha de conexão no Proxy:', error.message);
      errorDetail = error.message;
    }
    
    if (!res.headersSent) {
      // Se os cabeçalhos ainda não foram enviados, retornamos o erro HTTP padrão
      res.status(error.response?.status || 502).json({
        error: {
          message: errorDetail,
          type: 'proxy_error',
          code: error.response?.status || 502
        }
      });
    } else if (!streamClosedByClient) {
      // INJEÇÃO DE ERRO NO STREAM:
      // Transforma o erro real em uma mensagem de chat para que o cliente exiba na tela
      const errorPayload = {
        id: `chatcmpl-error-${Date.now()}`,
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model: 'proxy-error-reporter',
        choices: [{
          index: 0,
          delta: { content: `\n\n**[Erro no Servidor Proxy]** A comunicação com a NVIDIA falhou: ${errorDetail}\n\n` },
          finish_reason: 'stop'
        }]
      };
      
      res.write(`data: ${JSON.stringify(errorPayload)}\n\n`);
      res.write('data: [DONE]\n\n');
      res.end();
    }
  }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, type: 'not_found', code: 404 } });
});

process.on('uncaughtException', (err) => console.error('Falha Crítica (uncaughtException):', err));
process.on('unhandledRejection', (reason) => console.error('Falha Crítica (unhandledRejection):', reason));

const server = app.listen(PORT, () => {
  console.log(`Proxy rodando perfeitamente na porta ${PORT}`);
});

server.keepAliveTimeout = 65000;
server.headersTimeout = 66000;

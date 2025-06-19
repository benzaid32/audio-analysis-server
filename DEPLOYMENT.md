# Professional Audio Analysis Server Deployment

## 🚀 Production Deployment Options

### Option 1: Docker on Cloud Providers

#### AWS ECS / EC2
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker build -t audio-analysis-server .
docker tag audio-analysis-server:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/audio-analysis-server:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/audio-analysis-server:latest
```

#### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/audio-analysis-server
gcloud run deploy audio-analysis-server \
  --image gcr.io/PROJECT-ID/audio-analysis-server \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10
```

#### Azure Container Instances
```bash
# Build and push to ACR
az acr build --registry myregistry --image audio-analysis-server .
az container create \
  --resource-group myResourceGroup \
  --name audio-analysis-server \
  --image myregistry.azurecr.io/audio-analysis-server:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000
```

### Option 2: Local Production Server

```bash
# Clone and setup
git clone <your-repo>
cd audio-analysis-server

# Install production dependencies
pip install --no-cache-dir -r requirements.txt

# Run with production WSGI server
gunicorn app:app --bind 0.0.0.0:8000 --workers 4 --timeout 300
```

### Option 3: Railway Deployment

Create `railway.toml`:
```toml
[build]
builder = "dockerfile"

[deploy]
restartPolicyType = "always"
```

Deploy:
```bash
railway login
railway link
railway up
```

## 🔧 Environment Configuration

### Supabase Environment Variable

Set in Supabase Dashboard → Settings → API → Environment variables:

```bash
AUDIO_ANALYSIS_SERVER_URL=https://your-deployed-server.com
```

### Server Environment Variables

```bash
# Optional: Configure analysis parameters
export MAX_AUDIO_DURATION=180
export ANALYSIS_SAMPLE_RATE=22050
export REDIS_URL=redis://localhost:6379  # If using Redis caching
```

## 📊 Production Monitoring

### Health Check Endpoint
```bash
curl https://your-server.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "dependencies": {
    "librosa": "0.10.1",
    "essentia": "2.1b6",
    "aubio": "0.4.9"
  }
}
```

### Performance Monitoring
- Monitor CPU usage (high during analysis)
- Monitor memory usage (~500MB per analysis)
- Set up alerts for 500 errors
- Monitor response times (target: <5 seconds)

## 🔒 Production Security

### Firewall Rules
```bash
# Allow only HTTP traffic on port 8000
# Block all other ports
# Restrict access to known IP ranges if possible
```

### Rate Limiting (Optional)
Add to `app.py`:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/analyze")
@limiter.limit("10/minute")  # 10 requests per minute per IP
async def analyze_audio(request: Request, audio: UploadFile):
    # ... existing code
```

## 📈 Scaling Considerations

### Horizontal Scaling
- Deploy multiple instances behind load balancer
- Use Redis for caching repeated analysis
- Implement request queuing for high traffic

### Vertical Scaling
- Minimum: 2 CPU cores, 4GB RAM
- Recommended: 4 CPU cores, 8GB RAM
- Storage: 10GB (for temporary files)

## 🚨 Troubleshooting

### Common Issues

1. **Memory errors during analysis**
   - Increase container memory to 4GB+
   - Reduce MAX_AUDIO_DURATION

2. **Timeout errors**
   - Increase timeout settings
   - Optimize analysis parameters

3. **Library import errors**
   - Ensure all system dependencies installed
   - Check Docker base image compatibility

### Logs Analysis
```bash
# Docker logs
docker logs -f audio-analysis-server

# Check for errors
docker logs audio-analysis-server 2>&1 | grep ERROR
```

## ✅ Production Checklist

- [ ] Audio analysis server deployed and accessible
- [ ] Health check endpoint responding
- [ ] Supabase environment variable set
- [ ] Test analysis with sample audio file
- [ ] Monitor logs for errors
- [ ] Set up alerting for downtime
- [ ] Configure automatic restarts
- [ ] Document server URL for team

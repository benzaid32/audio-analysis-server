# Professional Audio Analysis Server

Enterprise-grade audio analysis using **librosa**, **essentia**, and **aubio** for perfect musical alignment.

## 🎯 Features

- **Tempo Detection**: Precise BPM analysis using beat tracking
- **Key Detection**: Musical key and scale analysis
- **Beat Alignment**: Extract beat times and downbeats for synchronization
- **Harmonic Analysis**: Advanced spectral and tonal analysis
- **Onset Detection**: Real-time musical event detection
- **Confidence Scoring**: Analysis quality metrics

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
# Build and run
docker-compose up --build

# Or build manually
docker build -t audio-analysis-server .
docker run -p 8000:8000 audio-analysis-server
```

## 📡 API Endpoints

### POST /analyze
Analyze audio file and return musical features.

**Request:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "audio=@song.wav"
```

**Response:**
```json
{
  "success": true,
  "analysis": {
    "tempo": 126.5,
    "key": "C",
    "scale": "major",
    "mode": "major",
    "energy": 0.72,
    "confidence": 0.89,
    "beat_times": [0.0, 0.474, 0.948, 1.422],
    "downbeats": [0.0, 1.896, 3.792],
    "duration": 30.5,
    "beat_duration": 0.474,
    "bar_duration": 1.896,
    "total_bars": 16,
    "onsets": [0.0, 0.23, 0.47, 0.94],
    "segments": [0.0, 7.8, 15.6, 23.4],
    "spectral_features": {
      "spectral_centroid": 2156.7,
      "spectral_rolloff": 4512.3,
      "zero_crossing_rate": 0.034
    }
  }
}
```

### GET /health
Health check endpoint.

## 🔧 Environment Variables

- `AUDIO_ANALYSIS_SERVER_URL`: External URL for the analysis server
- `MAX_AUDIO_DURATION`: Maximum audio duration in seconds (default: 180)
- `ANALYSIS_SAMPLE_RATE`: Sample rate for analysis (default: 22050)

## 🎵 Integration with Supabase

Set environment variable in Supabase:
```
AUDIO_ANALYSIS_SERVER_URL=https://your-server.com
```

The Supabase edge function will automatically forward requests to this server.

## 📊 Professional Analysis Features

### Librosa Features
- Tempo and beat tracking
- Onset detection
- Spectral analysis
- Audio segmentation

### Essentia Features  
- Key detection algorithm
- Harmonic analysis
- Tonal descriptors
- Audio quality metrics

### Aubio Features
- Real-time onset detection
- Pitch tracking
- Beat tracking validation
- Temporal feature analysis

## 🏗️ Architecture

```
Frontend → Supabase Edge Function → Professional Analysis Server
                                         ↓
                              librosa + essentia + aubio
                                         ↓
                                 Musical Features
```

## 🚨 Production Considerations

- Set appropriate CPU and memory limits
- Configure load balancing for high traffic
- Add Redis caching for repeated analysis
- Monitor server health and performance
- Set up proper logging and error tracking

## 📈 Performance

- **Analysis Time**: ~2-5 seconds for 30-second audio
- **Memory Usage**: ~500MB peak during analysis
- **CPU Usage**: High during analysis, low when idle
- **Supported Formats**: WAV, MP3, FLAC, M4A

## 🔒 Security

- No audio data is stored permanently
- All uploaded files are processed in memory
- CORS configured for cross-origin requests
- No user authentication required (stateless)

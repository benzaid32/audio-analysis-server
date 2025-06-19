#!/usr/bin/env python3
"""
Professional Audio Analysis Server
Using librosa, essentia, and aubio for enterprise-grade music analysis
"""

import librosa
import essentia.standard as es
import aubio
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Professional Audio Analysis", version="1.0.0")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProfessionalAudioAnalyzer:
    """Enterprise-grade audio analysis using industry-standard libraries"""
    
    def __init__(self):
        logger.info("🎵 Initializing Professional Audio Analyzer")
        
    def analyze_audio(self, audio_file_path: str) -> Dict:
        """
        Comprehensive audio analysis using librosa, essentia, and aubio
        Returns professional-grade musical features for perfect alignment
        """
        try:
            logger.info(f"🔍 Analyzing audio file: {audio_file_path}")
            
            # Load audio with librosa (industry standard)
            y, sr = librosa.load(audio_file_path, sr=None, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)
            
            logger.info(f"📊 Audio loaded: {duration:.2f}s at {sr}Hz")
            
            # === LIBROSA: Professional tempo and beat extraction ===
            tempo, beat_frames = librosa.beat.beat_track(
                y=y, 
                sr=sr, 
                trim=False,
                hop_length=512,
                units='frames'
            )
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            
            # Advanced tempo analysis with confidence
            onset_envelope = librosa.onset.onset_strength(y=y, sr=sr)
            tempo_variants = librosa.beat.tempo(onset_envelope=onset_envelope, sr=sr, aggregate=None)
            tempo_confidence = np.std(tempo_variants) / np.mean(tempo_variants)
            
            logger.info(f"🥁 Tempo detected: {tempo:.2f} BPM (confidence: {1-tempo_confidence:.2f})")
            
            # === ESSENTIA: Industry-standard key detection ===
            key_extractor = es.KeyExtractor()
            key, scale, key_strength = key_extractor(y.astype(np.float32))
            
            # Enhanced harmonic analysis
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            dominant_pitch_class = np.argmax(chroma_mean)
            pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            logger.info(f"🎹 Key detected: {key} {scale} (strength: {key_strength:.2f})")
            
            # === AUBIO: Precise onset detection for perfect alignment ===
            hop_size = 512
            onset_detector = aubio.onset("energy", 1024, hop_size, sr)
            
            onsets = []
            total_frames = 0
            
            while total_frames < len(y):
                chunk = y[total_frames:total_frames + hop_size]
                if len(chunk) < hop_size:
                    chunk = np.pad(chunk, (0, hop_size - len(chunk)), mode='constant')
                
                if onset_detector(chunk.astype(np.float32)):
                    onset_time = total_frames / float(sr)
                    onsets.append(onset_time)
                
                total_frames += hop_size
            
            logger.info(f"🎯 Detected {len(onsets)} onsets")
            
            # === Advanced Musical Structure Analysis ===
            # Segment detection for phrase structure
            chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
            segment_boundaries = librosa.segment.agglomerative(chroma_cqt, k=8)
            segment_times = librosa.frames_to_time(segment_boundaries, sr=sr)
            
            # Energy analysis for dynamics
            rms = librosa.feature.rms(y=y, hop_length=512)[0]
            energy_mean = np.mean(rms)
            energy_std = np.std(rms)
            
            # Spectral features for timbre analysis
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # Calculate bar structure for perfect alignment
            beat_duration = 60.0 / tempo
            bar_duration = beat_duration * 4  # Assuming 4/4 time
            total_bars = duration / bar_duration
            
            # Downbeat detection for precise alignment
            downbeats = []
            for i, beat_time in enumerate(beat_times):
                if i % 4 == 0:  # Every 4th beat is a downbeat
                    downbeats.append(beat_time)
            
            return {
                # Core musical features
                "tempo": float(tempo),
                "tempo_confidence": float(1 - tempo_confidence),
                "key": key,
                "scale": scale,
                "key_strength": float(key_strength),
                "mode": "major" if scale == "major" else "minor",
                
                # Timing and rhythm
                "beat_times": beat_times.tolist(),
                "downbeats": downbeats,
                "onsets": onsets,
                "beat_duration": float(beat_duration),
                "bar_duration": float(bar_duration),
                "total_bars": float(total_bars),
                
                # Musical structure
                "segments": segment_times.tolist(),
                "duration": float(duration),
                
                # Audio characteristics
                "energy": float(energy_mean),
                "energy_variance": float(energy_std),
                "spectral_centroid": float(np.mean(spectral_centroids)),
                "spectral_rolloff": float(np.mean(spectral_rolloff)),
                
                # Professional metadata
                "sample_rate": int(sr),
                "analysis_method": "librosa + essentia + aubio",
                "confidence": float((1 - tempo_confidence + key_strength) / 2)
            }
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")

# Initialize analyzer
analyzer = ProfessionalAudioAnalyzer()

@app.post("/analyze")
async def analyze_audio_endpoint(audio: UploadFile = File(...)):
    """
    Professional audio analysis endpoint
    Returns comprehensive musical features for perfect alignment
    """
    try:
        logger.info(f"📨 Received audio file: {audio.filename}")
        
        # Enterprise-grade file validation
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
            
        # Validate file extension for audio files (more reliable than content_type)
        valid_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        file_ext = os.path.splitext(audio.filename.lower())[1]
        
        if file_ext not in valid_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported audio format: {file_ext}. Supported formats: {', '.join(valid_extensions)}"
            )
        
        # Additional content type validation (with null safety)
        if audio.content_type and not audio.content_type.startswith('audio/'):
            logger.warning(f"Content type mismatch: {audio.content_type} for file {audio.filename}")
            # Don't fail - trust the file extension for enterprise flexibility
        
        logger.info(f"🎵 Processing {file_ext.upper()} file: {audio.filename}")
        
        # Create temporary file with original extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await audio.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Perform professional analysis
            analysis_result = analyzer.analyze_audio(tmp_file_path)
            
            logger.info(f"✅ Analysis completed: {analysis_result['key']} {analysis_result['scale']}, {analysis_result['tempo']:.1f} BPM")
            
            return {
                "success": True,
                "analysis": analysis_result,
                "message": "Professional audio analysis completed",
                "service": "Enterprise Audio Analysis Server"
            }
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"❌ Analysis endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Professional Audio Analysis Server",
        "libraries": {
            "librosa": librosa.__version__,
            "essentia": "2.1b6+",
            "aubio": "0.4.9+"
        }
    }

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Professional Audio Analysis Server",
        "description": "Enterprise-grade audio analysis using librosa, essentia, and aubio",
        "endpoints": {
            "/analyze": "POST - Upload audio file for comprehensive analysis",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        },
        "features": [
            "Professional tempo detection (librosa)",
            "Industry-standard key detection (essentia)",
            "Precise onset detection (aubio)",
            "Beat and downbeat tracking",
            "Musical structure analysis",
            "Harmonic and spectral features"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

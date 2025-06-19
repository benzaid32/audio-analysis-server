#!/usr/bin/env python3
"""
Test Professional Audio Analysis Server
"""

import requests
import os
import sys
import json
from pathlib import Path

# Configuration
SERVER_URL = "http://localhost:8000"
TEST_AUDIO = Path(__file__).parent.parent / "test-files" / "sample.wav"

def create_test_audio_if_missing():
    """Create a test audio file if it doesn't exist"""
    if not TEST_AUDIO.parent.exists():
        TEST_AUDIO.parent.mkdir(parents=True, exist_ok=True)
    
    if not TEST_AUDIO.exists():
        print(f"⚠️  Test audio file not found: {TEST_AUDIO}")
        print("⚙️  Generating a test sine wave audio file...")
        
        try:
            import numpy as np
            from scipy.io import wavfile
            
            # Generate a simple sine wave
            sample_rate = 44100
            duration = 5  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # Generate two sine waves (C4 = 261.63 Hz, E4 = 329.63 Hz)
            note_c4 = np.sin(261.63 * 2 * np.pi * t)
            note_e4 = np.sin(329.63 * 2 * np.pi * t)
            
            # Combine the tones with an amplitude adjustment
            audio_data = (note_c4 + note_e4) * 0.3
            
            # Convert to 16-bit PCM
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # Save the WAV file
            wavfile.write(TEST_AUDIO, sample_rate, audio_data)
            print(f"✅ Test audio created: {TEST_AUDIO}")
            
        except ImportError:
            print("❌ Could not generate test audio - missing dependencies")
            print("Please install numpy and scipy: pip install numpy scipy")
            sys.exit(1)

def test_health_check():
    """Test the health check endpoint"""
    print("\n🔍 Testing health check endpoint...")
    
    try:
        response = requests.get(f"{SERVER_URL}/health")
        response.raise_for_status()
        
        print("✅ Health check successful:")
        print(json.dumps(response.json(), indent=2))
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_audio_analysis():
    """Test the audio analysis endpoint with a sample file"""
    print("\n🔍 Testing audio analysis endpoint...")
    
    if not TEST_AUDIO.exists():
        print(f"❌ Test audio file not found: {TEST_AUDIO}")
        return False
    
    try:
        with open(TEST_AUDIO, 'rb') as f:
            files = {'audio': (TEST_AUDIO.name, f, 'audio/wav')}
            
            print(f"📤 Sending audio file: {TEST_AUDIO}")
            response = requests.post(f"{SERVER_URL}/analyze", files=files)
            response.raise_for_status()
            
            result = response.json()
            
            print("✅ Audio analysis successful:")
            print(f"  ♪ Tempo: {result['analysis']['tempo']:.1f} BPM")
            print(f"  ♪ Key: {result['analysis']['key']} {result['analysis']['scale']}")
            print(f"  ♪ Confidence: {result['analysis']['confidence']:.2f}")
            print(f"  ♪ Energy: {result['analysis']['energy']:.2f}")
            print(f"  ♪ Beat Duration: {result['analysis']['beat_duration']:.3f}s")
            print(f"  ♪ Bar Duration: {result['analysis']['bar_duration']:.3f}s")
            
            # Print the first few beat times
            beat_times = result['analysis']['beat_times'][:5]
            print(f"  ♪ Beat Times: {', '.join(f'{t:.2f}s' for t in beat_times)}...")
            
            return True
    except Exception as e:
        print(f"❌ Audio analysis failed: {e}")
        return False

def update_supabase_config():
    """Update Supabase environment configuration for local testing"""
    print("\n🔧 To test with Supabase edge function locally:")
    print("1. Run the following command in your Supabase project:")
    print("   supabase secrets set AUDIO_ANALYSIS_SERVER_URL=http://localhost:8000")
    print("2. Deploy the updated edge function:")
    print("   supabase functions deploy audio-analysis")
    print("3. Test with a sample request:")
    print("   curl -X POST -H \"Content-Type: application/json\" \\\n"
          "        -d '{\"audioUrl\":\"https://example.com/sample.wav\"}' \\\n"
          "        https://YOUR_PROJECT_REF.functions.supabase.co/audio-analysis")

def main():
    print("=" * 60)
    print("Professional Audio Analysis Server Tester")
    print("=" * 60)
    
    create_test_audio_if_missing()
    
    health_ok = test_health_check()
    
    if health_ok:
        analysis_ok = test_audio_analysis()
        
        if analysis_ok:
            print("\n✨ SUCCESS: Professional audio analysis server is working correctly!")
            update_supabase_config()
        else:
            print("\n❌ Audio analysis test failed. Please check the server logs.")
    else:
        print("\n❌ Server health check failed. Is the server running?")
        print(f"Please make sure the server is running at {SERVER_URL}")
        print("Run the server with: python app.py")

if __name__ == "__main__":
    main()

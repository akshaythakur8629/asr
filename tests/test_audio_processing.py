import tempfile, unittest, wave
from pathlib import Path
import numpy as np
from audio_processing import normalize_audio, read_pcm16_wav, slice_wav, wav_duration, write_pcm16_wav
class AudioTests(unittest.TestCase):
    def test_write_read_and_slice(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); audio=(np.arange(16000,dtype=np.int32)%2000-1000).astype('<i2')
            source=write_pcm16_wav(root/'source.wav',audio.tobytes(),16000); target=slice_wav(source,root/'slice.wav',.25,.75)
            pcm,sr=read_pcm16_wav(target); self.assertEqual(sr,16000); self.assertEqual(len(pcm)//2,8000); self.assertAlmostEqual(wav_duration(target),.5)
    def test_normalize_stereo_8k(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'source.wav'
            with wave.open(str(source),'wb') as f:
                f.setnchannels(2);f.setsampwidth(2);f.setframerate(8000);f.writeframes(np.zeros(8000*2,dtype='<i2').tobytes())
            target=normalize_audio(source,root/'normalized.wav'); pcm,sr=read_pcm16_wav(target)
            self.assertEqual(sr,16000);self.assertEqual(len(pcm)//2,16000)
if __name__=='__main__':unittest.main()

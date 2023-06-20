from essentia import Pool, array
import essentia.standard as estd
import numpy as np
import librosa   


class ChromaFeatures:
    """
    Class containing methods to compute various chroma features
    Methods :
                chroma_stft            : Computes chromagram using short fourier transform
                chroma_cqt             : Computes chromagram from constant-q transform of the audio signal
                chroma_cens            : Computes improved chromagram using CENS method 
                chroma_vqt             : Computes Variable-Q chromagram for the input audio signal
                chroma_hpcp            : Computes Harmonic pitch class profiles aka HPCP (improved chromagram)
                mel_spectrogram        : Compute a mel-scaled spectrogram
                audio_tempogram        : Compute the tempogram: local autocorrelation of the onset strength envelope. 
                beat_sync_chroma       : Computes the beat-sync chromagram
                two_dim_fft_magnitudes : Computes 2d - fourier transform magnitude coefficiants of the input feature vector (numpy array). 
                                         Usually fed by Constant-q transform or chroma feature vectors for cover detection tasks.
    Example use :
                chroma = ChromaFeatures("./data/test_audio.wav")
                #chroma cens with default parameters
                chroma.chroma_cens()
                #chroma stft with default parameters
                chroma.chroma_stft()
    """

    def __init__(self, audio_file, mono=True, sample_rate=44100, normalize_gain=False):
        """"""
        self.fs = sample_rate
        if normalize_gain:
            self.audio_vector = estd.EasyLoader(filename=audio_file, sampleRate=self.fs, replayGain=-9)()
        else:
            self.audio_vector = estd.MonoLoader(filename=audio_file, sampleRate=self.fs)()
        print("== Audio vector of %s loaded with shape %s and sample rate %s ==" % (audio_file, self.audio_vector.shape, self.fs))
        return


    def chroma_stft(self, frameSize=4096, hopSize=2048, display=False):
        """
        Computes the chromagram from the short-term fourier transform of the input audio signal
        """
        chroma = librosa.feature.chroma_stft(y=self.audio_vector, sr=self.fs, tuning=0, norm=2, hop_length=hopSize, n_fft=frameSize)
        return np.swapaxes(chroma, 0, 1)


    def chroma_cqt(self, hopSize=2048, display=False):
        """
        Computes the chromagram feature from the constant-q transform of the input audio signal
        """
        chroma = librosa.feature.chroma_cqt(y=self.audio_vector, sr=self.fs, hop_length=hopSize)
        return np.swapaxes(chroma, 0, 1)


    def chroma_cens(self, hopSize=2048, display=False):
        '''
        Computes CENS chroma vectors for the input audio signal (numpy array)
        '''
        chroma_cens = librosa.feature.chroma_cens(y=self.audio_vector, sr=self.fs, hop_length=hopSize)
        return np.swapaxes(chroma_cens, 0, 1)


    def chroma_vqt(self, hopSize=2048, display=False):
        '''
        Computes Chroma Distances for the input audio signal
        '''
        chroma_vqt = librosa.feature.chroma_vqt(y=self.audio_vector, sr=self.fs, hop_length=hopSize)
        return np.swapaxes(chroma_vqt, 0, 1)


    def chroma_hpcp(self, frameSize=4096, hopSize=2048, windowType='blackmanharris62', harmonicsPerPeak=8, magnitudeThreshold=1e-05, maxPeaks=1000, 
                whitening=True, referenceFrequency=440, minFrequency=40, maxFrequency=5000, nonLinear=False, numBins=12, display=False):
        '''
        Compute Harmonic Pitch Class Profiles (HPCP) for the input audio files using essentia standard mode using the default parameters.
        Parameters:
            harmonicsPerPeak : (integer ∈ [0, ∞), default = 0) :
            number of harmonics for frequency contribution, 0 indicates exclusive fundamental frequency contribution
            maxFrequency : (real ∈ (0, ∞), default = 5000) :
            the maximum frequency that contributes to the HPCP [Hz] (the difference between the max and split frequencies must not be less than 200.0 Hz)

            minFrequency : (real ∈ (0, ∞), default = 40) :
            the minimum frequency that contributes to the HPCP [Hz] (the difference between the min and split frequencies must not be less than 200.0 Hz)

            nonLinear : (bool ∈ {true, false}, default = false) :
            apply non-linear post-processing to the output (use with normalized='unitMax'). Boosts values close to 1, decreases values close to 0.
            normalized (string ∈ {none, unitSum, unitMax}, default = unitMax) :
            whether to normalize the HPCP vector

            referenceFrequency : (real ∈ (0, ∞), default = 440) :
            the reference frequency for semitone index calculation, corresponding to A3 [Hz]

            sampleRate : (real ∈ (0, ∞), default = 44100) :
            the sampling rate of the audio signal [Hz]

            numBins : (integer ∈ [12, ∞), default = 12) :
            the size of the output HPCP (must be a positive nonzero multiple of 12)
            whitening : (boolean (True, False), default = False)
            Optional step of computing spectral whitening to the output from speakPeak magnitudes
        '''

        audio = array(self.audio_vector)
        frameGenerator = estd.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize)
        window = estd.Windowing(type=windowType)
        spectrum = estd.Spectrum()

        spectralPeaks = estd.SpectralPeaks(magnitudeThreshold=0, maxFrequency=maxFrequency, minFrequency=minFrequency, maxPeaks=maxPeaks, orderBy="frequency", sampleRate=self.fs)
        spectralWhitening = estd.SpectralWhitening(maxFrequency= maxFrequency, sampleRate=self.fs)
        hpcp = estd.HPCP(sampleRate=self.fs, maxFrequency=maxFrequency, minFrequency=minFrequency, referenceFrequency=referenceFrequency, nonLinear=nonLinear, harmonics=harmonicsPerPeak, size=numBins)

        pool = Pool()

        #compute hpcp for each frame and add the results to the pool
        for frame in frameGenerator:
            spectrum_mag = spectrum(window(frame))
            frequencies, magnitudes = spectralPeaks(spectrum_mag)
            if whitening:
                w_magnitudes = spectralWhitening(spectrum_mag, frequencies, magnitudes)
                hpcp_vector = hpcp(frequencies, w_magnitudes)
            else:
                hpcp_vector = hpcp(frequencies, magnitudes)
            pool.add('tonal.hpcp',hpcp_vector)

        return pool['tonal.hpcp']

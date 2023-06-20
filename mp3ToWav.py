from pydub import AudioSegment
import os

folder_path = '/root/audio_compare_api/Audios'
files = os.listdir(folder_path)

for f in files:
    wav_file = folder_path + "/" + str(f[:-4]) + ".wav"
    mp3_file = folder_path + "/"  + f

    sound = AudioSegment.from_mp3(mp3_file)
    sound.export(wav_file, format="wav")

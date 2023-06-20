from chroma_features import ChromaFeatures 
import cover_similarity_measures as sims
import os
from flask import Flask, request, jsonify
import librosa
import pymysql
import numpy as np
from dotenv import load_dotenv
import random
import datetime
import wave
import threading


app = Flask(__name__)
load_dotenv('var.env')
HOST = os.environ['HOST']
PORT = int(os.environ['PORT'])
USER = 'user'
PASSWORD = os.environ['PASSWORD']
DATABASE = os.environ['DATABASE']


def update_db(scores):
    # database connection
    connection = pymysql.connect(host=HOST, port=PORT, user=USER, passwd=PASSWORD, database=DATABASE)
    cursor = connection.cursor()
    get_query = "SELECT * FROM Scores WHERE UserId = %s"
    value = (scores['Uid'])
    cursor.execute(get_query,value)
    rows = cursor.fetchall()

    consistancy = rows[0][1]
    consistancy = (consistancy + scores['Consistancy'])/2
    pronounciation = rows[0][2]
    pronounciation = (consistancy + scores['Pronounciation'])/2
    intonation = rows[0][3]
    intonation = (consistancy + scores['Intonation'])/2
    pitch = rows[0][4]
    pitch = (consistancy + scores['Pitch'])/2
    speed = rows[0][5]
    speed = (consistancy + scores['Speed'])/2
    updateTime = datetime.datetime.now()

    update_query = "UPDATE `Scores` SET `Consistency` = %s, `Pronunciation` = %s, `Intonation` = %s, `Pitch` = %s, `Speed` = %s, `updatedAt` = %s WHERE `Scores`.`UserId` = %s"
    values = (consistancy,pronounciation,intonation,pitch,speed,updateTime,scores['Uid'])
    cursor.execute(update_query,values)
    connection.commit()
    connection.close()


def get_chroma_features(q_audio, ref_audio):
    # Initiate the ChromaFeatures class 
    query_audio = ChromaFeatures(audio_file=q_audio, mono=True, sample_rate=44100) 
    reference_audio = ChromaFeatures(audio_file=ref_audio, mono=True, sample_rate=44100) 

    # compute chroma features for query song using default params
    c_stft_query = query_audio.chroma_stft()
    c_sqt_query = query_audio.chroma_cqt()
    c_cens_query = query_audio.chroma_cens()
    c_hpcp_query = query_audio.chroma_hpcp()
    # compute chroma features for reference song using default params
    c_stft_reference = reference_audio.chroma_stft()
    c_sqt_reference = reference_audio.chroma_cqt()
    c_cens_reference = reference_audio.chroma_cens()
    c_hpcp_reference = reference_audio.chroma_hpcp()

    sim_matrix = []
    sim_matrix.append(sims.cross_recurrent_plot(c_stft_query, c_stft_reference))
    sim_matrix.append(sims.cross_recurrent_plot(c_sqt_query, c_sqt_reference))
    sim_matrix.append(sims.cross_recurrent_plot(c_cens_query, c_cens_reference))
    sim_matrix.append(sims.cross_recurrent_plot(c_hpcp_query, c_hpcp_reference))

    similarity = []
    #Computing qmax cover song audio similarity measure (distance)
    for sim_mat in sim_matrix:
        qmax, cost_matrix = sims.qmax_measure(sim_mat)
        similarity.append(100 - (qmax*100))

    #Maximum similarity of query audio and reference audio
    max_sim_pct = max(similarity)

    return max_sim_pct


def getScores(ref_audio, q_audio, uid):
    y_ref, sample_rate = librosa.load(ref_audio)
    ref_duration = librosa.get_duration(y_ref, sample_rate)

    y_comp, sample_rate = librosa.load(q_audio)
    q_duration = librosa.get_duration(y_comp, sample_rate)

    if q_duration == ref_duration:
        speed = 100
    try:
        speed = (abs(q_duration - ref_duration) / ref_duration)* 100.0
    except ZeroDivisionError:
        pass
    
    pronounciation_score = get_chroma_features(q_audio, ref_audio)

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y_ref)), ref=np.max)
    D_comp = librosa.amplitude_to_db(np.abs(librosa.stft(y_comp)), ref=np.max)
    dsim = librosa.segment.cross_similarity(D_comp, D, metric='cosine', mode='affinity')
    pitch = (abs(np.max(dsim)) - abs(np.std(dsim)) - abs(np.mean(dsim)))*100

    hop_length = 1024
    chroma_ref = librosa.feature.chroma_cqt(y=y_ref, sr=sample_rate, hop_length=hop_length)
    chroma_comp = librosa.feature.chroma_cqt(y=y_comp, sr=sample_rate, hop_length=hop_length)
    x_ref = librosa.feature.stack_memory(chroma_ref, n_steps=1, delay=3)
    x_comp = librosa.feature.stack_memory(chroma_comp, n_steps=1, delay=3)
    xsim_aff = librosa.segment.cross_similarity(x_comp, x_ref, metric='cosine', mode='affinity')
    xsim_aff = np.ma.masked_equal(xsim_aff,0)
    xsim_aff.compressed()
    intonation = abs(np.mean(xsim_aff))*100

    flatness_ref = librosa.feature.spectral_flatness(y=y_ref)
    flatness_comp = librosa.feature.spectral_flatness(y=y_comp)
    fsim = librosa.segment.cross_similarity(flatness_comp, flatness_ref, mode='affinity')
    fsim = np.ma.masked_equal(fsim,0)
    fsim.compressed()
    consistancy = abs(np.mean(fsim))*100

    scores = {
        "Consistancy": consistancy,
        "Pronounciation": pronounciation_score,
        "Intonation": intonation,
        "Pitch": pitch,
        "Speed": speed,
        "Uid": uid
    }

    print(scores)
    update_db(scores)


@app.route('/match/CompareAudio', methods=['POST'])
def compareAudios():
    ref_audio = str(request.form.get('ref_audio_name'))
    q_audio = str(request.form.get('q_audio'))
    q_audio = "/root/recite-pro-backend/backend/userRecordings" + q_audio
    print(q_audio)
    # q_audio = request.files['q_audio'].read()
    uid = int(request.form.get('uid'))
    
    # nchannels = 1
    # sampwidth = 2
    # framerate = 44100
    # nframes = len(q_audio)
    
    # with wave.open("/root/audio_compare_api/q.wav", 'wb') as wav_file:
    #     wav_file.setnchannels(nchannels)
    #     wav_file.setsampwidth(sampwidth)
    #     wav_file.setframerate(framerate)
    #     wav_file.setnframes(nframes)
    #     wav_file.writeframes(q_audio)

    ref_audio = "/root/audio_compare_api/Audios/" + ref_audio + ".wav"
    audio, sample_rate = librosa.load(ref_audio)
    ref_duration = librosa.get_duration(audio, sample_rate)

    # q_audio = "/root/audio_compare_api/q.wav"
    print(q_audio)
    audio, sample_rate = librosa.load(q_audio)
    q_duration = librosa.get_duration(audio, sample_rate)
    
    threshold = ref_duration*0.25

    if q_duration >= threshold:
        thread = threading.Thread(target=getScores, args=(ref_audio,q_audio,uid))
        thread.start()

        max_sim_pct = get_chroma_features(q_audio, ref_audio)
        label = ""

        if max_sim_pct <= 45:
            labels = ["Practice More!","You can do better!"]
            label = random.sample(labels,1)
        elif max_sim_pct <= 60 and max_sim_pct > 45:
            labels = ["Better!","Good work","Great effort","Not Bad","Good try"]
            label = random.sample(labels,1)
        elif max_sim_pct <= 75 and max_sim_pct > 60:
            labels = ["Perfect!","Sounds beautiful","Great","Sounds amazing"]
            label = random.sample(labels,1)
        elif max_sim_pct > 75:
            labels = ["Fantastic!","Stupendous","Superb","Incredible","Wow","Fabulous"]
            label = random.sample(labels,1)
        
        print()
        print("Maximum Accuracy: ", max_sim_pct)
        print("Label: ", label)

        os.remove(q_audio)

        # retrunObj = {
        #     "label": str(label[0]),
        #     "score": max_sim_pct
        # }

        return jsonify((str(label[0]), round(max_sim_pct, 1)))
    else:
        return jsonify(("Too fast, please try again!!", 0.0))


@app.route("/match/testpost", methods=['POST'])
def testPost():
    if request.method == 'POST':
        name = request.form.get('ref_audio')
        uid = request.form.get('uid')

        # audio_file = request.files['file']
        # a, sr = librosa.load(audio_file)
        # ref_duration = librosa.get_duration(a, sr)

        return str(uid)
    else:
        return "API GET"


if __name__ == "__main__":
    app.run(host='0.0.0.0')

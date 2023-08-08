import threading
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
import streamlit as st
from youtubesearchpython import *
from pytube import YouTube
from pydub import AudioSegment
import moviepy.editor as mp
import concurrent.futures
import os
import shutil
import math
def downloading(l,j):
    yt = YouTube(l)
    yt = yt.streams.get_by_itag(18)
    yt.download('videos/',filename='song'+f'{j}'+'.mp4')

def convert(index):
    video_file = 'videos/song{}.mp4'.format(index)
    audio_file = 'audios/song{}.mp3'.format(index)
    
    clip = mp.VideoFileClip(video_file)
    audio = clip.audio
    audio.write_audiofile(audio_file)
    
def converting(n):
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(convert, index) for index in range(n)]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(e)
                
def cutting(z,m,output):
        file='audios/song'+str(m)+'.mp3'
        sound = AudioSegment.from_mp3(file)
    #Selecting Portion we want to cut
        StrtMin = 0
        StrtSec = 0
        EndMin = 0
        EndSec = min(z,sound.duration_seconds)
    # Time to milliseconds conversion
        StrtTime = StrtMin*60*1000+StrtSec*1000
        EndTime = EndMin*60*1000+EndSec*1000
    # Opening file and extracting portion of it
        extract = sound[StrtTime:EndTime]
        output.append(extract)  

def searchAndCreateMashup(result_frame,n,dur):
    z=dur
    s=math.ceil(n/19)
    list=[]
    for index, row in result_frame.iterrows():
        song_name = row['Song Name']
        artist_name = row['Artist Name']
        search_query = f"{song_name} {artist_name} official music video"  # Customizing the search query for better results
        result = CustomSearch(search_query,VideoDurationFilter.short)
        if result.result()['result'][0]['duration'] != 'Live':
            list.append(result.result()['result'][0]['link'])
    l1 = []
    count = 0
    for item in list:
        if item not in l1:
            count += 1
            l1.append(item)
    data = [(element,index) for index,element in enumerate(list[:n])]
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(lambda x: downloading(*x), data)
    converting(n)
    merged=AudioSegment.empty()
    output=[]
    data2 = [(z, index, output) for index in range(n)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor2:
        executor2.map(lambda x: cutting(*x), data2)
    for i in output:
         merged=merged+i
    merged.export('media/mashup.mp3', format='mp3')
    import zipfile
    zipObj = zipfile.ZipFile('media/mashup.zip', 'w')
    zipObj.write('media/mashup.mp3')
    zipObj.close()
    
    
    
    
def main():
    client_credentials_manager = SpotifyClientCredentials(
        client_id="71ba9810ebd04753bef391b4d5488e7e", client_secret="2336d2134d9c4a8db5a88c45a1cf7be4")
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    #Getting user playlist
    playlist_link = st.text_input("Enter your playlist link: ")
    if not playlist_link:
            st.warning("No matching playlists found.")
            st.stop()
    song_value=st.text_input("Write the song name you want to recommend from this playlist")
    if not song_value:
            st.warning("No matching song found.")
            st.stop()
    number=st.text_input("Number of Recommended Songs you want to see")

    if not number:
            st.warning("No matching number found.")
            st.stop()
    number=int(number)
    with st.spinner("Please wait while processing..."):

        playlist_URI = playlist_link.split("/")[-1].split("?")[0]
        offset = 0
        
        track_uris = []
        for i in range(100):
            a=[x["track"]["uri"] for x in sp.playlist_tracks(playlist_URI,offset=offset)["items"]]
            print("hello")
            for i in a:
                track_uris.append(i)
            offset += 100
        images = []
        
        print("hello Done")
        for i in range(len(track_uris)):
            track = sp.track(track_uris[i])
            images.append(track["album"]["images"][0]["url"])

        #making a new csv file with th data and headers as the features of the audio features and also Song Name and Artist Name
        song_name = []
        artist_name = []
        acousticness = []
        danceability = []
        duration_ms = []
        energy = []
        instrumentalness = []
        loudness = []
        liveness=[]
        speechiness = []
        time_signature = []
        key = []
        valence = []
        tempo = []
        mode = []
        target = []
        url = []
        #get track name and artist name
        for track_uri in track_uris:
            track = sp.track(track_uri)
            song_name.append(track["name"])
            artist_name.append(track["artists"][0]["name"])
            audio_features = sp.audio_features(track_uri)
            acousticness.append(audio_features[0]["acousticness"])
            danceability.append(audio_features[0]["danceability"])
            duration_ms.append(audio_features[0]["duration_ms"])
            energy.append(audio_features[0]["energy"])
            instrumentalness.append(audio_features[0]["instrumentalness"])
            loudness.append(audio_features[0]["loudness"])
            liveness.append(audio_features[0]["liveness"])
            speechiness.append(audio_features[0]["speechiness"])
            time_signature.append(audio_features[0]["time_signature"])
            key.append(audio_features[0]["key"])
            valence.append(audio_features[0]["valence"])
            tempo.append(audio_features[0]["tempo"])
            mode.append(audio_features[0]["mode"])
            target.append(1)
            url.append(track["external_urls"]["spotify"])
        #make a csv file with the data
        df = pd.DataFrame({'Song Name':song_name, 'Artist Name':artist_name, 'Acousticness':acousticness, 'Danceability':danceability, 'Duration_ms':duration_ms, 'Energy':energy, 'Instrumentalness':instrumentalness, 'Loudness':loudness,'Liveness':liveness ,'Speechiness':speechiness, 'Time_signature':time_signature, 'Key':key, 'Valence':valence, 'Tempo':tempo, 'Mode':mode, 'Target':target, 'URL':url, 'Image':images})
        

        from sklearn.preprocessing import MinMaxScaler
        feature_cols = ['Acousticness', 'Danceability', 'Duration_ms', 'Energy',
                        'Instrumentalness', 'Key', 'Liveness', 'Loudness', 'Mode',
                        'Speechiness', 'Tempo', 'Time_signature', 'Valence', ]

        scaler = MinMaxScaler()
        normalized_df = scaler.fit_transform(df[feature_cols])

        print(normalized_df[:2])

        # Create a pandas series with song titles as indices and indices as series values
        indices = pd.Series(df.index, index=df['Song Name']).drop_duplicates()

        # Create cosine similarity matrix based on given matrix
        cosine = cosine_similarity(normalized_df)


        def generate_recommendation(song_title,limit_songs, model_type=cosine):
            """
            Purpose: Function for song recommendations 
            Inputs: song title and type of similarity model
            Output: Pandas series of recommended songs
            """
            # Get song indices
            index = indices[song_title]
            global song_image
            global top_songs_link
            
            # Get list of songs for given songs
            score = list(enumerate(model_type[indices[song_title]]))
            # Sort the most similar songs
            similarity_score = sorted(score, key=lambda x: x[1], reverse=True)
            # Select the top-user input recommended songs
            similarity_score = similarity_score[1:limit_songs+1]
            top_songs_index = [i[0] for i in similarity_score]
            # Top user-input value recommended songs
            top_songs = df['Song Name'].iloc[top_songs_index]
            top_songs_artists = df['Artist Name'].iloc[top_songs_index]
            top_songs_link=df['URL'].iloc[top_songs_index]
            song_image=[df['Image'].iloc[i] for i in top_songs_index]
            result_frame=pd.DataFrame({'Song Name':top_songs, 'Artist Name':top_songs_artists})
            return result_frame
            
        for i in range(len(song_name)):
            print(song_name[i],":",artist_name[i])


        print("Recommended Songs:")
        result_frame=generate_recommendation(song_value,number,cosine)
        st.write(result_frame)
    st.success("Processing completed!")
    if not (os.path.exists('audios')):
        os.mkdir('audios')

    if not (os.path.exists('videos')):
        os.mkdir('videos')

    if not (os.path.exists('media')):
        os.mkdir('media')
    dur=20
    with st.spinner("untill we create mashup :"):
        searchAndCreateMashup(result_frame, number, dur)
    st.success("MashUp Created")
   
    
    st.write("Click the button below to download the mashup.")
    getMashup='media/mashup.mp3'
    def download_file(file_path, file_name, mime_type):
        with open(file_path, 'rb') as f:
            bytes_data = f.read()
        st.download_button(label='Click to download', data=bytes_data, file_name=file_name, mime=mime_type)

    # Usage example
    file_path = 'media/mashup.mp3'  # Replace this with the actual file path on your system
    file_name = 'mashup.mp3'
    mime_type = 'audio/mpeg'
    
    download_file(file_path, file_name, mime_type)
    shutil.rmtree('audios')
    shutil.rmtree('videos')
    shutil.rmtree('media')
if __name__=="__main__":
    main()
    

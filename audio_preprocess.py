
import os
import librosa
import math
import json
from pathlib import Path


DATASET_PATH = '../dataset/WaveClusters'
SAMPLE_RATE = 22050
DURATION = 30  # in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


# Save MFCC
def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=1):
    # dataset_path: dataset path
    # json_path: the path to save json
    # num_segments: divide audio files into several segments as diff samples (data augment)

    # dictionary to store data
    data = {
        "mapping": [],  # mapping diff labels onto numbers
        "mfcc": [],  # tonal color
        "labels": [],
        "id": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
    print('\nMFCC:')

    # loop through all the clusters
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
    
        # ensure not at the root level
        if dirpath is not dataset_path:

            # save the semantic label
            dirpath_components = dirpath.split('/')
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print('Processing {}'.format(semantic_label))

            # process files for a specific cluster
            for filename in filenames:

                # load audio file
                file_path = os.path.join(dirpath, filename)
                signal, sample_rate = librosa.load(file_path, sr = SAMPLE_RATE)

                # process segments, extract mfcc, store data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    # mfcc
                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr=sample_rate,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)

                    # if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(i-1)
                    data["id"].append(Path(filename).stem)
                        # print("{}, segment: {}".format(file_path, s+1))

    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4)
        print('Saved: ', len(data["mfcc"]))


"""### Save Melspectrogram"""


def save_melspectrogram(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=1):
    # dataset_path: dataset path
    # json_path: the path to save json
    # num_segments: divide audio files into several segments as diff samples (data augment)

    # dictionary to store data
    data = {
        "mapping": [],  # mapping diff labels onto numbers
        "melspectrogram": [],
        "labels": [],
        "id": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
    print('\nMelspectrogram:')

    # loop through all the clusters
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure not at the root level
        if dirpath is not dataset_path:

            # save the semantic label
            dirpath_components = dirpath.split('/')
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print('Processing {}'.format(semantic_label))

            # process files for a specific cluster
            for filename in filenames:

                # load audio file
                file_path = os.path.join(dirpath, filename)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process segments, extract mfcc, store data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    # melspectrogram
                    melspectrogram = librosa.feature.melspectrogram(signal[start_sample:finish_sample],
                                                                    sr=sample_rate,
                                                                    hop_length=hop_length)

                    # if len(melspectrogram) == expected_num_mfcc_vectors_per_segment:
                    data["melspectrogram"].append(melspectrogram.tolist())
                    data["labels"].append(i-1)
                    data["id"].append(Path(filename).stem)
                    # print("{}, segment: {}".format(file_path, s+1))

    with open(json_path, 'w') as fp:
      json.dump(data, fp, indent=4)
      print('Saved: ', len(data["melspectrogram"]))


"""### Save Tempo"""


def save_tempo(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=1):
    # dataset_path: dataset path
    # json_path: the path to save json
    # num_segments: divide audio files into several segments as diff samples (data augment)

    # dictionary to store data
    data = {
        "mapping": [],  # mapping diff labels onto numbers
        "tempo": [],  # rhythm, global tempo
        "labels": [],
        "id": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
    print('\nTempo:')

    # loop through all the clusters
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure not at the root level
        if dirpath is not dataset_path:

            # save the semantic label
            dirpath_components = dirpath.split('/')
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print('Processing {}'.format(semantic_label))

            # process files for a specific cluster
            for filename in filenames:

                # load audio file
                file_path = os.path.join(dirpath, filename)
                signal, sample_rate = librosa.load(file_path, sr = SAMPLE_RATE)

                # process segments, extract mfcc, store data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    # tempo
                    tempo = librosa.beat.beat_track(signal,
                                                    sr=sample_rate,
                                                    hop_length=hop_length)[0]

                    # if len(tempogram) == expected_num_mfcc_vectors_per_segment:
                data["tempo"].append(tempo.tolist())
                data["labels"].append(i-1)
                data["id"].append(Path(filename).stem)
                # print("{}, segment: {}".format(file_path, s+1))

    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4)
        print('Saved: ', len(data["tempo"]))


"""### Save RMS"""


def save_rms(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=1):
    # dataset_path: dataset path
    # json_path: the path to save json
    # num_segments: divide audio files into several segments as diff samples (data augment)

    # dictionary to store data
    data = {
        "mapping": [], # mapping diff labels onto numbers
        "rms": [], # dynamics
        "labels": [],
        "id": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
    print('\nRMS:')

    # loop through all the clusters
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure not at the root level
        if dirpath is not dataset_path:

            # save the semantic label
            dirpath_components = dirpath.split('/')
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print('Processing {}'.format(semantic_label))

            # process files for a specific cluster
            for filename in filenames:

                # load audio file
                file_path = os.path.join(dirpath, filename)
                signal, sample_rate = librosa.load(file_path, sr = SAMPLE_RATE)

                # process segments, extract mfcc, store data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    # rms
                    rms = librosa.feature.rms(signal[start_sample:finish_sample],
                                              hop_length=hop_length)

                    # if len(rms) == expected_num_mfcc_vectors_per_segment:
                    data["rms"].append(rms.tolist())
                    data["labels"].append(i-1)
                    data["id"].append(Path(filename).stem)
                    # print("{}, segment: {}".format(file_path, s+1))

    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4)
        print('Saved: ', len(data["rms"]))


"""### Save Chroma_cens"""


def save_chroma_cens(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=1):
    # dataset_path: dataset path
    # json_path: the path to save json
    # num_segments: divide audio files into several segments as diff samples (data augment)

    # dictionary to store data
    data = {
        "mapping": [],  # mapping diff labels onto numbers
        "chroma_cens": [],  # harmony
        "labels": [],
        "id": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
    print('\nChroma_cens:')

    # loop through all the clusters
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure not at the root level
        if dirpath is not dataset_path:

            # save the semantic label
            dirpath_components = dirpath.split('/')
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print('Processing {}'.format(semantic_label))

            # process files for a specific cluster
            for filename in filenames:

                # load audio file
                file_path = os.path.join(dirpath, filename)
                signal, sample_rate = librosa.load(file_path, sr = SAMPLE_RATE)

                # process segments, extract mfcc, store data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    # chroma_cens
                    chroma_cens = librosa.feature.chroma_cens(signal[start_sample:finish_sample],
                                                              sr=sample_rate,
                                                              hop_length=hop_length)

                    # if len(chroma_cens) == expected_num_mfcc_vectors_per_segment:
                    data["chroma_cens"].append(chroma_cens.tolist())
                    data["labels"].append(i-1)
                    data["id"].append(Path(filename).stem)
                    # print("{}, segment: {}".format(file_path, s+1))

    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4)
        print('Saved: ', len(data["chroma_cens"]))


"""### Main for audio feature"""


MFCC_PATH = '../dataset/json/mfcc.json'
MELSPECTROGRAM_PATH = '../dataset/json/melspectrogram.json'
TEMPO_PATH = '../dataset/json/tempo.json'
RMS_PATH = '../dataset/json/rms.json'
CHROMA_PATH = '../dataset/json/chroma_cens.json'

if __name__ == '__main__':
    if not os.path.exists('../dataset/json'):
        os.makedirs('../dataset/json')
    save_mfcc(DATASET_PATH, MFCC_PATH, num_segments = 1)
    save_melspectrogram(DATASET_PATH, MELSPECTROGRAM_PATH, num_segments = 1)
    save_tempo(DATASET_PATH, TEMPO_PATH, num_segments = 1)
    save_rms(DATASET_PATH, RMS_PATH, num_segments = 1)
    save_chroma_cens(DATASET_PATH, CHROMA_PATH, num_segments = 1)

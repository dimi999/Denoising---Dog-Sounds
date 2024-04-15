import copy
import random

import pandas
import pandas as pd
import librosa
import soundfile as sf
from pydub import AudioSegment

people_df = pd.read_csv('demographics.csv').to_numpy()
ct = 0

dataset_df = pandas.DataFrame(columns=['File', 'Cleaned_file'])

for i in range(1, 100):
    idx = str(i).zfill(4)
    adult_dog = AudioSegment.from_wav(f'dog-dataset/adult_dog/adult_dog_{idx}.wav')
    adult_dog = adult_dog - 10
    adult_dog.export(f'dog-dataset/adult_dog/adult_dog_{idx}.wav', 'wav')

    dogs = AudioSegment.from_wav(f'dog-dataset/dogs/dogs_{idx}.wav')
    dogs = dogs - 10
    dogs.export(f'dog-dataset/dogs/dogs_{idx}.wav', 'wav')

    puppy = AudioSegment.from_wav(f'dog-dataset/puppy/puppy_{idx}.wav')
    puppy = puppy - 10
    puppy.export(f'dog-dataset/puppy/puppy_{idx}.wav', 'wav')


for it in people_df:
    person_signal, sr = librosa.load(f'Clean-sounds/{it[2]}.wav')
    idx = random.randint(1, 100)
    idx = str(idx).zfill(4)

    adult_dog, sr = librosa.load(f'dog-dataset/adult_dog/adult_dog_{idx}.wav')
    sound1 = copy.deepcopy(person_signal)
    ct += 1
    if len(adult_dog) > len(person_signal):
        adult_dog = adult_dog[:len(person_signal)]
        for i in range(len(adult_dog)):
            sound1[i] += adult_dog[i]
    else:
        start_idx = random.randint(0, len(person_signal) - len(adult_dog))
        for i in range(start_idx, start_idx + len(adult_dog)):
            sound1[i] += adult_dog[i - start_idx]
    sf.write(f'Noisy-sounds/Sound{ct}.wav', sound1, sr)
    crt_df = pd.DataFrame({'File': f'Sound{ct}', 'Cleaned_file': it[2]}, columns=['File', 'Cleaned_file'], index=[0])
    dataset_df = pd.concat([dataset_df, crt_df], ignore_index=True)

    idx = random.randint(1, 100)
    idx = str(idx).zfill(4)

    dogs, sr = librosa.load(f'dog-dataset/dogs/dogs_{idx}.wav')
    sound1 = copy.deepcopy(person_signal)
    ct += 1
    if len(dogs) > len(person_signal):
        dogs = dogs[:len(person_signal)]
        for i in range(len(dogs)):
            sound1[i] += dogs[i]
    else:
        start_idx = random.randint(0, len(person_signal) - len(dogs))
        for i in range(start_idx, start_idx + len(dogs)):
            sound1[i] += dogs[i - start_idx]
    sf.write(f'Noisy-sounds/Sound{ct}.wav', sound1, sr)
    crt_df = pd.DataFrame({'File': f'Sound{ct}', 'Cleaned_file': it[2]}, columns=['File', 'Cleaned_file'], index=[0])
    dataset_df = pd.concat([dataset_df, crt_df], ignore_index=True)

    idx = random.randint(1, 100)
    idx = str(idx).zfill(4)

    puppy, sr = librosa.load(f'dog-dataset/puppy/puppy_{idx}.wav')
    sound1 = copy.deepcopy(person_signal)
    ct += 1
    if len(puppy) > len(person_signal):
        puppy = puppy[:len(person_signal)]
        for i in range(len(puppy)):
            sound1[i] += puppy[i]
    else:
        start_idx = random.randint(0, len(person_signal) - len(puppy))
        for i in range(start_idx, start_idx + len(puppy)):
            sound1[i] += puppy[i - start_idx]
    sf.write(f'Noisy-sounds/Sound{ct}.wav', sound1, sr)
    crt_df = pd.DataFrame({'File': f'Sound{ct}', 'Cleaned_file': it[2]}, columns=['File', 'Cleaned_file'], index=[0])
    dataset_df = pd.concat([dataset_df, crt_df], ignore_index=True)

dataset_df.to_csv('dataset.csv', index=False)
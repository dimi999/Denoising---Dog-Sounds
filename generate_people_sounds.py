import pandas as pd
import numpy as np
import random
import shutil

actors_df = pd.read_csv('CREMA-D/VideoDemographics.csv')

actors_df = actors_df.sort_values(by=['Age', 'Sex'])
actors_np = actors_df.to_numpy()

age_ids = dict()
age_ids['16-24'] = dict()
age_ids['16-24']['Male'] = []
age_ids['16-24']['Female'] = []
age_ids['25-34'] = dict()
age_ids['25-34']['Male'] = []
age_ids['25-34']['Female'] = []
age_ids['35-44'] = dict()
age_ids['35-44']['Male'] = []
age_ids['35-44']['Female'] = []
age_ids['45-54'] = dict()
age_ids['45-54']['Male'] = []
age_ids['45-54']['Female'] = []
age_ids['55+'] = dict()
age_ids['55+']['Male'] = []
age_ids['55+']['Female'] = []

distributions = dict()
distributions['16-24'] = [7.87, 7.87]
distributions['25-34'] = [7.32, 7.31]
distributions['35-44'] = [9.75, 9.75]
distributions['45-54'] = [7.32, 7.32]
distributions['55+'] = [14.25, 21.39]

def transform_number_to_category(x):
    s = 0
    for key, value in distributions.items():
        if s + value[0] >= x:
            return key, 'Male'
        elif s + value[0] + value[1] >= x:
            return key, 'Female'
        else:
            s += value[0] + value[1]


for it in actors_np:
    if it[1] <= 24:
        age_ids['16-24'][it[2]].append(it[0])
    elif it[1] <= 34:
        age_ids['25-34'][it[2]].append(it[0])
    elif it[1] <= 44:
        age_ids['35-44'][it[2]].append(it[0])
    elif it[1] <= 54:
        age_ids['45-54'][it[2]].append(it[0])
    else:
        age_ids['55+'][it[2]].append(it[0])

audios_df = pd.read_csv('CREMA-D/SentenceFilenames.csv').to_numpy()

audios_by_index = dict()

for it in audios_df:
    idx = int(it[1][:4])
    if idx in audios_by_index:
        audios_by_index[idx].append(it[1])
    else:
        audios_by_index[idx] = [it[1]]

my_dataset_df = pd.DataFrame(columns=['Age', 'Gender', 'Filename'])

ct = 0


for i in range(7500):
    x = np.random.rand() * 100
    key, gender = transform_number_to_category(x)

    if ct >= 50:
        break

    if len(age_ids[key][gender]) == 0:
        ct += 1
        continue

    id = random.choice(age_ids[key][gender])
    while len(audios_by_index[id]) == 0:
        id = random.choice(age_ids[key][gender])
    filename = random.choice(audios_by_index[id])
    audios_by_index[id].remove(filename)
    if len(audios_by_index[id]) == 0:
        age_ids[key][gender].remove(id)
    crt_df = pd.DataFrame({'Age': key, 'Gender': gender, 'Filename': filename}, columns=['Age', 'Gender', 'Filename'], index=[0])
    my_dataset_df = pd.concat([my_dataset_df, crt_df], ignore_index=True)
    shutil.copy(f'CREMA-D/AudioWAV/{filename}.wav', f'Clean-sounds/{filename}.wav')

print(ct)
my_dataset_df.to_csv('demographics.csv', index=False)






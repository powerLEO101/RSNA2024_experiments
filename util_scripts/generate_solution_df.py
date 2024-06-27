# %%
import pandas as pd

train_path = '../../input/rsna-2024-lumbar-spine-degenerative-classification/train.csv'

def to_label(meta):
    study_id = meta['study_id']
    keys = ['spinal_canal_stenosis_l1_l2',
       'spinal_canal_stenosis_l2_l3', 'spinal_canal_stenosis_l3_l4',
       'spinal_canal_stenosis_l4_l5', 'spinal_canal_stenosis_l5_s1',
       'left_neural_foraminal_narrowing_l1_l2',
       'left_neural_foraminal_narrowing_l2_l3',
       'left_neural_foraminal_narrowing_l3_l4',
       'left_neural_foraminal_narrowing_l4_l5',
       'left_neural_foraminal_narrowing_l5_s1',
       'right_neural_foraminal_narrowing_l1_l2',
       'right_neural_foraminal_narrowing_l2_l3',
       'right_neural_foraminal_narrowing_l3_l4',
       'right_neural_foraminal_narrowing_l4_l5',
       'right_neural_foraminal_narrowing_l5_s1',
       'left_subarticular_stenosis_l1_l2', 'left_subarticular_stenosis_l2_l3',
       'left_subarticular_stenosis_l3_l4', 'left_subarticular_stenosis_l4_l5',
       'left_subarticular_stenosis_l5_s1', 'right_subarticular_stenosis_l1_l2',
       'right_subarticular_stenosis_l2_l3',
       'right_subarticular_stenosis_l3_l4',
       'right_subarticular_stenosis_l4_l5',
       'right_subarticular_stenosis_l5_s1']
    label = [0] * 75
    for i, name in enumerate(keys):
        if meta[name] == -1: continue
        label[int(i * 3 + meta[name])] = 1
    keys = [f'{study_id}_{x}' for x in keys]
    df_sol = []
    for i in range(25):
        df_sol.append([keys[i], *label[i * 3:(i + 1) * 3]])
        if sum(df_sol[-1][1:]) == 0:
            df_sol[-1].append(0)
        elif df_sol[-1][1] == 1:
            df_sol[-1].append(1)
        elif df_sol[-1][2] == 1:
            df_sol[-1].append(2)
        elif df_sol[-1][3] == 1:
            df_sol[-1].append(4)
        else:
            print(f'Error in processing {study_id}')
    #df_sol = pd.DataFrame(df_sol, columns=['row_id', 'normal_mild', 'moderate', 'severe', 'sample_weight'])
    return df_sol

df = pd.read_csv(train_path)
df = df.fillna(-1)
df = df.replace({'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2})

df_sol = []
for i in range(len(df)):
    df_sol.extend(to_label(df.iloc[i]))
df_sol = pd.DataFrame(df_sol, columns=['row_id', 'normal_mild', 'moderate', 'severe', 'sample_weight'])
df_sol = df_sol.sort_values(by=['row_id'])

df_sol.head(50)
df_sol.to_csv('../../input/rsna-2024-lumbar-spine-degenerative-classification/solution.csv', index=False)
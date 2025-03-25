import pandas as pd

splits = {'train': 'splits_20-03_12-37\\train.csv', 'val': 'splits_20-03_12-37\\val.csv', 'test': 'splits_20-03_12-37\\test.csv'}
dfs = []

for split, filepath in splits.items():
    df = pd.read_csv(filepath)
    df['split'] = split
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

df['duration'] = (
    df['duration']
    .str.replace(r"[\[\]']", "", regex=True)  # Remove brackets and quotes
    .astype(float)                           # Convert to float
)

speaker_style_agg = (
    df.groupby(['split', 'spk_id', 'demog', 'speaking_style'])
      .agg(total_duration=('duration', 'sum'))
      .reset_index()
)

final = (
    speaker_style_agg.groupby(['split', 'demog', 'speaking_style'])
    .agg(num_speakers=('spk_id', 'nunique'),
         total_duration=('total_duration', 'sum'))
    .reset_index()
)

pivot = final.pivot(index=['demog', 'speaking_style'], columns='split')
pivot.columns = [f"{col[0]}_{col[1]}" for col in pivot.columns]

for split in ['train', 'val', 'test']:
    pivot[f'total_duration_{split}'] = (pivot[f'total_duration_{split}'] / 60).round()

pivot = pivot.reset_index()

latex = r'''
\begin{table}[h!]
\centering
\caption{Speaker Stats by Demographic, Style, and Data Split}
\begin{tabular}{llccc|ccc}
\toprule
\multirow{2}{*}{Demographic} & \multirow{2}{*}{Speaking Style} & \multicolumn{3}{c|}{Num Speakers} & \multicolumn{3}{c}{Total Duration (s)} \\
 & & Train & Dev & Test & Train & Dev & Test \\
\midrule
'''

# Step 3 — Fill rows manually
for _, row in pivot.iterrows():
    latex += f"{row['demog']} & {row['speaking_style']} & " \
             f"{int(row['num_speakers_train'])} & {int(row['num_speakers_val'])} & {int(row['num_speakers_test'])} & " \
             f"{row['total_duration_train']:.2f} & {row['total_duration_val']:.2f} & {row['total_duration_test']:.2f} \\\\\n"

latex += r'''\bottomrule
\end{tabular}
\end{table}
'''

print(latex)

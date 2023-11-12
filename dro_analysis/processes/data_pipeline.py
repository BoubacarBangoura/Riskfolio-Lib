import os

import pandas as pd
import dro_analysis.paths as paths


class DataPipeline:
    def __init__(self, file: str):
        if file == 'PublicData.xlsx':
            raw_df = pd.read_excel(os.path.join(paths.RAW_DATA, file))
            df = self.clean_publicdata(raw_df)
        elif file == 'PublicData2.xlsx':
            raw_df = pd.read_excel(os.path.join(paths.RAW_DATA, 'PublicData2.xlsx'), sheet_name='W')
            df = self.clean_publicdata(raw_df, row_of_col=3, nb_rows_to_remove=4)
        else:
            df = None
        self.df = df

    @staticmethod
    def clean_publicdata(raw_df, row_of_col=2, nb_rows_to_remove=3):
        df = raw_df.rename(columns=raw_df.iloc[row_of_col])
        df = df.drop([i for i in range(nb_rows_to_remove)])
        df = df.rename(columns={'two': 'new_name'})
        col = df.columns.tolist()
        col[0] = 'date'
        df.columns = col
        df = df.set_index('date')
        df = df[:-1]
        return df

    def save(self, save_name, save_path=paths.CLEAN_DATA):
        save_path = os.path.join(save_path, save_name)
        self.df.to_pickle(save_path)

    def remove_empty(self):
        pass


if __name__ == "__main__":
    dp = DataPipeline('PublicData2.xlsx')
    dp.save('PublicData2')




from settings import DATA_URL, DATA_DIR, FILE_NAME
from io_ import DataDownloader
from io_ import CSVLoader
from model import Meaning, Word, Words

def main():

    downloader = DataDownloader(url=DATA_URL, dir_path=DATA_DIR, file_name=FILE_NAME)
    downloader.download()


    loader = CSVLoader(file_path=downloader.file_path)
    data = loader.load()


    words_experiment = Words()

    for row in data.loc[:, ['word_id', 'word', 'language', 'meaning', 'is_sound_symbolic']].drop_duplicates().itertuples(
            index=False):
        meaning = Meaning.from_string(row.meaning)
        is_sound_symbolic = row.is_sound_symbolic == "yes"

        new_word = Word(
            id_=row.word_id,
            word=row.word,
            language=row.language,
            meaning=meaning,
            is_sound_symbolic=is_sound_symbolic
        )

        words_experiment.add_word(word=new_word)

    words_experiment.plot_vowels_distr(title="Vowels over experiment words")

if __name__ == "__main__":

    main()

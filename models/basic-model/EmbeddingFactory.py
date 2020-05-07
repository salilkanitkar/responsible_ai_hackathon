from pathlib import Path
from typing import List
import zipfile, chakin, logging, sqlite3, re
from tempfile import TemporaryDirectory
import pandas as pd


class EmbeddingFactory:
    """Manages embeddings at scale"""

    def __init__(self, embed_folder: Path, embedding_name: str, embed_dim: int, embed_zip_folder: str = None, **read_csv_kwargs):
        """embed_zip_folder looks for an existing embedding zip file in the folder if provided"""
        self.embed_folder, self.embedding_name = Path(embed_folder), embedding_name
        self.embed_folder.mkdir(exist_ok=True, parents=True)  # create directory if not exists
        self.db = sqlite3.connect(f'file:{embed_folder / "embeddings"}.db', uri=True)
        self.cur = self.db.cursor()

        self._prepare_embeddings(embedding_name, embed_dim, embed_zip_folder, **read_csv_kwargs)

    def __del__(self):
        self.cur.close()
        self.db.close()  # safely close the DB connection when the object is destroyed

    def _tablename(self):
        """Construct a valid SQL table from embedding folder and name"""
        return re.sub(r"[^_0-9a-zA-Z]", "", (self.embed_folder / self.embedding_name).as_posix())

    def embedding_exists(self):
        """Checks if a SQL table exists for embedding"""
        sql = f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{self._tablename()}';"
        return self.cur.execute(sql).fetchone()[0] > 0

    def _prepare_embeddings(self, embedding_name: str, embed_dim: int, embed_zip_folder: str = None, **read_csv_kwargs):
        tablename = self._tablename()

        with TemporaryDirectory() as d:
            if not self.embedding_exists():
                if not embed_zip_folder:
                    logging.info(f"Can't find {embedding_name} locally. Started download.")
                    download_filename = chakin.download(name=embedding_name, save_dir=d)
                else:
                    d = embed_zip_folder  # Use the downloaded folder

                reader = self._parse_embeddings(d, embed_dim, **read_csv_kwargs)
                for chunk_df in reader:
                    # perform any transformations to these rows in memory
                    df = chunk_df.word_vec.str.split(" ", n=1, expand=True).rename({0: "word", 1: "vector_str"}, axis=1)
                    df.to_sql(tablename, self.db, if_exists="append")

                # create an index for faster lookups
                self.cur.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {tablename}_vector_str_idx ON {tablename}(vector_str)")

    def _parse_embeddings(self, embed_dir: Path, embed_dim: int, **read_csv_kwargs):
        embed_zip_path = next(iter(Path(embed_dir).glob('**/*.zip')))
        embed_zip = zipfile.ZipFile(embed_zip_path)
        embed_file_in_zip = next(filter(lambda n: str(embed_dim) in n, embed_zip.namelist()))
        # df = dd.read_csv(ZIP_FILE, header='infer', compression="gzip", skiprows=NUM_ROWS_TO_SKIP_FROM_EMBED_FILE, names=["word_vec"], sep="\t", engine="python", error_bad_lines=False, warn_bad_lines=True)
        reader = pd.read_csv(embed_zip.open(embed_file_in_zip), iterator=True, chunksize=30000, names=["word_vec"], sep="SOME_RANDOM_STR", engine="python",
                             error_bad_lines=False, **read_csv_kwargs)  # Use a separator that doesn't exist to get all data in 1 col
        return reader

    def fetch_word_vectors(self, words: List[str]):
        params = ",".join(["?"] * len(words))
        sql = f"SELECT word, vector_str FROM {self._tablename()} WHERE word IN ({params});"
        r = self.cur.execute(sql, words)
        return r.fetchall()


if __name__ == '__main__':
    embedding_index = EmbeddingFactory(Path("./embeddings/"), "GloVe.6B.100d", 100, nrows=100, skiprows=None)
    r = embedding_index.fetch_word_vectors(['love', 'sky', "mom's"])
    print("Done")

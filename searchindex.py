from pathlib import Path
import numpy as np
import pandas as pd
import torch
import nmslib
from lang_model_utils import load_lm_vocab, Query2Emb
from general_utils import create_nmslib_search_index

input_path = Path('./data/processed_data/')
code2emb_path = Path('./data/code2emb/')
output_path = Path('./data/search')
output_path.mkdir(exist_ok=True)

# read file of urls
url_df = pd.read_csv(input_path/'without_docstrings.lineage', header=None, names=['url'])


#read code
code_df = pd.read_json(input_path/'without_docstrings_original_function.json.gz')
code_df.columns = ['code']

#check for same rows
assert code_df.shape[0] == url_df.shape[0]

#collect these two together into a dataframe
ref_df = pd.concat([url_df, code_df], axis = 1).reset_index(drop=True)
ref_df.head()


nodoc_vecs = np.load(code2emb_path/'nodoc_vecs.npy')
assert nodoc_vecs.shape[0] == ref_df.shape[0]

search_index = create_nmslib_search_index(nodoc_vecs)
search_index.saveIndex('./data/search/search_index.nmslib')

lang_model = torch.load('./data/lang_model/lang_model_cpu_v2.torch', 
                        map_location=lambda storage, loc: storage)

vocab = load_lm_vocab('./data/lang_model/vocab_v2.cls')
q2emb = Query2Emb(lang_model = lang_model.cpu(),
                  vocab = vocab)

search_index = nmslib.init(method='hnsw', space='cosinesimil')
search_index.loadIndex('./data/search/search_index.nmslib')

test = q2emb.emb_mean('Hello World!  This is a test.')
test.shape


class search_engine:
    """Organizes all the necessary elements we need to make a search engine."""
    def __init__(self, 
                 nmslib_index, 
                 ref_df, 
                 query2emb_func):
        """
        Parameters
        ==========
        nmslib_index : nmslib object
            This is pre-computed search index.
        ref_df : pandas.DataFrame
            This dataframe contains meta-data for search results, 
            must contain the columns 'code' and 'url'.
        query2emb_func : callable
            This is a function that takes as input a string and returns a vector
            that is in the same vector space as what is loaded into the search index.

        """
        assert 'url' in ref_df.columns
        assert 'code' in ref_df.columns
        
        self.search_index = nmslib_index
        self.ref_df = ref_df
        self.query2emb_func = query2emb_func
    
    def search(self, str_search, k=2):
        """
        Prints the code that are the nearest neighbors (by cosine distance)
        to the search query.
        
        Parameters
        ==========
        str_search : str
            a search query.  Ex: "read data into pandas dataframe"
        k : int
            the number of nearest neighbors to return.  Defaults to 2.
        
        """
        query = self.query2emb_func(str_search)
        idxs, dists = self.search_index.knnQuery(query, k=k)
        
        for idx, dist in zip(idxs, dists):
            code = self.ref_df.iloc[idx].code
            url = self.ref_df.iloc[idx].url
            print(f'cosine dist:{dist:.4f}  url: {url}\n---------------\n')
            print(code)

se = search_engine(nmslib_index=search_index,
                   ref_df=ref_df,
                   query2emb_func=q2emb.emb_mean)

se.search('read data into pandas dataframe')

#result

#cosine dist:0.1045  url: https://github.com/timvandermeij/sentiment-analysis/blob/master/plot.py#L23


import torch,cv2
from lang_model_utils import lm_vocab, load_lm_vocab, train_lang_model
from general_utils import save_file_pickle, load_file_pickle
import logging
from pathlib import Path
from fastai.text import *

source_path = Path('./data/processed_data/')

with open(source_path/'train.docstring', 'r') as f:
    trn_raw = f.readlines()

with open(source_path/'valid.docstring', 'r') as f:
    val_raw = f.readlines()
    
with open(source_path/'test.docstring', 'r') as f:
    test_raw = f.readlines()

#preprocess data for language model
vocab = lm_vocab(max_vocab=50000,
                 min_freq=10)

# fit the transform on the training data, then transform
trn_flat_idx = vocab.fit_transform_flattened(trn_raw)



# apply transform to validation data
val_flat_idx = vocab.transform_flattened(val_raw)

if not use_cache:
    vocab.save('./data/lang_model/vocab_v2.cls')
    save_file_pickle('./data/lang_model/trn_flat_idx_list.pkl_v2', trn_flat_idx)
    save_file_pickle('./data/lang_model/val_flat_idx_list.pkl_v2', val_flat_idx)


vocab = load_lm_vocab('./data/lang_model/vocab.cls')
trn_flat_idx = load_file_pickle('./data/lang_model/trn_flat_idx_list.pkl')
val_flat_idx = load_file_pickle('./data/lang_model/val_flat_idx_list.pkl')

if not use_cache:
    fastai_learner, lang_model = train_lang_model(model_path = './data/lang_model_weights_v2',
                                                  trn_indexed = trn_flat_idx,
                                                  val_indexed = val_flat_idx,
                                                  vocab_size = vocab.vocab_size,
                                                  lr=3e-3,
                                                  em_sz= 500,
                                                  nh= 500,
                                                  bptt=20,
                                                  cycle_len=1,
                                                  n_cycle=3,
                                                  cycle_mult=2,
                                                  bs = 200,
                                                  wd = 1e-6)
                                                  
                                                  
fastai_learner.fit(1e-3, 3, wds=1e-6, cycle_len=2)
fastai_learner.fit(1e-3, 2, wds=1e-6, cycle_len=3, cycle_mult=2)
fastai_learner.fit(1e-3, 2, wds=1e-6, cycle_len=3, cycle_mult=10)
fastai_learner.save('lang_model_learner_v2.fai')
lang_model_new = fastai_learner.model.eval()
torch.save(lang_model_new, './data/lang_model/lang_model_gpu_v2.torch')
torch.save(lang_model_new.cpu(), './data/lang_model/lang_model_cpu_v2.torch')


from lang_model_utils import load_lm_vocab
vocab = load_lm_vocab('./data/lang_model/vocab_v2.cls')
idx_docs = vocab.transform(trn_raw + val_raw, max_seq_len=30, padding=False)
lang_model = torch.load('./data/lang_model/lang_model_gpu_v2.torch', 
                        map_location=lambda storage, loc: storage)
                        
lang_model.eval()


#the below code extracts embeddings for docstrings one docstring at a time, which is very inefficient. Ideally, you want to extract embeddings in batch but account for the fact that you will have padding, etc. when extracting the hidden states
def list2arr(l):
    "Convert list into pytorch Variable."
    return V(np.expand_dims(np.array(l), -1)).cpu()

def make_prediction_from_list(model, l):
    """
    Encode a list of integers that represent a sequence of tokens.  The
    purpose is to encode a sentence or phrase.

    Parameters
    -----------
    model : fastai language model
    l : list
        list of integers, representing a sequence of tokens that you want to encode

    """
    arr = list2arr(l)# turn list into pytorch Variable with bs=1
    model.reset()  # language model is stateful, so you must reset upon each prediction
    hidden_states = model(arr)[-1][-1] # RNN Hidden Layer output is last output, and only need the last layer

    #return avg-pooling, max-pooling, and last hidden state
    return hidden_states.mean(0), hidden_states.max(0)[0], hidden_states[-1]


def get_embeddings(lm_model, list_list_int):
    """
    Vectorize a list of sequences List[List[int]] using a fast.ai language model.

    Paramters
    ---------
    lm_model : fastai language model
    list_list_int : List[List[int]]
        A list of sequences to encode

    Returns
    -------
    tuple: (avg, mean, last)
        A tuple that returns the average-pooling, max-pooling over time steps as well as the last time step.
    """
    n_rows = len(list_list_int)
    n_dim = lm_model[0].nhid
    avgarr = np.empty((n_rows, n_dim))
    maxarr = np.empty((n_rows, n_dim))
    lastarr = np.empty((n_rows, n_dim))

    for i in tqdm_notebook(range(len(list_list_int))):
        avg_, max_, last_ = make_prediction_from_list(lm_model, list_list_int[i])
        avgarr[i,:] = avg_.data.numpy()
        maxarr[i,:] = max_.data.numpy()
        lastarr[i,:] = last_.data.numpy()

    return avgarr, maxarr, lastarr
    
avg_hs, max_hs, last_hs = get_embeddings(lang_model, idx_docs)

idx_docs_test = vocab.transform(test_raw, max_seq_len=30, padding=False)
avg_hs_test, max_hs_test, last_hs_test = get_embeddings(lang_model, idx_docs_test)


savepath = Path('./data/lang_model_emb/')
np.save(savepath/'avg_emb_dim500_v2.npy', avg_hs)
np.save(savepath/'max_emb_dim500_v2.npy', max_hs)
np.save(savepath/'last_emb_dim500_v2.npy', last_hs)

# save the test set embeddings also
np.save(savepath/'avg_emb_dim500_test_v2.npy', avg_hs_test)
np.save(savepath/'max_emb_dim500_test_v2.npy', max_hs_test)
np.save(savepath/'last_emb_dim500_test_v2.npy', last_hs_test)


loadpath = Path('./data/lang_model_emb/')
avg_emb_dim500 = np.load(loadpath/'avg_emb_dim500_test_v2.npy')

# Build search index (takes about an hour on a p3.8xlarge)
dim500_avg_searchindex = create_nmslib_search_index(avg_emb_dim500)

# save search index
dim500_avg_searchindex.saveIndex('./data/lang_model_emb/dim500_avg_searchindex.nmslib')

dim500_avg_searchindex = nmslib.init(method='hnsw', space='cosinesimil')
dim500_avg_searchindex.loadIndex('./data/lang_model_emb/dim500_avg_searchindex.nmslib')

lang_model = torch.load('./data/lang_model/lang_model_cpu_v2.torch')
vocab = load_lm_vocab('./data/lang_model/vocab_v2.cls')

q2emb = Query2Emb(lang_model = lang_model.cpu(),
                  vocab = vocab)

query = q2emb.emb_mean('Read data into pandas dataframe')
query.shape


class search_engine:
    def __init__(self, 
                 nmslib_index, 
                 ref_data, 
                 query2emb_func):
        
        self.search_index = nmslib_index
        self.data = ref_data
        self.query2emb_func = query2emb_func
    
    def search(self, str_search, k=3):
        query = self.query2emb_func(str_search)
        idxs, dists = self.search_index.knnQuery(query, k=k)
        
        for idx, dist in zip(idxs, dists):
            print(f'cosine dist:{dist:.4f}\n---------------\n', self.data[idx])

se = search_engine(nmslib_index=dim500_avg_searchindex,
                   ref_data = test_raw,
                   query2emb_func = q2emb.emb_mean)


import logging
logging.getLogger().setLevel(logging.ERROR)

se.search('read csv into pandas dataframe')

#result

#cosine dist:0.0977
#---------------
# load csv or json into pandas dataframe


    

import ast
import glob
import re
from pathlib import Path

import astor
import pandas as pd
import spacy
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from general_utils import apply_parallel, flattenlist

EN = spacy.load('en_core_web_md')

df = pd.concat([pd.read_csv(f'https://storage.googleapis.com/kubeflow-examples/code_search/raw_data/00000000000{i}.csv') \
                for i in range(1)])

df['nwo'] = df['repo_path'].apply(lambda r: r.split()[0])
df['path'] = df['repo_path'].apply(lambda r: r.split()[1])
df.drop(columns=['repo_path'], inplace=True)
df = df[['nwo', 'path', 'content']]
df.head()

print(df.shape)



def tokenize_docstring(text):
    #tokenization with spacy
    tokens = EN.tokenizer(text)
    return [token.text.lower() for token in tokens if not token.is_space]


def tokenize_code(text):
    #tokenizing code strings
    return RegexpTokenizer(r'\w+').tokenize(text)


def get_function_docstring_pairs(blob):
    #extract functions and methods from a given code block
    pairs = []
    try:
        module = ast.parse(blob)
        classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
        functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
        for _class in classes:
            functions.extend([node for node in _class.body if isinstance(node, ast.FunctionDef)])

        for f in functions:
            source = astor.to_source(f)
            docstring = ast.get_docstring(f) if ast.get_docstring(f) else ''
            function = source.replace(ast.get_docstring(f, clean=False), '') if docstring else source

            pairs.append((f.name,
                          f.lineno,
                          source,
                          ' '.join(tokenize_code(function)),
                          ' '.join(tokenize_docstring(docstring.split('\n\n')[0]))
                         ))
    except (AssertionError, MemoryError, SyntaxError, UnicodeEncodeError):
        pass
    return pairs


def get_function_docstring_pairs_list(blob_list):
    return [get_function_docstring_pairs(b) for b in blob_list]

pairs = flattenlist(apply_parallel(get_function_docstring_pairs_list, df.content.tolist(), cpu_cores=32))

assert len(pairs) == df.shape[0], f'Row count mismatch. `df` has {df.shape[0]:,} rows; `pairs` has {len(pairs):,} rows.'
df['pairs'] = pairs
df.head()


#flatten the set of pairs
df = df.set_index(['nwo', 'path'])['pairs'].apply(pd.Series).stack()
df = df.reset_index()
df.columns = ['nwo', 'path', '_', 'pair']

df['function_name'] = df['pair'].apply(lambda p: p[0])
df['lineno'] = df['pair'].apply(lambda p: p[1])
df['original_function'] = df['pair'].apply(lambda p: p[2])
df['function_tokens'] = df['pair'].apply(lambda p: p[3])
df['docstring_tokens'] = df['pair'].apply(lambda p: p[4])
df = df[['nwo', 'path', 'function_name', 'lineno', 'original_function', 'function_tokens', 'docstring_tokens']]
df['url'] = df[['nwo', 'path', 'lineno']].apply(lambda x: 'https://github.com/{}/blob/master/{}#L{}'.format(x[0], x[1], x[2]), axis=1)

#and remove duplicates 

before_dedup = len(df)
df = df.drop_duplicates(['original_function', 'function_tokens'])
after_dedup = len(df)

print(f'Removed {before_dedup - after_dedup:,} duplicate rows')


df.head()



def listlen(x):
    if not isinstance(x, list):
        return 0
    return len(x)

# separate functions without docstrings docstrings should be at least 3 words to be valid

with_docstrings = df[df.docstring_tokens.str.split().apply(listlen) >= 3]
without_docstrings = df[df.docstring_tokens.str.split().apply(listlen) < 3]

grouped = with_docstrings.groupby('nwo')

train, test = train_test_split(list(grouped), train_size=0.87, shuffle=True, random_state=8081)
train, valid = train_test_split(train, train_size=0.82, random_state=8081)

train = pd.concat([d for _, d in train]).reset_index(drop=True)
valid = pd.concat([d for _, d in valid]).reset_index(drop=True)
test = pd.concat([d for _, d in test]).reset_index(drop=True)

print(f'train set num rows {train.shape[0]:,}')
print(f'valid set num rows {valid.shape[0]:,}')
print(f'test set num rows {test.shape[0]:,}')
print(f'without docstring rows {without_docstrings.shape[0]:,}')

print("train head:")
train.head()


#

#write the data to the files 
import csv

def write_to(df, filename, path='./data/processed_data/'):
    "Helper function to write processed files to disk."
    out = Path(path)
    out.mkdir(exist_ok=True)
    df.function_tokens.to_csv(out/'{}.function'.format(filename), index=False)
    df.original_function.to_json(out/'{}_original_function.json.gz'.format(filename), orient='values', compression='gzip')
    if filename != 'without_docstrings':
        df.docstring_tokens.to_csv(out/'{}.docstring'.format(filename), index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
    df.url.to_csv(out/'{}.lineage'.format(filename), index=False)

import os
if not os.path.exists('data/'):
    os.makedirs('data/')
# write to output files
write_to(train, 'train')
write_to(valid, 'valid')
write_to(test, 'test')
write_to(without_docstrings, 'without_docstrings')

#

use_cache = False

from pathlib import Path
from general_utils import get_step2_prerequisite_files, read_training_files
from keras.utils import get_file
OUTPUT_PATH = Path('./data/seq2seq/')
OUTPUT_PATH.mkdir(exist_ok=True)

if use_cache:
    get_step2_prerequisite_files(output_directory = './data/processed_data')

# you want to supply the directory where the files are from step 1.
train_code, holdout_code, train_comment, holdout_comment = read_training_files('./data/processed_data/')


#
from ktext.preprocess import processor

if not use_cache:    
    code_proc = processor(heuristic_pct_padding=.7, keep_n=20000)
    t_code = code_proc.fit_transform(train_code)

    comment_proc = processor(append_indicators=True, heuristic_pct_padding=.7, keep_n=14000, padding ='post')
    t_comment = comment_proc.fit_transform(train_comment)

elif use_cache:
    logging.warning('Not fitting transform function because use_cache=True')

import dill as dpickle
import numpy as np

if not use_cache:
    with open(OUTPUT_PATH/'py_code_proc_v2.dpkl', 'wb') as f:
        dpickle.dump(code_proc, f)

    with open(OUTPUT_PATH/'py_comment_proc_v2.dpkl', 'wb') as f:
        dpickle.dump(comment_proc, f)

    #save processed data
    np.save(OUTPUT_PATH/'py_t_code_vecs_v2.npy', t_code)
    np.save(OUTPUT_PATH/'py_t_comment_vecs_v2.npy', t_comment)


from ktext.preprocess import processor

if not use_cache:    
    code_proc = processor(heuristic_pct_padding=.7, keep_n=20000)
    t_code = code_proc.fit_transform(train_code)

    comment_proc = processor(append_indicators=True, heuristic_pct_padding=.7, keep_n=14000, padding ='post')
    t_comment = comment_proc.fit_transform(train_comment)

elif use_cache:
    logging.warning('Not fitting transform function because use_cache=True')


import dill as dpickle
import numpy as np

if not use_cache:
    # Save the preprocessor
    with open(OUTPUT_PATH/'py_code_proc_v2.dpkl', 'wb') as f:
        dpickle.dump(code_proc, f)

    with open(OUTPUT_PATH/'py_comment_proc_v2.dpkl', 'wb') as f:
        dpickle.dump(comment_proc, f)

    # Save the processed data
    np.save(OUTPUT_PATH/'py_t_code_vecs_v2.npy', t_code)
    np.save(OUTPUT_PATH/'py_t_comment_vecs_v2.npy', t_comment)



#


from seq2seq_utils import load_decoder_inputs, load_encoder_inputs, load_text_processor
from seq2seq_utils import build_seq2seq_model

encoder_input_data, encoder_seq_len = load_encoder_inputs(OUTPUT_PATH/'py_t_code_vecs_v2.npy')
decoder_input_data, decoder_target_data = load_decoder_inputs(OUTPUT_PATH/'py_t_comment_vecs_v2.npy')
num_encoder_tokens, enc_pp = load_text_processor(OUTPUT_PATH/'py_code_proc_v2.dpkl')
num_decoder_tokens, dec_pp = load_text_processor(OUTPUT_PATH/'py_comment_proc_v2.dpkl')

#building the model
seq2seq_Model = build_seq2seq_model(word_emb_dim=800,
                                    hidden_state_dim=1000,
                                    encoder_seq_len=encoder_seq_len,
                                    num_encoder_tokens=num_encoder_tokens,
                                    num_decoder_tokens=num_decoder_tokens)

#summarize the model 

seq2seq_Model.summary()

#train the model

from keras.models import Model, load_model
import pandas as pd
import logging

if not use_cache:

    from keras.callbacks import CSVLogger, ModelCheckpoint
    import numpy as np
    from keras import optimizers

    seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=0.00005), loss='sparse_categorical_crossentropy')

    script_name_base = 'py_func_sum_v9_'
    csv_logger = CSVLogger('{:}.log'.format(script_name_base))

    model_checkpoint = ModelCheckpoint('{:}.epoch{{epoch:02d}}-val{{val_loss:.5f}}.hdf5'.format(script_name_base),
                                       save_best_only=True)

    batch_size = 1100
    epochs = 16
    history = seq2seq_Model.fit([encoder_input_data, decoder_input_data], np.expand_dims(decoder_target_data, -1),
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.12, callbacks=[csv_logger, model_checkpoint])
    
#evaluate the model and save it to the disk 

from seq2seq_utils import Seq2Seq_Inference
import pandas as pd

seq2seq_inf = Seq2Seq_Inference(encoder_preprocessor=enc_pp,
                                 decoder_preprocessor=dec_pp,
                                 seq2seq_model=seq2seq_Model)

demo_testdf = pd.DataFrame({'code':holdout_code, 'comment':holdout_comment, 'ref':''})
seq2seq_inf.demo_model_predictions(n=15, df=demo_testdf)

seq2seq_Model.save(OUTPUT_PATH/'code_summary_seq2seq_model.h5')

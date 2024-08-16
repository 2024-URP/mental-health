import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine

if len(sys.argv) != 2:
	sys.exit("Use: python build_graph.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr','psysym'] # 데이터셋
# build corpus
dataset = sys.argv[1]

if dataset not in datasets: # 데이터셋 리스트에 없는 이름은 에러 뜨고 종료 -> 수정 필요
	sys.exit("wrong dataset name")

# Read Word Vectors: 여기서는 단어 벡터가 저장된 파일을 따로 사용했음
# word_vector_file = 'data/glove.6B/glove.6B.300d.txt'
# word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
#_, embd, word_vector_map = loadWord2Vec(word_vector_file) # loadWord2Vec은 util에서 따로 정의한 함수? 클래스?
# word_embeddings_dim = len(embd[0])

word_embeddings_dim = 300 # 워드 임베딩의 차원
word_vector_map = {} # 단어와 임베딩을 매핑하는 딕셔너리일듯

# shulffing
doc_name_list = [] # train test 가리지 않고 모든 데이터의 제목을 다 넣음
doc_train_list = [] # train데이터의 제목만 넣음
doc_test_list = [] # test데이터의 제목만 넣음

f = open('data/' + dataset + '.txt', 'r') # 데이터셋 루트
lines = f.readlines() # 데이터셋(제목데이터셋) 읽어오기
for line in lines:
    doc_name_list.append(line.strip()) # 파일의 모든 행에서 좌우공백 제거해서 리스트에 입력하기
    temp = line.split("\t") # tab 단위로 문자열 자르기
    if temp[1].find('test') != -1: # 두번째 문자열이 test면
        doc_test_list.append(line.strip()) # 제목을 test 리스트에 넣기
    elif temp[1].find('train') != -1: # 두번째 문자열이 train이면
        doc_train_list.append(line.strip()) # 제목을 train 리스트에 넣기
f.close() # 파일 닫기
# print(doc_train_list)
# print(doc_test_list)

doc_content_list = [] # train test 가리지 않고 모든 실제 데이터를 담을 리스트
f = open('data/corpus/' + dataset + '.clean.txt', 'r') # 실제 정제한 데이터셋 파일 불러오기 
lines = f.readlines()
for line in lines:
    doc_content_list.append(line.strip()) # 좌우 공백 삭제해서 리스트에 넣기
f.close()
# print(doc_content_list)

train_ids = [] # train 데이터셋의 아이디(전체 데이터에서의 아이디임)
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name) # 전체 제목 리스트에서의 인덱스를 데이터 제목 아이디로 지정함
    train_ids.append(train_id) 
print(train_ids)
#random.shuffle(train_ids) # 제목의 인덱스 순서를 섞음->미리 데이터셋에서 섞어둠

# partial labeled data
#train_ids = train_ids[:int(0.2 * len(train_ids))]

train_ids_str = '\n'.join(str(index) for index in train_ids) # 섞은 인덱스들을 문자열로 만듦
f = open('data/' + dataset + '.train.index', 'w') # 왜냐면 파일로 만들기 위해...
f.write(train_ids_str) # 파일에 쓰기
f.close()

# test에 대해서도 똑같이 반복
test_ids = [] 
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
print(test_ids)
#random.shuffle(test_ids)->미리 데이터셋에서 섞어둠

test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open('data/' + dataset + '.test.index', 'w')
f.write(test_ids_str)
f.close()

ids = train_ids + test_ids # 두리스트를 합침
print(ids)
print(len(ids))

shuffle_doc_name_list = [] # 데이터 제목 리스트
shuffle_doc_words_list = [] # 데이터 본문 리스트
for id in ids:
    shuffle_doc_name_list.append(doc_name_list[int(id)]) # id번째의 데이터의 제목 가져오기
    shuffle_doc_words_list.append(doc_content_list[int(id)]) # id번째의 데이터 본문 가져오기
shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list) # 또 제목을 모아서 파일로 만들어..
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

f = open('data/' + dataset + '_shuffle.txt', 'w')
f.write(shuffle_doc_name_str)
f.close()

f = open('data/corpus/' + dataset + '_shuffle.txt', 'w')
f.write(shuffle_doc_words_str)
f.close()

# build vocab
word_freq = {} # 단어가 데이터셋에서 등장하는 횟수 누적
word_set = set() # 단어 집합
for doc_words in shuffle_doc_words_list:
    words = doc_words.split() # 띄어쓰기 단위로 단어 분리
    for word in words:
        word_set.add(word) # 집합에 단어 추가
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set) # 단어 집합의 리스트로 변환(리스트 인덱스를 사용하기 위해)
vocab_size = len(vocab) # 단어 집합 내 단어 개수

word_doc_list = {} # 전체 단어 딕셔너리: 각 단어를 키로, 해당 단어가 존재하는 데이터 인덱스의 리스트를 값으로 가짐

for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i] # 본문 데이터
    words = doc_words.split() # 본문 데이터를 띄어쓰기 단위로 분리하여 단어 리스트 생성
    appeared = set() # 한 데이터에서 등장한 단어들을 담는 집합
    for word in words:
        if word in appeared: # 해당 데이터셋에 해당 단어가 존재하면 반복문 넘어감
            continue
        if word in word_doc_list: # 전체단어집합에 해당 단어가 존재하면 
            doc_list = word_doc_list[word] 
            doc_list.append(i)
            word_doc_list[word] = doc_list # 해당 단어가 존재하는 데이터의 인덱스를 리스트에 넣어줌
        else: # 전체단어집합에 해당 단어가 존재하지 않았으면 키도 함께 추가해줘야함
            word_doc_list[word] = [i]
        appeared.add(word) # 해당 데이터에서 봤던 단어로 추가

word_doc_freq = {} # 각 단어가 등장하는 데이터셋 개수 딕셔너리
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {} # 각 단어에 인덱스로 아이디를 부여함
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

vocab_str = '\n'.join(vocab)

f = open('data/corpus/' + dataset + '_vocab.txt', 'w') # 단어집합을 아이디 순서대로 저장한 파일
f.write(vocab_str)
f.close()

'''
Word definitions begin
'''

definitions = []

for word in vocab:
    word = word.strip()
    synsets = wn.synsets(clean_str(word))
    word_defs = []
    for synset in synsets:
        syn_def = synset.definition() 
        word_defs.append(syn_def) # 모든 유의어의 정의를 저장
    word_des = ' '.join(word_defs) # 정의를 하나의 문자열로 만듦
    if word_des == '': # 단어의 정의가 존재하지 않으면 패딩
        word_des = '<PAD>'
    definitions.append(word_des)

string = '\n'.join(definitions)


f = open('data/corpus/' + dataset + '_vocab_def.txt', 'w')
f.write(string)
f.close()

tfidf_vec = TfidfVectorizer(max_features=1000) 
tfidf_matrix = tfidf_vec.fit_transform(definitions) # 정의에 등장하는 단어들로 word를 tf-idf 벡터화
tfidf_matrix_array = tfidf_matrix.toarray()
print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))

word_vectors = []

for i in range(len(vocab)):
    word = vocab[i]
    vector = tfidf_matrix_array[i] # word의 벡터 가져오기
    str_vector = []
    for j in range(len(vector)):
        str_vector.append(str(vector[j]))
    temp = ' '.join(str_vector) # word의 벡터를 문자열로 만듦 '1 2 3 .. 10'
    word_vector = word + ' ' + temp
    word_vectors.append(word_vector) # ['word vec','word vec',...] 형태

string = '\n'.join(word_vectors)

f = open('data/corpus/' + dataset + '_word_vectors.txt', 'w')
f.write(string)
f.close()

word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
_, embd, word_vector_map = loadWord2Vec(word_vector_file)
word_embeddings_dim = len(embd[0]) # word vector의 차원


'''
Word definitions end
'''
'''
# label list
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2]) # 제목 형태가 (데이터 이름?, train/test, 레이블)인듯
label_list = list(label_set) # 레이블 집합을 리스트로 만듦

label_list_str = '\n'.join(label_list)
f = open('data/corpus/' + dataset + '_labels.txt', 'w')
f.write(label_list_str)
f.close()
'''
# x: feature vectors of training docs, no initial features
# slect 90% training set
#train_size = len(train_ids) # train 데이터 개수
#val_size = int(0.1 * train_size) # validation 데이터 개수(가 될 것)
# psysym 데이터셋에 대해
train_size = len(train_ids)
val_size = 885
real_train_size = train_size - val_size  # - int(0.5 * train_size) # 실제 train 데이터로 사용될 데이터 개수
# different training rates

real_train_doc_names = shuffle_doc_name_list[:real_train_size] # train 데이터셋 이름 리스트
real_train_doc_names_str = '\n'.join(real_train_doc_names)

f = open('data/' + dataset + '.real_train.name', 'w')
f.write(real_train_doc_names_str)
f.close()

row_x = [] # x의 행인 건가
col_x = [] # x의 열인 건가
data_x = [] # x의 행은 document의 id를, 모든 열은 각 document의 임베딩? [행렬에 들어가는 실제 값들]
for i in range(real_train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)]) # 단어 임베딩이 될 벡터 초기화
    doc_words = shuffle_doc_words_list[i] # 본문
    words = doc_words.split() # 본문 띄어쓰기 단위로 분리
    doc_len = len(words) # 한 본문의 단어 개수
    for word in words:
        if word in word_vector_map: # 단어 임베딩이 기존에 저장해둔 단어벡터 파일에 이미 존재하면
            word_vector = word_vector_map[word] # 현재 단어의 임베딩
            # print(doc_vec)
            # print(np.array(word_vector))
            doc_vec = doc_vec + np.array(word_vector) # 등장하는 단어들의 임베딩을 더해서 document의 임베딩을 만들어냄

    for j in range(word_embeddings_dim):
        row_x.append(i) # 데이터번호가 되...
        col_x.append(j) 
        # np.random.uniform(-0.25, 0.25)
        data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len # 임베딩의 각 차원의 평균을 내줌..근데 왜 이렇게 어렵게 값을 한개한개 추가해주는거지

# x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    real_train_size, word_embeddings_dim)) # train 데이터 크기만큼 희소행렬로 만들기 

y = []
for i in range(real_train_size): # train 데이터셋의 레이블만 가져옴
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    '''
    one_hot = [0 for l in range(len(label_list))] # 레이블에 따라 원핫벡터 생성
    label_index = label_list.index(label)
    one_hot[label_index] = 1 # 해당 레이블 위치에 1 넣어주기
    y.append(one_hot) # 레이블 배열 넣어주기
    '''
    label_classification = np.array(np.array([*map(int,label.split(','))]))
    y.append(label_classification) # 레이블 배열 넣어주기
y = np.array(y) # 리스트를 넘파이 배열로 만들어주기
print(y)

# tx: feature vectors of test docs, no initial features
test_size = len(test_ids) # 테스트 데이터 수

row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i + train_size] # 학습 데이터 뒤의 데이터들을 테스트 데이터로 사용
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map: 
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

# tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, word_embeddings_dim))

ty = []
for i in range(test_size):
    doc_meta = shuffle_doc_name_list[i + train_size]
    temp = doc_meta.split('\t')
    label = temp[2]
    '''
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    '''
    label_classification = np.array(np.array([*map(int,label.split(','))]))
    ty.append(label_classification) # 레이블 배열 넣어주기
ty = np.array(ty)
print(ty)

# allx: the the feature vectors of both labeled and unlabeled training instances
# (a superset of x)
# unlabeled training instances -> words

word_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, word_embeddings_dim)) # word 임베딩을 랜덤 값으로 초기화

for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector # word 집합의 벡터가 있는 경우 가져옴

# train 데이터의 document와 데이터셋 내 모든 word의 임베딩 정보를 가질 행렬
# validation, train 데이터셋을 모두 합하여 만든 행렬
row_allx = [] 
col_allx = []
data_allx = []

for i in range(train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim): # train 데이터의 document들에 대해서만 임베딩 입력해줌
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))


row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim)) # train, validation 데이터셋을 모두 합친 희소행렬

ally = []
for i in range(train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    #one_hot = [0 for l in range(len(label_list))]
    #label_index = label_list.index(label)
    #one_hot[label_index] = 1
    label_classification = np.array(np.array([*map(int,label.split(','))]))
    ally.append(label_classification) # 레이블 배열 넣어주기

for i in range(vocab_size):
    #one_hot = [0 for l in range(len(label_list))] # 단어 벡터는 그냥 0벡터
    one_hot = np.zeros_like(label_classification)
    ally.append(one_hot)

ally = np.array(ally)

print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

'''
Doc word heterogeneous graph
'''

# word co-occurence with context windows
window_size = 20
windows = []

for doc_words in shuffle_doc_words_list: # 모든 데이터셋의 본문에 대해 반복
    words = doc_words.split() # 본문을 단어로 분할
    length = len(words) # 본문 내 단어 개수
    if length <= window_size: # 본문의 단어 개수가 20개 이하라면
        windows.append(words) # 분할된 단어 리스트를 넣어줌
    else: # 본문의 단어 개수가 20개를 초과한다면
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1): # 초과하는 단어수 만큼 반복
            window = words[j: j + window_size] # 20단어 단위로 window 만듦
            windows.append(window) # window 리스트에 넣어줌
            # print(window)


word_window_freq = {} # 주어진 데이터셋에서 어떤 단어가 등장하는 횟수
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared: # 문서 내 재등장은 카운트하지 않음
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

word_pair_count = {} # window에서 함께 등장하는 단어쌍 id 
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i] # 단어 아이디 매핑 딕셔너리에서 해당 단어의 아이디 찾기
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id: # 같은 단어면 지나가기
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id) # 단어쌍의 id를 문자열로 이어줌
            if word_pair_str in word_pair_count: # 단어쌍 개수로 추가
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id) # 단어쌍 id 순서 바꿔서 문자열로 이어줌
            if word_pair_str in word_pair_count: # 단어쌍 개수로 추가
                word_pair_count[word_pair_str] += 1 
            else:
                word_pair_count[word_pair_str] = 1

row = []
col = []
weight = []

# pmi as weights
# positive pointwise mutual information
# 점별 상호 정보량(Pointwise Mutual Information, PMI)는 두 단어의 동시에 일어나는 관계를 나타내는 지표

num_window = len(windows) # 본문 개수

for key in word_pair_count:
    temp = key.split(',') # 단어쌍 딕셔너리의 key문자열에서 id를 분리해옴
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]] # id가 i인 word의 전체 데이터셋에서의 등장빈도수
    word_freq_j = word_window_freq[vocab[j]] # id가 j인 word의 전체 데이터셋에서의 등장빈도수
    pmi = log((1.0 * count / num_window) / 
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)

# word vector cosine similarity as weights

'''
for i in range(vocab_size):
    for j in range(vocab_size):
        if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
            vector_i = np.array(word_vector_map[vocab[i]])
            vector_j = np.array(word_vector_map[vocab[j]])
            similarity = 1.0 - cosine(vector_i, vector_j)
            if similarity > 0.9:
                print(vocab[i], vocab[j], similarity)
                row.append(train_size + i)
                col.append(train_size + j)
                weight.append(similarity)
'''
# doc word frequency
doc_word_freq = {} # document, word 쌍으로 해당 doc에 해당 word가 등장하는 횟수를 기록하는 딕셔너리

for doc_id in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[doc_id]
    words = doc_words.split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id) # document와 word 쌍
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split() 
    doc_word_set = set()
    for word in words:
        if word in doc_word_set: # 이미 처리 한 단어에 대해서는 넘어감
            continue
        j = word_id_map[word] # 해당 단어의 id
        key = str(i) + ',' + str(j) # document의 id와 word의 id 쌍
        freq = doc_word_freq[key] # 단어가 doc에서 등장한 횟수 가져오기
        if i < train_size: # 학습 데이터셋에 포함되면
            row.append(i) # 해당 인덱스를 row에 넣어줌
        else:
            row.append(i + vocab_size) # vocab 뒤에 넣어줌
        col.append(train_size + j)
        idf = log(1.0 * len(shuffle_doc_words_list) /
                  word_doc_freq[vocab[j]])
        weight.append(freq * idf)
        doc_word_set.add(word)

node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))

# dump objects
f = open("data/ind.{}.x".format(dataset), 'wb')
pkl.dump(x, f)
f.close()

f = open("data/ind.{}.y".format(dataset), 'wb')
pkl.dump(y, f)
f.close()

f = open("data/ind.{}.tx".format(dataset), 'wb')
pkl.dump(tx, f)
f.close()

f = open("data/ind.{}.ty".format(dataset), 'wb')
pkl.dump(ty, f)
f.close()

f = open("data/ind.{}.allx".format(dataset), 'wb')
pkl.dump(allx, f)
f.close()

f = open("data/ind.{}.ally".format(dataset), 'wb')
pkl.dump(ally, f)
f.close()

f = open("data/ind.{}.adj".format(dataset), 'wb')
pkl.dump(adj, f)
f.close()
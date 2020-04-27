import gensim

path ="./word2vec_model/baike_26g_news_13g_novel_229g.bin"
model = gensim.models.KeyedVectors.load_word2vec_format(path, limit=500000,binary=True)
print(model.similarity("西红柿","番茄"))
word = '中国'
if word in model.wv.index2word:
    print(model.most_similar(word))

print(len(model.index2word))
print(model.vectors.shape)
home_index = model.index2word.index('家')
print(model.vectors[home_index])
# import numpy as np
# words_vectors = np.load('./word2vec_model/baike_26g_news_13g_novel_229g.model.wv.vectors.npy')
# print('载入word vetors')
# words_list= words_list.tolist()
# words_list = [word.decode('UTF-8') for word in words_list]
# word_vectors = np.load('')
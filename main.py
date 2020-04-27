import json
import jieba
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.pylab import style
from keras.utils import plot_model
style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import os
from tqdm import tqdm
# import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import Encoder,BahdanauAttention,Decoder
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
#混合训练语料
me_train_dir ='./baiduQA_corpus/me_train.json'
#test集
#一个问题配一段材料，有答案
me_test_ann_dir = './baiduQA_corpus/me_test.ann.json'
#一个问题配多段材料，有或无答案
me_test_ir_dir = './baiduQA_corpus/me_test.ir.json'
#validation集
#一个问题配一段材料，有答案
me_validation_ann_dir = './baiduQA_corpus/me_validation.ann.json '
#一个问题配多段材料，有或无答案
me_validation_ir_dir = './baiduQA_corpus/me_validation.ir.json'
#停用词
stop_word_dir = './stop_words.utf8'
def json_load(s:str):
    return json.load(open(s))
def get_stop_words(path):
    with open(path,encoding='utf8') as f:
        return [l.strip() for l in f]
stop_words = get_stop_words(stop_word_dir) + ['']
def segement(strs,sw = stop_words):
    words = jieba.lcut(strs,cut_all=False)
    # get_TF(words)
    # words = ["<start>"]+words
    # words = words.append("<end>")
    return [x for x in words if x not in sw]
def get_TF(words,topK=20):
    tf_dic = {}
    for w in words:
        tf_dic[w] = tf_dic.get(w,0)+1
    return sorted(tf_dic.items(),key= lambda x:x[1],reverse=True)[:topK]
def preprocess_sentence(s):
    words = segement(s)
    words = ["<start>"] + words + ["<end>"]
    return words
    pass
def lls_to_ls(lls:list):
    '''
    :param lls: List[List[Str]]
    :return: List[Str]
    '''
    ls =[]
    for l in lls:
        ls = ls+l
    # ls = str(lls)
    # ls = ls.replace('[', '')
    # ls = ls.replace(']', '')
    # ls = ls.split(',')
    print(ls)
    return ls
def create_dataset(path,num_examples):
    data = json_load(path)
    # if num_examples ==None:
    #     num_examples = len(data)
    word_pairs = []
    pair = []
    for w in tqdm(list(data.values())[:num_examples]):
        lq=["<start> "]+segement(w['question'])+[" <end>"]
        pair.append(" ".join(lq))

        # print(pair[0])
        for e in w['evidences'].values():
            if e['answer'][0] != 'no_answer':
                # print(e['answer'])
                la= ["<start> "]+segement(e['answer'][0])+[" <end>"]
                pair.append(" ".join(la))
                break
        if e['answer'][0] == 'no_answer':
            la=["<start> "]+e['answer']+[" <end>"]
            pair.append(" ".join(la))
        word_pairs.append(pair)
        # print(pair[1])
        # print(pair)
        pair = []
    # q= [x[0] for x in word_pairs]
    # print(q)
    # print(q[0])
    # print(q[-1])
    # q=lls_to_ls(q)
    # print("question的词频:",str(get_TF(q)))
    return zip(*word_pairs)
    # return [x[0] for x in word_pairs],[x[1] for x in word_pairs]

    # return zip(*zip(*word_pairs))
def create_dataset_txt(path,num_examples):
    import io
    lines = io.open(path, encoding='gbk').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split(' ')] for l in lines[:num_examples]]

    # return [x[0] for x in word_pairs], [x[1] for x in word_pairs]
    return zip(*word_pairs)
# Q,A=create_dataset(me_train_dir,None)
# print(Q[-1])
# print(A[-1])
def max_length(tensor):
    return max(len(t) for t in tensor)
def tokenize(input):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='',split=" ")
    #生成词典
    tokenizer.fit_on_texts(input)
    #文档化作词向量(张量)
    tensor = tokenizer.texts_to_sequences(input)
    #结尾补零
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')
    return tensor, tokenizer
def load_dataset(path,num_examples=None,txt=True):
    if txt:
    # inp, targ = create_dataset(path, num_examples)
        inp, targ = create_dataset_txt(path, num_examples)
    else:
        inp, targ = create_dataset(path, num_examples)
    # inp,targ = list(temp)[0], list(temp)[1]
    input_tensor, inp_tokenizer = tokenize(inp)
    target_tensor, targ_tokenizer = tokenize(targ)
    return input_tensor,target_tensor,inp_tokenizer,targ_tokenizer
# print(me_train['Q_TRN_010878']['evidences']['Q_TRN_010878#05'])

#
# q1= me_train['Q_TRN_010878']['question']
# print(list(me_train.values())[:1])
# print(q1)
# # q1 = preprocess_sentence(q1)
# seg_q1= segement(q1)
# print(seg_q1)
def convert(tokenizer, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, tokenizer.index_word[t]))
#loss
def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)
@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([targ_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

    # 教师强制 - 将目标词作为下一个输入
    for t in range(1, targ.shape[1]):
      # 将编码器输出 （enc_output） 传送至解码器
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += loss_function(targ[:, t], predictions)

      # 使用教师强制
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1])) #批量loss= batch_size个loss求和/batch_size

  variables = encoder.trainable_variables + decoder.trainable_variables #就是 W 、b、U、h、c等参数

  gradients = tape.gradient(loss, variables)#就是对Loss 求上述各个参数的偏导

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss
def training():
    #train
    EPOCHS = 10
    # gpu_options = tf.GPUOptions(allow_growth=True)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    for epoch in range(EPOCHS):
      start = time.time()

      enc_hidden = encoder.initialize_hidden_state()
      total_loss = 0

      for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):#将总数据集拆成steps_per_epoch（2000）个batch
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
      # 每 2 个周期（epoch），保存（检查点）一次模型
      if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

      print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                          total_loss / steps_per_epoch))
      print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    #end train
#翻译
def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    inp = preprocess_sentence(sentence)
    # inp = sentence.split(' ')
    print(inp_tokenizer)

    inputs = []
    for i in inp:
        if i in inp_tokenizer.word_index:
            inputs.append(inp_tokenizer.word_index[i])
        else:
            active_learning(i)
            # print('请问\''+i+'\'是什么意思？')
            # inputs.append('')

    # inputs = [inp_tokenizer.word_index[i] for i in inp]#词向量
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]#先初始化,后面restore后覆盖这个变量
    enc_out, enc_hidden = encoder(inputs, hidden)#这里是调用call方法

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_tokenizer.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # 存储注意力权重以便后面制图
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()#对于文本Gan而言 因为GAN的Discriminator需要一整个句子（具体来说是它们的embedding）作为输入，所以需要用max返回的下标（就相当于argmax操作）得到one-hot，再得到embedding，然后丢给Discriminator。
        if targ_tokenizer.index_word[predicted_id] == '<end>' and t==0:
            temp_predicts = list(predictions[0])
            del temp_predicts[predicted_id]
            predicted_id = tf.argmax(temp_predicts).numpy()

        result += targ_tokenizer.index_word[predicted_id] + ' '

        if targ_tokenizer.index_word[predicted_id] == '<end>' :
            return result, sentence, attention_plot

        # 预测的 ID 被输送回模型
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot
def getYesterday():
    today=datetime.date.today()
    oneday=datetime.timedelta(days=1)
    yesterday=today-oneday
    return yesterday
def Automatic_train():
    filename = 'new_knowledge_'+datetime.datetime.now().strftime('%Y-%m-%d')+'.txt'
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    new_QA_dir= 'new_knowledge_'+getYesterday()+'.txt'
    input_tensor, target_tensor, inp_tokenizer, targ_tokenizer = load_dataset(new_QA_dir)
    # 采用 80 - 20 的比例切分训练集和验证集
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,target_tensor,test_size=0.2)
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)  # BUFFER_SIZE代表shuffle的程度，越大shuffle越强
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)  # drop_remainder=True :数据达不到batch_size时抛弃(v1.10以上)
    # train
    EPOCHS = 10
    # gpu_options = tf.GPUOptions(allow_growth=True)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):  # 将总数据集拆成steps_per_epoch（2000）个batch
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # 每 2 个周期（epoch），保存（检查点）一次模型
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    # end train
def active_learning(s):
    q = '请问\''+s+'\'是什么意思？'
    # print(q)
    answer=input(q)
    if answer=='不知道':
        return
    filename = 'new_knowledge_'+datetime.datetime.now().strftime('%Y-%m-%d')+'.txt'
    with open(filename, 'a') as file_object:
        file_object.write(q+' '+answer+'\n')
    file_object.close()

# 注意力权重制图函数
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))
    # sentence =preprocess_sentence(sentence)
    # result = preprocess_sentence(result)
    # attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    #preprocess_sentence

    attention_plot = attention_plot[:len(preprocess_sentence(result)), :len(preprocess_sentence(sentence))]
    plot_attention(attention_plot, preprocess_sentence(sentence), preprocess_sentence(result))

if __name__ == '__main__':
    # 尝试实验不同大小的数据集
    num_examples = 30000
    QA_dir = './QA_data.txt'
    input_tensor, target_tensor, inp_tokenizer, targ_tokenizer = load_dataset(QA_dir, num_examples)

    # 计算目标张量的最大长度 （max_length）
    max_length_inp, max_length_targ = max_length(input_tensor), max_length(target_tensor)
    print(max_length_inp)
    print(max_length_targ)
    # 采用 80 - 20 的比例切分训练集和验证集
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                    target_tensor,
                                                                                                    test_size=0.2)

    # 显示长度
    print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))
    # 24000 24000 6000 6000
    me_train = json.load(open(me_train_dir))
    print(me_train['Q_TRN_010878'])
    print(me_train['Q_TRN_010878']['question'])

    print("Input ; index to word mapping")
    convert(inp_tokenizer, input_tensor_train[0])
    print()
    print("Target ; index to word mapping")
    convert(targ_tokenizer, target_tensor_train[0])

    # 创建一个 tf.data 数据集
    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = 12
    steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
    embedding_dim = 24
    units = 64
    vocab_inp_size = len(inp_tokenizer.word_index) + 1
    vocab_tar_size = len(targ_tokenizer.word_index) + 1
    # print("相关参数：")
    # print(BATCH_SIZE)
    # print(steps_per_epoch)
    # print(vocab_inp_size)
    # print(vocab_tar_size)
    # # print(vocab_inp_size)
    # print(vocab_tar_size)
    # vocab_tar_size = 5000
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(
        BUFFER_SIZE)  # BUFFER_SIZE代表shuffle的程度，越大shuffle越强
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)  # drop_remainder=True :数据达不到batch_size时抛弃(v1.10以上)
    example_input_batch, example_target_batch = next(iter(dataset))  # (v1.10以上)
    print(example_input_batch.shape, example_target_batch.shape)

    # encoder
    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    plot_model(encoder, to_file='encoder.png', show_shapes=True, show_layer_names=True, rankdir='TB', dpi=900,
               expand_nested=True)
    # 样本输入
    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder.call(example_input_batch, sample_hidden)
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

    # attention
    attention_layer = BahdanauAttention(10)
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
    plot_model(attention_layer, to_file='attention_layer.png', show_shapes=True, show_layer_names=True, rankdir='TB',
               dpi=900, expand_nested=True)

    print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    # decoder
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
    plot_model(decoder, to_file='decoder.png', show_shapes=True, show_layer_names=True, rankdir='TB', dpi=900,
               expand_nested=True)
    sample_decoder_output, _, _ = decoder(tf.random.uniform((12, 1)),
                                          sample_hidden, sample_output)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

    # optimizer
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    # checkpoint & save model as object
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    #训练
    training()
    #end 训练
    #翻译
    # # 恢复检查点目录 （checkpoint_dir） 中最新的检查点
    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    #
    # # translate(u'世界上最早的报纸诞生于')
    # # translate(u'世界上最早的报纸在哪个国家')
    # # translate(u'世界上最早的深水炸弹是哪一年研制成功的?')
    # # translate(u'鸟类中最小的是')
    # # translate(u'世界上最长寿的人叫什么')
    # translate(u'你是傻子吗')
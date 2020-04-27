import json
import jieba
import time
import datetime
import math
import collections
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.pylab import style
from keras.utils import plot_model
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import OneHotEncoder
style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import os
from tqdm import tqdm
# import re
import tensorflow as tf
import tensorflow_addons as tfa
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
class MyQA:
    # 混合训练语料
    me_train_dir = './baiduQA_corpus/me_train.json'
    # test集
    # 一个问题配一段材料，有答案
    me_test_ann_dir = './baiduQA_corpus/me_test.ann.json'
    # 一个问题配多段材料，有或无答案
    me_test_ir_dir = './baiduQA_corpus/me_test.ir.json'
    # validation集
    # 一个问题配一段材料，有答案
    me_validation_ann_dir = './baiduQA_corpus/me_validation.ann.json '
    # 一个问题配多段材料，有或无答案
    me_validation_ir_dir = './baiduQA_corpus/me_validation.ir.json'
    # 停用词


    def __init__(self):

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = './logs/gradient_tape/' + current_time + '/train'
        test_log_dir = './logs/gradient_tape/' + current_time + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        self.m = tf.keras.metrics.SparseCategoricalAccuracy()
        # self.recall = tf.keras.metrics.Recall()
        self.recall = [0]
        # self.F1Score = 2*self.m.result()*self.recall.result()/(self.recall.result()+self.m.result())
        self.BATCH_SIZE = 128
        self.embedding_dim = 24
        self.units = 64
        # 尝试实验不同大小的数据集
        stop_word_dir = './stop_words.utf8'
        self.stop_words = self.get_stop_words(stop_word_dir) + ['']
        num_examples = 30000
        QA_dir = './QA_data.txt'
        # QA_dir = 'C:/Users/Administrator/raw_chat_corpus/qingyun-11w/qinyun-11w.csv'
        self.input_tensor, self.target_tensor, self.inp_tokenizer, self.targ_tokenizer = self.load_dataset(QA_dir,
                                                                                                           num_examples)
        self.num_classes = len(self.targ_tokenizer.index_word)#目标词类别
        #初始化混淆矩阵(训练用和测试用)：
        self.train_confusion_matrix = tfa.metrics.MultiLabelConfusionMatrix(num_classes=self.num_classes)
        self.test_confusion_matrix = tfa.metrics.MultiLabelConfusionMatrix(num_classes=self.num_classes)


        self.F1Score = tfa.metrics.F1Score(num_classes=len(self.targ_tokenizer.index_word), average="micro")
        # self.F1Score = tfa.metrics.F1Score(num_classes=self.max_length_targ, average="micro")
        # input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
        #     self.input_tensor,
        #     self.target_tensor,
        #     test_size=0.2)
        # self.load_split_dataset(input_tensor_train,target_tensor_train)
        self.vocab_inp_size = len(self.inp_tokenizer.word_index) + 1
        self.vocab_tar_size = len(self.targ_tokenizer.word_index) + 1


        # encoder初始化
        self.encoder = Encoder(self.vocab_inp_size, self.embedding_dim, self.units, self.BATCH_SIZE)
        plot_model(self.encoder, to_file='encoder.png', show_shapes=True, show_layer_names=True, rankdir='TB', dpi=900,
                   expand_nested=True)
        # 样本输入
        # sample_hidden = self.encoder.initialize_hidden_state()
        # sample_output, sample_hidden = self.encoder.call(self.example_input_batch, sample_hidden)
        # print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
        # print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

        # attention初始化
        attention_layer = BahdanauAttention(10)
        # attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
        plot_model(attention_layer, to_file='attention_layer.png', show_shapes=True, show_layer_names=True,
                   rankdir='TB',
                   dpi=900, expand_nested=True)

        # print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
        # print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

        # decoder初始化
        self.decoder = Decoder(self.vocab_tar_size, self.embedding_dim, self.units, self.BATCH_SIZE)
        plot_model(self.decoder, to_file='decoder.png', show_shapes=True, show_layer_names=True, rankdir='TB', dpi=900,
                   expand_nested=True)
        # sample_decoder_output, _, _ = self.decoder(tf.random.uniform((self.BATCH_SIZE, 1)),
        #                                       sample_hidden, sample_output)
        #
        # print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

        # optimizer初始化
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        # checkpoint & save model as object 初始化
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                         encoder=self.encoder,
                                         decoder=self.decoder)
    def load_word2vec(self):
        import gensim
        path = "./word2vec_model/baike_26g_news_13g_novel_229g.bin"
        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(path, limit=500000, binary=True)
        print(self.word2vec_model.similarity("西红柿", "番茄"))
    def split_load_Data(self):
        # # 采用 80 - 20 的比例切分训练集和验证集
        # input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
        #     self.input_tensor,
        #     self.target_tensor,
        #     test_size=0.2)
        # input_tensor_train, input_tensor_val, = \
        self.splited_data_group = self.Repeated_KFold(input=self.input_tensor, target=self.target_tensor)
        # print(self.splited_data_group)
        # target_tensor_train, target_tensor_val = self.Repeated_KFold(input=self.target_tensor)
        # self.load_split_dataset(input_tensor_train, target_tensor_train)
    def Repeated_KFold(self,input,target,n_splits=10,n_repeats=1):
        # X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        # y = np.array([1, 2, 3, 4])
        splited_data_group=[]
        kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
        for train_index, test_index in kf.split(input):
            splited_data = {}
            input_train= np.array(list(map(lambda x:input[x,:],train_index)))
            splited_data['input_train'] = input_train
            input_val = np.array(list(map(lambda x:input[x,:],test_index)))
            splited_data['input_val'] = input_val
            target_train = np.array(list(map(lambda x:target[x,:],train_index)))
            splited_data['target_train'] = target_train
            target_val = np.array(list(map(lambda x:target[x,:],test_index)))
            splited_data['target_val'] = target_val
            splited_data_group.append(splited_data)
            # print('train_index', train_index, 'test_index', test_index)
        return splited_data_group
    def load_split_dataset(self,input_tensor,target_tensor):
        # 显示长度
        # print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))
        # 24000 24000 6000 6000
        # me_train = json.load(open(self.me_train_dir))
        # print(me_train['Q_TRN_010878'])
        # print(me_train['Q_TRN_010878']['question'])
        #
        # print("Input ; index to word mapping")
        # self.convert(self.inp_tokenizer, input_tensor_train[0])
        # print()
        # print("Target ; index to word mapping")
        # self.convert(self.targ_tokenizer, target_tensor_train[0])

        # 创建一个 tf.data 数据集
        self.BUFFER_SIZE = len(input_tensor)
        self.steps_per_epoch = len(input_tensor) // self.BATCH_SIZE

        dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(
            self.BUFFER_SIZE)  # BUFFER_SIZE代表shuffle的程度，越大shuffle越强


        dataset = dataset.batch(self.BATCH_SIZE,
                                     drop_remainder=True)  # drop_remainder=True :数据达不到batch_size时抛弃(v1.10以上)
        self.example_input_batch, self.example_target_batch = next(iter(dataset))  # (v1.10以上)
        print(self.example_input_batch.shape, self.example_target_batch.shape)#
        return dataset
        pass
    def json_load(self,s: str):
        return json.load(open(s))

    def get_stop_words(slef,path):
        with open(path, encoding='utf8') as f:
            return [l.strip() for l in f]



    def segement(self,strs):
        sw = self.stop_words
        words = jieba.lcut(strs, cut_all=False)
        # get_TF(words)
        # words = ["<start>"]+words
        # words = words.append("<end>")
        return [x for x in words if x not in sw]

    def get_TF(self,words, topK=20):
        tf_dic = {}
        for w in words:
            tf_dic[w] = tf_dic.get(w, 0) + 1
        return sorted(tf_dic.items(), key=lambda x: x[1], reverse=True)[:topK]

    def preprocess_sentence(self,s):
        words = self.segement(s)
        words = ["<start>"] + words + ["<end>"]
        return words
        pass

    def lls_to_ls(self,lls: list):
        '''
        :param lls: List[List[Str]]
        :return: List[Str]
        '''
        ls = []
        for l in lls:
            ls = ls + l
        # ls = str(lls)
        # ls = ls.replace('[', '')
        # ls = ls.replace(']', '')
        # ls = ls.split(',')
        print(ls)
        return ls

    def create_dataset(self,path, num_examples):
        data = self.json_load(path)
        # if num_examples ==None:
        #     num_examples = len(data)
        word_pairs = []
        pair = []
        for w in tqdm(list(data.values())[:num_examples]):
            lq = ["<start> "] + self.segement(w['question']) + [" <end>"]
            pair.append(" ".join(lq))

            # print(pair[0])
            for e in w['evidences'].values():
                if e['answer'][0] != 'no_answer':
                    # print(e['answer'])
                    la = ["<start> "] + self.segement(e['answer'][0]) + [" <end>"]
                    pair.append(" ".join(la))
                    break
            if e['answer'][0] == 'no_answer':
                la = ["<start> "] + e['answer'] + [" <end>"]
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

    def create_dataset_txt(self,path, num_examples):
        import io
        lines = io.open(path).read().strip().split('\n')

        word_pairs = [[self.preprocess_sentence(w) for w in l.split(' ')] for l in lines[:num_examples]]

        # return [x[0] for x in word_pairs], [x[1] for x in word_pairs]
        return zip(*word_pairs)
    def create_dataset_csv(self,path, num_examples):
        import csv
        # lines = io.open(path, encoding='gbk').read().strip().split('\n')
        with open(path, "r") as csvfile:
            reader = csv.reader(csvfile)

            # 这里不需要readlines


        word_pairs = [[self.preprocess_sentence(w) for w in l.split('|')] for l in reader[:num_examples]]

        # return [x[0] for x in word_pairs], [x[1] for x in word_pairs]
        return zip(*word_pairs)
    # Q,A=create_dataset(me_train_dir,None)
    # print(Q[-1])
    # print(A[-1])
    def max_length(self,tensor):
        result = max(len(t) for t in tensor)
        print('最大长度：')
        return result
    def average_length(self,tensor):
        result = sum(len(np.trim_zeros(t)) for t in tensor) // len(tensor)
        print('平均长度：')
        print(result)
        return result
    def tokenize(self,input):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', split=" ")
        # 生成词典
        tokenizer.fit_on_texts(input)
        # 文档化作词向量(张量)
        tensor = tokenizer.texts_to_sequences(input)
        # 结尾补零

        # tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
        #                                                        padding='post',
        #                                                        truncating='post',
        #                                                        value=0)
        return tensor, tokenizer

    def load_dataset(self,path, num_examples=None, txt=True):
        if txt:
            # inp, targ = create_dataset(path, num_examples)
            inp, targ= self.create_dataset_txt(path, num_examples)
        else:
            inp, targ = self.create_dataset(path, num_examples)
        # inp,targ = list(temp)[0], list(temp)[1]
        input_tensor, inp_tokenizer = self.tokenize(inp)
        target_tensor, targ_tokenizer = self.tokenize(targ)
        # 计算目标张量的最大长度 （max_length）
        self.max_length_inp, self.max_length_targ = self.average_length(input_tensor), self.average_length(
            target_tensor)
        input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,maxlen = self.max_length_inp,padding= 'post',truncating = 'post',value=0)
        target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,maxlen = self.max_length_targ,padding= 'post',truncating = 'post',value=0)

        print(self.max_length_inp)
        print(self.max_length_targ)
        return input_tensor, target_tensor, inp_tokenizer, targ_tokenizer

    # print(me_train['Q_TRN_010878']['evidences']['Q_TRN_010878#05'])

    #
    # q1= me_train['Q_TRN_010878']['question']
    # print(list(me_train.values())[:1])
    # print(q1)
    # # q1 = preprocess_sentence(q1)
    # seg_q1= segement(q1)
    # print(seg_q1)
    def convert(self,tokenizer, tensor):
        for t in tensor:
            if t != 0:
                print("%d ----> %s" % (t, tokenizer.index_word[t]))

    # loss
    def loss_function(self,real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)
        # return tf.nn.softmax_cross_entropy_with_logits(loss_)

    @tf.function
    def train_step(self,inp, targ, enc_hidden):
        loss = 0
        accruacy = 0
        recall = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([self.targ_tokenizer.word_index['<start>']] * self.BATCH_SIZE, 1)

            # 教师强制 - 将目标词作为下一个输入
            for t in range(1, targ.shape[1]):
                actual = targ[:, t]
                end = tf.constant([0])#掩码掩去 padding 0的影响
                mask = tf.math.not_equal(actual, end)
                reduced_actual = tf.boolean_mask(actual,mask)
                # self.m.reset_states()
                # 将编码器输出 （enc_output） 传送至解码器
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                predicted_id = tf.argmax(predictions, 1)
                reduced_predictions = tf.boolean_mask(predictions,mask)
                reduced_predicted_id = tf.argmax(reduced_predictions, 1)
                actual = tf.cast(actual, tf.float32)

                predicted_id = tf.cast(predicted_id, tf.float32)

                loss += self.loss_function(actual, predictions)


                reduced_actual_onehot = tf.one_hot(reduced_actual,self.num_classes)
                # reduced_actual_onehot = tf.cast(reduced_actual_onehot,tf.float32)
                reduced_predicted_id_onehot =  tf.one_hot(reduced_predicted_id,self.num_classes)
                # reduced_predicted_id_onehot = tf.cast(reduced_predicted_id_onehot,tf.float32)
                self.train_confusion_matrix.update_state(reduced_actual_onehot,reduced_predicted_id_onehot)
                _ = self.m.update_state(reduced_actual, reduced_predictions)
                reduced_actual = tf.cast(reduced_actual,tf.float32)
                reduced_predicted_id = tf.cast(reduced_predicted_id,tf.float32)


                # 使用教师强制:
                # dec_input = tf.expand_dims(targ[:, t], 1)
                dec_input = tf.expand_dims(reduced_predicted_id, 1)
                # #不使用教师强制:
                # dec_input = tf.expand_dims(predictions[:,t], 1)
        # recall = 1
        batch_accuracy = self.m.result()#13组 12 个词的平均准确度
        batch_loss = (loss / int(targ.shape[1]))  # 批量loss= batch_size个loss求和/batch_size
        # batch_pres, batch_recalls, batch_f1s = self.cal_precision_recall_F1Score(self.train_confusion_matrix.result().numpy())
        # self.train_confusion_matrix.reset_states()

        variables = self.encoder.trainable_variables +  self.decoder.trainable_variables  # 就是 W 、b、U、h、c等参数
        gradients = tape.gradient(loss, variables)  # 就是对Loss 求上述各个参数的偏导
        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss,batch_accuracy

    def training(self,inp_train,tar_train,inp_val,tar_val):
        # train
        self.train_dataset = self.load_split_dataset(inp_train, tar_train)
        # self.val_dataset = self.load_split_dataset(inp_val,tar_val)
        EPOCHS = 20
        # gpu_options = tf.GPUOptions(allow_growth=True)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        for epoch in range(EPOCHS):
            start = time.time()
            self.m.reset_states()
            self.recall = []
            # self.recall.reset_states()
            self.F1Score.reset_states()
            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0
            total_accuracy = 0
            total_pres = 0
            total_recalls = 0
            total_f1s = 0

            test_total_loss = 0
            test_total_accuracy = 0
            test_total_pres = 0
            test_total_recalls = 0
            test_total_f1s = 0
            for (batch, (inp, targ)) in enumerate(self.train_dataset.take(self.steps_per_epoch)):  # 将总数据集拆成steps_per_epoch（2000）个batch
                batch_loss, batch_accuracy = self.train_step(inp, targ, enc_hidden)
                # batch_pres, batch_recalls, batch_f1s = self.cal_precision_recall_F1Score(self.train_confusion_matrix.result().numpy())
                # self.train_confusion_matrix.reset_states()
                total_loss += batch_loss
                total_accuracy += batch_accuracy
                # total_pres += batch_pres
                # total_recalls += batch_recalls
                # total_f1s += batch_f1s
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', total_loss / self.steps_per_epoch, step=epoch)
                    tf.summary.scalar('accuracy', total_accuracy / self.steps_per_epoch, step=epoch)
                #     tf.summary.scalar('Precision', total_pres / self.steps_per_epoch, step=epoch)
                #     tf.summary.scalar('Recall', total_recalls / self.steps_per_epoch, step=epoch)
                #     tf.summary.scalar('F1Score', total_f1s / self.steps_per_epoch, step=epoch)
            # for (batch, (inp, targ)) in enumerate(self.val_dataset.take(self.steps_per_epoch)):
            #     test_batch_loss, test_batch_accuracy, test_batch_pres, test_batch_recalls, test_batch_f1s = self.evaluate(inp_vec=inp, targ_vec=targ)
            #     test_total_loss += batch_loss
            #     test_total_accuracy += batch_accuracy
            #     test_total_pres += batch_pres
            #     test_total_recalls += batch_recalls
            #     test_total_f1s += batch_f1s
            if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} & Accuracy {:.4f}'.format(epoch + 1,
                          batch,
                          batch_loss.numpy(),
                          batch_accuracy.numpy()))
                    # print('Test Epoch {} Batch {} Loss {:.4f} & Accuracy {:.4f} & Precision {:.4f} & Recall {:.4f} & F1Score {:.4f}'.format(
                    #         epoch + 1,
                    #         batch,
                    #         test_batch_loss.numpy(),
                    #         test_batch_accuracy.numpy(),
                    #         test_batch_pres,
                    #         test_batch_recalls,
                    #         test_batch_f1s))
            # 每 2 个周期（epoch），保存（检查点）一次模型
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Epoch {} Loss {:.4f} & Accuracy{:.4f}'.format(epoch + 1,
                  total_loss / self.steps_per_epoch,
                  total_accuracy / self.steps_per_epoch))

            # print('Test Epoch {} Loss {:.4f} & Accuracy{:.4f} & F1Score{:.4f}'.format(epoch + 1,
            #       test_total_loss / self.steps_per_epoch,
            #       test_total_accuracy / self.steps_per_epoch,
            #       test_total_pres / self.steps_per_epoch,
            #       test_total_recalls / self.steps_per_epoch,
            #       test_total_f1s / self.steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
            #重置指标
            self.m.reset_states()
            # print(self.train_confusion_matrix.result().numpy())
            pres,recalls,f1s = self.cal_precision_recall_F1Score(self.train_confusion_matrix.result().numpy())
            self.train_confusion_matrix.reset_states()
            with self.train_summary_writer.as_default():
                tf.summary.scalar('Precision', pres, step=epoch)
                tf.summary.scalar('Recall', recalls, step=epoch)
                tf.summary.scalar('F1Score', f1s, step=epoch)
            print('Epoch {} precision {:.4f} & recall{:.4f} & F1Score{:.4f}'.format(epoch + 1, pres, recalls, f1s))
        # end train

    # 翻译

    # @tf.function
    def evaluate(self,sentence='',inp_vec=[],targ_vec=[]):
        attention_plot = np.zeros((self.max_length_targ, self.max_length_inp))
        if inp_vec==[]:
            inp = self.preprocess_sentence(sentence)

            # inp = sentence.split(' ')
            print(self.inp_tokenizer)

            inputs = []
            for i in inp:
                if i in self.inp_tokenizer.word_index:
                    inputs.append(self.inp_tokenizer.word_index[i])
                else:
                    self.active_learning(i)
                    # print('请问\''+i+'\'是什么意思？')
                    # inputs.append('')
        else:
            inputs = inp_vec
        # inputs = [inp_tokenizer.word_index[i] for i in inp]#词向量
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                               maxlen=self.max_length_inp,
                                                               padding='post',
                                                               value=0)
        inputs = tf.convert_to_tensor(inputs)
        result_ids=[1]
        result_predictions=[]
        result = ''

        hidden = [tf.zeros((1, self.units))]  # 先初始化,后面restore后覆盖这个变量
        enc_out, enc_hidden = self.encoder(inputs, hidden)  # 这里是调用call方法
        loss = 0
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.targ_tokenizer.word_index['<start>']], 0)
        for t in range(1,self.max_length_targ):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                 dec_hidden,
                                                                 enc_out)
            result_predictions.append(predictions)
            # result_predictions.append(predictions)
            # 存储注意力权重以便后面制图
            attention_weights = tf.reshape(attention_weights, (-1,))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[
                                         0]).numpy()  # 对于文本Gan而言 因为GAN的Discriminator需要一整个句子（具体来说是它们的embedding）作为输入，所以需要用max返回的下标（就相当于argmax操作）得到one-hot，再得到embedding，然后丢给Discriminator。
            #以下是通过target_vec 计算指标
            if targ_vec!=[]:
                end = tf.constant([0])  # 掩码掩去 padding 0的影响
                mask = tf.math.not_equal([targ_vec[t]], end)
                actual = tf.cast([targ_vec[t]], tf.float32)
                reduced_actual = tf.boolean_mask(actual, mask)
                mask2 = tf.math.not_equal([predicted_id], end)
                predicted_idx = tf.cast([predicted_id], tf.float32)
                reduced_predictions = tf.boolean_mask(predictions,mask2)
                reduced_predicted_idx = tf.boolean_mask(predicted_idx,mask2)
                _ = self.m.update_state(reduced_actual,reduced_predictions)
                _ = self.F1Score.update_state(reduced_actual, reduced_predicted_idx)

                reduced_actual = tf.cast(reduced_actual, tf.int32)
                reduced_predicted_idx = tf.cast(reduced_predicted_idx, tf.int32)
                # loss += self.loss_function(actual, predictions)
                reduced_actual_onehot = tf.one_hot(reduced_actual, self.num_classes)
                # reduced_actual_onehot = tf.cast(reduced_actual_onehot,tf.float32)
                reduced_predicted_id_onehot = tf.one_hot(reduced_predicted_idx, self.num_classes)
                # reduced_predicted_id_onehot = tf.cast(reduced_predicted_id_onehot,tf.float32)



                self.test_confusion_matrix.update_state(reduced_actual_onehot, reduced_predicted_id_onehot)
                # n = len(self.targ_tokenizer.index_word)
                # self.recall = [0] * n
                # self.precision = [0] * n
                # for k in range(n):
                #     re = tf.keras.metrics.Recall()
                #     prec = tf.keras.metrics.Precision()
                #
                #     y_true = tf.equal(reduced_actual, k)
                #     y_pred = tf.equal(reduced_predicted_idx, k)
                #     re.update_state(y_true, y_pred)
                #     prec.update_state(y_true, y_pred)
                #     self.recall[k] = re.result().numpy()
                #     self.precision[k] = prec.result().numpy()

            # if self.targ_tokenizer.index_word[predicted_id] == '<end>' and t == 1:
            #     temp_predicts = list(predictions[0])
            #     del temp_predicts[predicted_id]
            #     predicted_id = tf.argmax(temp_predicts).numpy()
            #     result += self.targ_tokenizer.index_word[predicted_id] + ' '
            #     result_ids.append(predicted_id)

            if self.targ_tokenizer.index_word[predicted_id] == '<end>':
                result_ids.append(2)
                result_ids = tf.keras.preprocessing.sequence.pad_sequences([result_ids],
                                                                           maxlen=self.max_length_targ,
                                                                           padding='post',
                                                                           value = 0)
                return result, sentence, attention_plot,result_ids,result_predictions
            else:
                result += self.targ_tokenizer.index_word[predicted_id] + ' '
                result_ids.append(predicted_id)
                # result_predictions.append(predictions)
            # 预测的 ID 被输送回模型
            dec_input = tf.expand_dims([predicted_id], 0)

        result_ids = tf.keras.preprocessing.sequence.pad_sequences([result_ids],
                                                                   maxlen=self.max_length_targ,
                                                                   padding='post',
                                                                   value=0)

        # batch_accuracy = self.m.result()  # 13组 12 个词的平均准确度
        # batch_loss = loss   # 批量loss= batch_size个loss求和/batch_size
        # batch_pres, batch_recalls, batch_f1s = self.cal_precision_recall_F1Score(self.train_confusion_matrix.result().numpy())
        return result, sentence, attention_plot,result_ids,result_predictions,


    def getYesterday(self):
        today = datetime.date.today()
        oneday = datetime.timedelta(days=1)
        yesterday = today - oneday
        return yesterday

    def Automatic_train(self):
        filename = 'new_knowledge_' + datetime.datetime.now().strftime('%Y-%m-%d') + '.txt'
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

        new_QA_dir = './new_knowledge_' + self.getYesterday().strftime('%Y-%m-%d') + '.txt'

        input_tensor, target_tensor, inp_tokenizer, targ_tokenizer = self.load_dataset(new_QA_dir)
        new_max_length_inp,new_max_length_tar = self.max_length(input_tensor),self.max_length(target_tensor)
        self.max_length_inp,self.max_length_targ = max(new_max_length_inp,self.max_length_inp),max(new_max_length_tar,self.max_length_targ)
        # new_max_length = self.max_length()
        # 采用 80 - 20 的比例切分训练集和验证集
        input_tensor_train, target_tensor_train= input_tensor,target_tensor
        dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(
            self.BUFFER_SIZE)  # BUFFER_SIZE代表shuffle的程度，越大shuffle越强
        dataset = dataset.batch(self.BATCH_SIZE, drop_remainder=True)  # drop_remainder=True :数据达不到batch_size时抛弃(v1.10以上)
        # train
        EPOCHS = 5
        # gpu_options = tf.GPUOptions(allow_growth=True)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        for epoch in range(EPOCHS):
            start = time.time()

            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(dataset.take(self.steps_per_epoch)):  # 将总数据集拆成steps_per_epoch（2000）个batch
                batch_loss = self.train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.numpy()))
            # 每 2 个周期（epoch），保存（检查点）一次模型
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / self.steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        # end train

    def active_learning(self,s):
        q = '请问\'' + s + '\'是什么意思？'
        # print(q)
        answer = input(q)
        if answer == '不知道':
            return
        filename = 'new_knowledge_' + datetime.datetime.now().strftime('%Y-%m-%d') + '.txt'
        with open(filename, 'a') as file_object:
            file_object.write(q + ' ' + answer + '\n')
        file_object.close()

    # 注意力权重制图函数
    def plot_attention(self,attention, sentence, predicted_sentence):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(attention, cmap='viridis')

        fontdict = {'fontsize': 14}

        ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
        ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()

    def translate(self,sentence):
        result, sentence, attention_plot,result_ids,result_predictions = self.evaluate(sentence)

        print('Input: %s' % (sentence))
        print('Predicted translation: {}'.format(result))
        # sentence =preprocess_sentence(sentence)
        # result = preprocess_sentence(result)
        # attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
        # preprocess_sentence

        attention_plot = attention_plot[:len(self.preprocess_sentence(result)), :len(self.preprocess_sentence(sentence))]
        self.plot_attention(attention_plot, self.preprocess_sentence(sentence), self.preprocess_sentence(result))

    def bleu(self,pred_tokens, label_tokens, k):
        len_pred, len_label = len(pred_tokens), len(label_tokens)
        score = math.exp(min(0, 1 - len_label / len_pred))
        for n in range(1, k + 1):
            num_matches, label_subs = 0, collections.defaultdict(int)
            for i in range(len_label - n + 1):
                pass
                label_subs[''.join(list(map(lambda x:self.targ_tokenizer.index_word[x],label_tokens[i: i + n])))] += 1
            for i in range(len_pred - n + 1):
                if label_subs[''.join(list(map(lambda x:self.targ_tokenizer.index_word[x],pred_tokens[i: i + n])))] > 0:
                    num_matches += 1
                    label_subs[''.join(list(map(lambda x:self.targ_tokenizer.index_word[x],pred_tokens[i: i + n])))] -= 1
            score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
        return score

    def cal_precision_recall_F1Score(self,cf_matrixes):
        num_classes = len(cf_matrixes)
        file_name = "wri2te_test.txt"
        # 以写入的方式打开
        f = open(file_name, 'wb')
        # 写入内容
        total_precision=0
        total_recall = 0
        total_F1Score =0

        num_precison=0
        num_recall=0
        num_f1score=0
        # np.array(cf_matrixes).tofile("result.txt")
        for i in range(num_classes):

            # f.write(np.array(cf_matrixes[i]).tostring().)

            # f.write('\n')
            precision = 0
            recall=0
            f1score =0
            # if( cf_matrixes[i, 1, 1] == 0 and cf_matrixes[i,0,1] ==0 and  )

            if (cf_matrixes[i, 1, 1] + cf_matrixes[i, 0, 1]) == 0:
                pass
            else:
                num_precison+=1
                precision = cf_matrixes[i,1,1]/(cf_matrixes[i,1,1]+cf_matrixes[i,0,1])
                total_precision += precision

            if (cf_matrixes[i, 1, 1] + cf_matrixes[i, 1, 0]) == 0:
                pass
            else:
                num_recall+=1
                recall = cf_matrixes[i,1,1]/(cf_matrixes[i,1,1]+cf_matrixes[i,1,0])
                total_recall += recall

            if (2*cf_matrixes[i, 1, 1] + cf_matrixes[i, 0, 1]+cf_matrixes[i, 1, 0]) == 0:
                pass
            else:
                num_f1score+=1
                f1score = 2*cf_matrixes[i, 1, 1]/(2*cf_matrixes[i, 1, 1]+cf_matrixes[i, 0, 1]+cf_matrixes[i, 1, 0])
                total_F1Score += f1score
        # print(total_precision,total_recall,total_F1Score)
        average_precision = 0
        average_recall = 0
        average_f1score = 0
        if num_precison != 0:
            average_precision = total_precision/num_precison
        if num_recall != 0:
            average_recall = total_recall/num_recall
        if num_f1score!=0:
            average_f1score = total_F1Score/num_f1score
        return average_precision,average_recall,average_f1score
    # def score(self,input_seq, label_seq, k):
    #     pred_tokens = self.translate(input_seq)
    #     label_tokens = label_seq.split(' ')
    #     print('bleu %.3f, predict: %s' % (self.bleu(pred_tokens, label_tokens, k),
    #                                       ' '.join(pred_tokens)))
if __name__ == '__main__':
    mq =MyQA()
    # mq.split_load_Data()
    mq.checkpoint.restore(tf.train.latest_checkpoint(mq.checkpoint_dir))#验证时开启
    i=0
    # for data in mq.splited_data_group:
    #     input_tensor_train=data['input_train']
    #     input_tensor_val=data['input_val']
    #     target_tensor_train=data['target_train']
    #     target_tensor_val=data['target_val']
    #     print('------------------以下是训练集'+str(i))
    #     mq.encoder.dropout_keep_prob = 0.5
    #     mq.training(input_tensor_train,target_tensor_train,input_tensor_val,target_tensor_val)
    #     print('------------------以下是验证集'+str(i))
    #     mq.checkpoint.restore(tf.train.latest_checkpoint(mq.checkpoint_dir))#训练要开启
    #     mq.encoder.dropout_keep_prob = 1
    #     test_out = []
    #     test_accuracy = []
    #     j =0
    #     mq.m.reset_states()
    #     # mq.recall.reset_states()
    #     mq.F1Score.reset_states()
    #     # mq.precision.reset_states()
    #     for inp in input_tensor_val:
    #         result, sentence, attention_plot,result_ids,result_predictions= mq.evaluate(inp_vec=inp,targ_vec=target_tensor_val[j])
    #         # mq.train_confusion_matrix.reset_states()
    #         end = tf.constant([0])  # 掩码掩去 padding 0的影响
    #         mask = tf.math.not_equal(target_tensor_val[j], end)
    #         reduced_actual = tf.boolean_mask(target_tensor_val[j], mask)
    #         mask2 = tf.math.not_equal(result_ids, end)
    #         reduced_result_ids = tf.boolean_mask(result_ids, mask2)
    #         vector1 = np.array(reduced_result_ids)
    #         vector2 = np.array(reduced_actual)
    #
    #         # vector1 = tf.cast(result_ids, tf.float32)
    #         #
    #         # vector2 = tf.cast(target_tensor_val[j], tf.float32)
    #         # vector1 = np.squeeze(vector1)[1:]
    #         # vector2 = np.squeeze(vector2)[1:]
    #
    #         similarity = mq.bleu(vector1, vector2, 2)
    #         # _ = mq.m.update_state(vector2[:len(result_predictions)], result_predictions)
    #         # _ = mq.F1Score.update_state(vector2, vector1)
    #
    #         # similarity = np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    #         test_accuracy.append(mq.m.result().numpy())
    #         # test_precision.append(tf.reduce_mean(mq.precision).numpy())
    #         # test_recall.append(tf.reduce_mean(mq.recall).numpy())
    #         test_out.append(similarity)
    #         j+=1
    #
    #     pres, recalls, f1s = mq.cal_precision_recall_F1Score(mq.test_confusion_matrix.result().numpy())
    #     bleu = sum(test_out) / len(target_tensor_val)
    #     accuracy = sum(test_accuracy) / len(test_accuracy)
    #     with mq.test_summary_writer.as_default():
    #         tf.summary.scalar('Precision', pres, step=i)
    #         tf.summary.scalar('Recall', recalls, step=i)
    #         tf.summary.scalar('F1Score', f1s, step=i)
    #         tf.summary.scalar('Accuracy', accuracy, step=i)
    #     print('test Epoch {} precision {:.4f} & recall{:.4f} & F1Score{:.4f}'.format(i + 1, pres, recalls, f1s))
    #     mq.test_confusion_matrix.reset_states()
    #     # test_out = np.array(test_out)
    #
    #
    #     # recall = sum(test_recall) / len(test_recall)
    #     # precision = sum(test_precision) / len(test_precision)
    #     print('验证集'+str(i)+'bleu:'+str(bleu)+' accuracy:'+str(accuracy))
    #     i += 1
    #     #一次训练
    #     # if i==1:
    #     #     break


    # mq.checkpoint.restore(tf.train.latest_checkpoint(mq.checkpoint_dir))
    mq.translate(u'世界上最早的报纸诞生于')
    mq.translate(u'季羡林的关门弟子是谁')
    mq.translate(u'目前在世界范围内仍使用的历法又叫：')
    mq.translate(u'商丘属于哪里')
    mq.translate(u'柳宗元和韩愈一起倡导什么运动')
    mq.translate(u'柳岩是哪里人？')
    while(True):
        q = input()
        if q == '!':
            break
        mq.translate(q)

    # mq.Automatic_train()

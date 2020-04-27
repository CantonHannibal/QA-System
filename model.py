'''编写编码器 （encoder） 和解码器 （decoder） 模型
实现一个基于注意力的编码器 - 解码器模型。关于这种模型，你可以阅读 TensorFlow 的 神经机器翻译 (序列到序列) 教程。本示例采用一组更新的 API。此笔记本实现了上述序列到序列教程中的 注意力方程式。下图显示了注意力机制为每个输入单词分配一个权重，然后解码器将这个权重用于预测句子中的下一个单词。下图和公式是 Luong 的论文中注意力机制的一个例子。

attention mechanism

输入经过编码器模型，编码器模型为我们提供形状为 (批大小，最大长度，隐藏层大小) 的编码器输出和形状为 (批大小，隐藏层大小) 的编码器隐藏层状态。

下面是所实现的方程式：

attention equation 0 attention equation 1

本教程的编码器采用 Bahdanau 注意力。在用简化形式编写之前，让我们先决定符号：

FC = 完全连接（密集）层
EO = 编码器输出
H = 隐藏层状态
X = 解码器输入
以及伪代码：

score = FC(tanh(FC(EO) + FC(H)))
attention weights = softmax(score, axis = 1)。 Softmax 默认被应用于最后一个轴，但是这里我们想将它应用于 第一个轴, 因为分数 （score） 的形状是 (批大小，最大长度，隐藏层大小)。最大长度 （max_length） 是我们的输入的长度。因为我们想为每个输入分配一个权重，所以 softmax 应该用在这个轴上。
context vector = sum(attention weights * EO, axis = 1)。选择第一个轴的原因同上。
embedding output = 解码器输入 X 通过一个嵌入层。
merged vector = concat(embedding output, context vector)
此合并后的向量随后被传送到 GRU
每个步骤中所有向量的形状已在代码的注释中阐明：'''
import tensorflow as tf
import gensim
path = "./word2vec_model/baike_26g_news_13g_novel_229g.bin"
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(path, limit=500000, binary=True)
print(word2vec_model.similarity("西红柿", "番茄"))
# from keras.utils import plot_model
# from tensorflow.keras.utils import plot_model
class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) #121 24
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,#是返回输出序列还是返回最后一个输出
                                   return_state=True,   #是否返回最后一个输出前的状态
                                   recurrent_initializer='glorot_uniform')#Glorot均匀分布初始化方法，又成Xavier均匀初始化，参数从[-limit, limit]的均匀分布产生，其中limit为sqrt(6 / (fan_in + fan_out))。fan_in为权值张量的输入单元数，fan_out是权重张量的输出单元数。
    # tf.keras.utils.plot_model(self.embedding, to_file='./encoder/embedding.png')
    # tf.keras.utils.plot_model(self.gru, to_file='./encoder/gru.png')

  def call(self, x, hidden):
    # print('bbbbb')
    # print(x.shape)#12 ,121
    x = self.embedding(x)
    # x = tf.nn.embedding_lookup(word2vec_model.vectors,x)#500000 128 12 121 ->12 * 121 *128
    # print(x.shape)#12 121 24
    output, state = self.gru(x, initial_state = hidden)#返回输出序列和最后一个状态Ht

    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))



class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # 隐藏层的形状 == （批大小，隐藏层大小）
    # hidden_with_time_axis 的形状 == （批大小，1，隐藏层大小）
    # 这样做是为了执行加法以计算分数
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # 分数的形状 == （批大小，最大长度，1）
    # 我们在最后一个轴上得到 1， 因为我们把分数应用于 self.V
    # 在应用 self.V 之前，张量的形状是（批大小，最大长度，单位）
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # 注意力权重 （attention_weights） 的形状 == （批大小，最大长度，1）
    attention_weights = tf.nn.softmax(score, axis=1)

    # 上下文向量 （context_vector） 求和之后的形状 == （批大小，隐藏层大小）
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # 用于注意力
    self.attention = BahdanauAttention(self.dec_units)
    # tf.keras.utils.plot_model(self.embedding, to_file='./decoder/embedding.png')
    # tf.keras.utils.plot_model(self.gru, to_file='./decoder/gru.png')
  def call(self, x, hidden, enc_output):
    # 编码器输出 （enc_output） 的形状 == （批大小，最大长度，隐藏层大小）
    context_vector, attention_weights = self.attention(hidden, enc_output)
    # print('ccc')
    # print(x.shape)
    # x 在通过嵌入层后的形状 == （批大小，1，嵌入维度）
    x = self.embedding(x)
    # print(x.shape)
    # x = tf.nn.embedding_lookup(word2vec_model.vectors,x)#500000 128 12 1 ->12 * 1 *128

    # x 在拼接 （concatenation） 后的形状 == （批大小，1，嵌入维度 + 隐藏层大小）
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # 将合并后的向量传送到 GRU
    output, state = self.gru(x)

    # 输出的形状 == （批大小 * 1，隐藏层大小）
    output = tf.reshape(output, (-1, output.shape[2]))

    # 输出的形状 == （批大小，vocab）
    x = self.fc(output)

    return x, state, attention_weights
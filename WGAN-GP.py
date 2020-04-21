#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.optimizers import RMSprop
import keras.backend as K
from tqdm import tqdm


# In[2]:


import codecs
with codecs.open("neg_small.txt", "rb", encoding="utf-8", errors="ignore") as f:
    lines = f.read().split("\n")
    neg_text = []
    for line in lines:
        data = line.split("\n")[0]
        neg_text.append(data)


# In[3]:


len(neg_text)


# In[4]:


with codecs.open("pos_small.txt", "rb", encoding="utf-8", errors="ignore") as f:
    lines = f.read().split("\n")
    pos_text = []
    for line in lines:
        data = line.split("\n")[0]
        pos_text.append(data)


# In[5]:


len(pos_text)


# In[6]:


full_text = pos_text + neg_text


# In[7]:


from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()


# In[8]:


tokenizer.fit_on_texts(full_text)
word2index = tokenizer.word_index
VOCAB_SIZE = len(word2index) + 1
index2word = tokenizer.index_word


# In[9]:


neg_sequences = tokenizer.texts_to_sequences(neg_text)


# In[10]:


pos_sequences = tokenizer.texts_to_sequences(pos_text)


# In[11]:


import numpy as np
MAX_LEN = 15
# num_samples = len(encoder_sequences)
# decoder_output_data = np.zeros((num_samples, MAX_LEN, VOCAB_SIZE), dtype="float32")


# In[12]:


from keras.preprocessing.sequence import pad_sequences
neg_data = pad_sequences(neg_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
pos_data = pad_sequences(pos_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')


# In[13]:


full_data = pos_data + neg_data
print(full_data.shape)
print(type(full_data))


# In[14]:


import math
embeddings_index = {}
max_embedding_value = -math.inf
min_embedding_value = math.inf
with open('glove.6B.200d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        c_max = np.amax(coefs)
        c_min = np.amin(coefs)
        max_embedding_value = max(max_embedding_value, c_max)
        min_embedding_value = min(min_embedding_value, c_min)
        embeddings_index[word] = coefs
    f.close()

print("Glove Loded!")
print(len(embeddings_index))
print(max_embedding_value)
print(min_embedding_value)


# In[15]:


embedding_dimention = 200
def embedding_matrix_creater(embedding_dimention, word_index):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimention))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
          # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# In[16]:


embedding_matrix = embedding_matrix_creater(embedding_dimention, word_index=word2index)


# In[ ]:





# In[17]:


from keras.layers import Input, Dense, LSTM, TimeDistributed, Embedding
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.merge import _Merge
from functools import partial
import tensorflow as tf


# In[18]:


BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
TRAINING_RATIO = 5


# In[ ]:





# In[19]:


embed_layer = Embedding(input_dim=VOCAB_SIZE, 
                        output_dim=200, 
                        trainable=False, 
                        input_length=MAX_LEN,
                        weights=[embedding_matrix])


# # Wasserstein Loss and Gradient Penalty

# In[20]:


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


# In[21]:


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight = 10):
     # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,axis=np.arange(1, len(gradients_sqr.shape)))
    #     #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
#     gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


# # Generate Averaged Samples

# In[22]:


class RandomAveragedSamples(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


# # Generator model and Discriminator model

# In[29]:


def make_generator():
    model = Sequential()
    model.add(LSTM(1024, return_sequences=True))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(1024, return_sequences=True))
    model.add(TimeDistributed(Dense(200)))
    return model


# In[30]:


def make_discriminator():
    model = Sequential()
    model.add(LSTM(1024, return_sequences=True))
    model.add(LSTM(512))
    model.add(Dense(128))
    model.add(Dense(1))
    return model


# In[31]:


generator = make_generator()
discriminator = make_discriminator()


# In[32]:


# discrminator trainable false for generator training

for layer in discriminator.layers:
    layer.trainable = False

discriminator.trainable = False

# for layer in generator.layers:
#     layer.trainable = True

# generator.trainable = True

input_sequences = Input(shape=(MAX_LEN,), dtype='int32')
embedding_output = embed_layer(input_sequences)

generator_layers = generator(embedding_output)

pre_train_generator_model = Model([input_sequences], [generator_layers])
pre_train_generator_model.compile(optimizer=Adam(lr=LEARNING_RATE), 
                                  loss='mse', 
                                  metrics=['mae'])
print(pre_train_generator_model.summary())


discriminator_layers_for_generator = discriminator(generator_layers)
generator_model = Model([input_sequences], [discriminator_layers_for_generator])
generator_model.compile(optimizer=Adam(lr=0.0001), 
              loss=wasserstein_loss, 
              metrics=['mae'])
print(generator_model.summary())


# In[33]:


# generator trainable false for discriminator training 
for layer in generator.layers:
    layer.trainable = False

generator.trainable = False

for layer in discriminator.layers:
    layer.trainable = True

discriminator.trainable = True

# real sample input and output
real_samples = Input(shape=(MAX_LEN,), dtype='int32')
real_samples_embedding = embed_layer(real_samples)

discriminator_output_from_real_samples = discriminator(real_samples_embedding)


# generator input and discriminator output for generated sentence
generator_input_for_discriminator = Input(shape=(MAX_LEN,), dtype='int32')
generator_input_for_discriminator_embedding = embed_layer(generator_input_for_discriminator)
generated_samples_for_discriminator = generator(generator_input_for_discriminator_embedding)

discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)


# averaged samples output
averaged_samples = RandomAveragedSamples()([real_samples_embedding, generator_input_for_discriminator_embedding])
# averaged_samples_embedding = embed_layer(averaged_samples)
discriminator_output_from_averaged_samples = discriminator(averaged_samples)

# gradient penalty loss
partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=averaged_samples)
partial_gp_loss.__name__ = 'gradient_penalty'

discriminator_model = Model(inputs=[real_samples, 
                                    generator_input_for_discriminator],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator, 
                                     discriminator_output_from_averaged_samples])

discriminator_model.compile(optimizer=Adam(0.0001),
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss])
print(discriminator_model.summary())


# # Training

# In[34]:


def get_output_embeddings(sentences):
    target_data = np.zeros(shape=(len(sentences),15,200))
    for i, sentence in enumerate(sentences):
        for j, index in enumerate(sentence):
            embedding_vector = embedding_matrix[index]
            target_data[i,j,:] = embedding_vector
    return target_data


# In[ ]:


positive_y = np.ones(shape=(BATCH_SIZE, 1), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros(shape=(BATCH_SIZE, 1), dtype=np.float32)
for epoch in tqdm(range(2)):
    print("Epoch: ", epoch)
    print("Number of batches: ", int(pos_data.shape[0] // BATCH_SIZE))
    discriminator_loss = []
    generator_loss = []
    pre_train_generator_loss = []
    minibatch_size = BATCH_SIZE * TRAINING_RATIO
    for i in range(int(pos_data.shape[0] // (minibatch_size))):
        generator_minibatch = pos_data[i * minibatch_size: (i + 1) * minibatch_size]
#         print(generator_minibatch.shape)
        actual_generator_minibatch_output = get_output_embeddings(generator_minibatch)
#         print(actual_generator_minibatch_output.shape)
#         print(actual_generator_minibatch_output)
        pre_train_generator_loss.append(pre_train_generator_model.train_on_batch([generator_minibatch],
                                                                                 [actual_generator_minibatch_output]))
        
        discriminator_minibatch = neg_data[i * minibatch_size: (i + 1) * minibatch_size]
        for j in range(TRAINING_RATIO):
            neg_sentence_batch = discriminator_minibatch[j * BATCH_SIZE: (j + 1) * BATCH_SIZE]
            d_noise = np.random.randint(0, VOCAB_SIZE + 1, size=(BATCH_SIZE, MAX_LEN))
            discriminator_loss.append(discriminator_model.train_on_batch([neg_sentence_batch, d_noise], 
                                                                         [negative_y, positive_y, dummy_y]))
        
        g_noise = np.random.randint(0, VOCAB_SIZE + 1, size=(BATCH_SIZE, MAX_LEN))
        generator_loss.append(generator_model.train_on_batch([g_noise], 
                                                             [positive_y]))
            


# In[ ]:


prediction = []
for i, sentence in enumerate(test):
    res = []
    for j, embedding in enumerate(sentence):
#         print(embedding)
        res.append(nearest_word_to_embedding(embedding))
    prediction.append(res)
print(prediction)


# In[ ]:


def l2_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


# In[ ]:


import math
def nearest_word_to_embedding(embedding):
    # First find the word with nearest distance.
    min_distance = math.inf
    nearest_word = ""

    for word, word_index in word2index.items():

        distance = l2_distance(embedding, embedding_matrix[word_index])
        if distance < min_distance:
            min_distance = distance
            nearest_word = word

    return nearest_word


# In[ ]:





# In[ ]:





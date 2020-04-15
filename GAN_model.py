#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: bhavsar.y, shah.h
"""
from tqdm import tqdm
import torch
import torch.utils.data
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import re
import string
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from generator import GenerativeModel
from discriminator import DiscriminaterModel
# from rec_loss import rec_loss

VISIBLE_OUTPUT_RATE = 10

NEG_SAMPLES = list()
POS_SAMPLES = list()
BATCH_SIZE = 300
WORD2INDEX = dict()
EMBEDDINGS = list()
VOCAB = list()
CORPUS = list()
EMBEDDING_SIZE = -1
learning_rate = 0.0001
num_epochs = 50
BATCH_ITERATION_LIMIT = 150

D_model = None
G_model = None

MODEL_NAME = "model_lr_" + str(learning_rate) + "_epochs_" + str(num_epochs) + \
    "_itr_" + str(BATCH_ITERATION_LIMIT) + "_batchsize_"+ str(BATCH_SIZE)


def read_yelp_dataset():
    # First read the pos.txt file and then read the neg.txt
    POS_DATASET = "pos_small.txt"
    NEG_DATASET = "neg_small.txt"

    pos = list()
    neg = list()

    with open(POS_DATASET, "r") as pos_file:
        data = pos_file.readlines()
        for line in data:
            if 2 <= len(line.split()) <= 6:
                pos.append(line)

    with open(NEG_DATASET, "r") as neg_file:
        data = neg_file.readlines()
        for line in data:
            if 2 <= len(line.split()) <= 6:
                neg.append(line)

    return pos, neg

def read_embeddings():
    # Using ready to use embeddings.
    EMBEDDING_FILE = 'embeddings.txt'

    words = []
    idx = 0
    word2idx = {}
    vectors = []

    with open(EMBEDDING_FILE, "r") as embeddings_file:
        for line in embeddings_file:
            line = line.split()
            word = remove_punctuations(line[0])
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vector = np.array(line[1:]).astype(np.float)
            vectors.append(vector)

    vectors = np.array(vectors)
    return words, vectors, word2idx


def generate_randoms(lower_bound, upper_bound, num_samples):
    return [np.random.randint(lower_bound, upper_bound) for i in range(0, num_samples)]


def remove_punctuations(str):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub("", str)


def convert_sentences_to_embeddings(sentences):

    result = list()
    for single_sentence in sentences:
        for word in single_sentence.split():
            idx = WORD2INDEX.get(
                remove_punctuations(word), WORD2INDEX['UNK'])
            result.append(EMBEDDINGS[idx])

    return np.array(result)


def read_random_samples(batch_size=BATCH_SIZE):
    samples = [CORPUS[i] for i in generate_randoms(0, len(CORPUS), batch_size)]
    return samples


def read_neg_samples(sample_size=BATCH_SIZE):
    neg_samples = [NEG_SAMPLES[i]
                   for i in generate_randoms(0, len(NEG_SAMPLES), sample_size)]
    return neg_samples


def read_real_samples(sample_size=BATCH_SIZE):
    real_text = [POS_SAMPLES[i] for i in generate_randoms(
        0, len(POS_SAMPLES), sample_size)]
    return real_text

def loss_function_1(discriminator_output, expected_output):
    prediction_transferred = discriminator_output
    prediction_target = expected_output
    #pt
    inner_term = prediction_transferred<0.0
    middle_term = inner_term.type(torch.FloatTensor)
    outer_term = torch.mean(middle_term)
    transferred_accuracy = outer_term
    #transferred_accuracy = tf.reduce_mean(tf.cast(tf.less(prediction_transferred, 0.0), tf.float32))transferred_accuracy = tf.reduce_mean(tf.cast(tf.less(prediction_transferred, 0.0), tf.float32))
    #pt
    #pt
    inner_term = prediction_target<0.0
    middle_term = inner_term.type(torch.FloatTensor)
    outer_term = torch.mean(middle_term)
    target_accuracy = outer_term    
    # target_accuracy = tf.reduce_mean(tf.cast(tf.greater_equal(prediction_target, 0.0), tf.float32))
    #pt


    
    
    #pt
    target_loss = torch.mean(prediction_target)
    # target_loss = tf.reduce_mean(prediction_target)
    #pt
    #pt
    transferred_loss = torch.mean(prediction_transferred)
    # transferred_loss = tf.reduce_mean(prediction_transferred)
    #pt
    # total loss is the sum of losses
    total_loss = - target_loss + transferred_loss
    # total accuracy is the avg of accuracies
    total_accuracy = 0.5 * (transferred_accuracy + target_accuracy)
    return total_loss, total_accuracy

# def rec_loss():
#         # def get_margin_loss_v2(self, true_embeddings, decoded_embeddings, random_words_embeddings, padding_mask, margin):
#     per_word_distance = tf.sqrt(tf.reduce_sum(tf.squared_difference(true_embeddings, decoded_embeddings), axis=-1))

#     if random_words_embeddings is not None:
#         len_random_words = tf.shape(random_words_embeddings)[2]

#         sq_difference_expand = tf.expand_dims(per_word_distance, 2)
#         sq_difference_expand = tf.tile(sq_difference_expand, [1, 1, len_random_words])

#         target_expand = tf.expand_dims(decoded_embeddings, 2)
#         target_expand = tf.tile(target_expand, [1, 1, len_random_words, 1])
#         per_random_word_distance = tf.sqrt(
#             tf.reduce_sum(tf.squared_difference(target_expand, random_words_embeddings), axis=-1))

#         per_word_margin_loss = tf.maximum(0.0, margin + sq_difference_expand - per_random_word_distance)
#         per_word_distance = tf.reduce_mean(per_word_margin_loss, axis=-1)

#     mask = tf.where(padding_mask, tf.ones_like(padding_mask, dtype=tf.float32),
#                     tf.zeros_like(padding_mask, dtype=tf.float32))
#     sum = tf.reduce_sum(per_word_distance * mask)
#     mask_sum = tf.reduce_sum(mask)

#     return sum / mask_sum

def discriminator_training(optimizer=None):

    loss_function = nn.BCELoss()

    ERROR_REAL = 0
    ERROR_FAKE = 0

    pos_batch_iterator = torch.utils.data.DataLoader(
        POS_SAMPLES, batch_size=BATCH_SIZE, shuffle=True)

    neg_batch_iterator = torch.utils.data.DataLoader(
        NEG_SAMPLES, batch_size=BATCH_SIZE, shuffle=True)

    for batch_index, batch in enumerate(pos_batch_iterator):

        real_samples = batch

        # discriminator output should be for real embeddings
        real_embeddings = convert_sentences_to_embeddings(real_samples)
        real_embeddings = Variable(torch.from_numpy(real_embeddings)).float()

        real_probability_output = np.ones(shape=(real_embeddings.shape[0], 1))
        expected_output = Variable(
            torch.from_numpy(real_probability_output)).float()

        discriminator_output = D_model(real_embeddings)
        error_real = loss_function_1(discriminator_output, expected_output)[0]
        ERROR_REAL += error_real
        optimizer.zero_grad()
        error_real.backward()
        optimizer.step()

        if batch_index == BATCH_ITERATION_LIMIT:
            break

    for batch_index, batch  in enumerate(neg_batch_iterator):
        optimizer.zero_grad()
        fake_samples = batch

        # discriminator output should be for real embeddings
        fake_embeddings = convert_sentences_to_embeddings(fake_samples)
        fake_embeddings = Variable(torch.from_numpy(fake_embeddings)).float()

        fake_probability_output = np.zeros(shape=(fake_embeddings.shape[0], 1))
        expected_output = Variable(
            torch.from_numpy(fake_probability_output)).float()

        discriminator_output = D_model(fake_embeddings)
        error_fake = loss_function_1(discriminator_output, expected_output)[0]
        ERROR_FAKE += error_fake
        optimizer.zero_grad()
        error_fake.backward()
        optimizer.step()

        if batch_index == BATCH_ITERATION_LIMIT:
            break

    # step
    optimizer.step()
    return ERROR_REAL + ERROR_FAKE

def random_word_generator(size):
    randoms = np.random.randint(0,high = EMBEDDING_SIZE, size = size)
    res = []
    for i in randoms:
        res.append(EMBEDDINGS[i])
    return Variable(torch.from_numpy(np.array(res))).float()
    # return 

def rec_loss(output, embeddings, margin = 1.0):
    # print(type(output))

    squared_diff = (output ** 2 - embeddings ** 2)
    random_words = random_word_generator(output.shape[0])

    per_word_distance = torch.norm((embeddings - output))**2
    per_random_word_distance = torch.norm((random_words - output))**2
    # print(type(per_word_distance))
    # print(per_word_distance)
    # print(type(per_random_word_distance))
    # loss = max(torch.tensor(0.0), torch.tensor(margin) + per_word_distance - per_random_word_distance)
    loss =torch.tensor(margin) + per_word_distance - per_random_word_distance
    # print(loss)
    # loss.requires_grad = True
    # print(loss)
    # print(random_words.shape)
    # exit()
    return loss
    # per_word_distance = torch.norm(squared_diff, 2)


def generator_training(optimizer=None):

    optimizer.zero_grad()

    pos_batch_iterator = torch.utils.data.DataLoader(
        POS_SAMPLES, batch_size=BATCH_SIZE, shuffle=True)

    neg_batch_iterator = torch.utils.data.DataLoader(
        CORPUS, batch_size=BATCH_SIZE, shuffle=True)

    ERROR_GENERATOR = 0

    # Iterate over the positive domain and apply reconstruction loss.
    for batch_index, batch in enumerate(pos_batch_iterator):
        # print(batch)
        # Get the embeddings and convert to Variable
        embeddings = convert_sentences_to_embeddings(batch)
        # print(embeddings)
        # exit()
        embeddings = Variable(torch.from_numpy(embeddings)).float()

        # get the generator output
        output = G_model(embeddings)
# 
        # Calculate the loss using mseLoss.
        loss_mse = nn.MSELoss()
        # rec_loss = rec_loss(EMBEDDING_SIZE, EMBEDDINGS, output.shape[0])
        # loss_mse(output, embeddings)
        reconstruction_error = rec_loss(output, embeddings)
        optimizer.zero_grad()
        reconstruction_error.backward()
        optimizer.step()

        if batch_index == BATCH_ITERATION_LIMIT:
            break

    # Iterate over negative batch and connect to the discriminator with
    # expected output of 1 and then see the error.
    for batch_index, batch in enumerate(neg_batch_iterator):
        # embeddings and Variable conversion.
        embeddings = convert_sentences_to_embeddings(batch)
        embeddings = Variable(torch.from_numpy(embeddings)).float()

        expected_output = np.ones(shape=(embeddings.shape[0], 1))
        expected_output = Variable(torch.from_numpy(expected_output)).float()

        generated_embeddings = G_model(embeddings)

        # discriminate
        discriminator_output = D_model(generated_embeddings)

        # get the error function
        loss_bce = nn.BCELoss()
        error = loss_function_1(discriminator_output, expected_output)[0]
        ERROR_GENERATOR += error
        optimizer.zero_grad()
        error.backward()

        optimizer.step()

        if batch_index == BATCH_ITERATION_LIMIT:
            break

    return ERROR_GENERATOR


def l2_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def nearest_word_to_embedding(embedding):
    # First find the word with nearest distance.
    min_distance = math.inf
    nearest_word = ""

    for word, word_index in WORD2INDEX.items():

        distance = l2_distance(embedding, EMBEDDINGS[word_index])
        if distance < min_distance:
            min_distance = distance
            nearest_word = word

    return nearest_word


def transform_sentences(test_samples_count=10, save_path = "output_" + MODEL_NAME + ".txt", load_model = False):
    # Get some random samples from the negative sentences
    samples = read_neg_samples(sample_size=test_samples_count)


    save_file_str = ""
    
    # Convert a sentence at a time, as we are not familiar with
    # the number of words in each sentence.

    for sample_sentence in samples:

        # get the embeddings for the sample_sentence
        sample_embedding = convert_sentences_to_embeddings([sample_sentence])

        sample_embedding = Variable(torch.from_numpy(sample_embedding)).float()

        # Convert the embedding using the generator
        if load_model:
            VOCAB, EMBEDDINGS, WORD2INDEX = read_embeddings()
            EMBEDDING_SIZE = EMBEDDINGS.shape[1]
            model = GenerativeModel(EMBEDDING_SIZE)
            # model.load_state_dict('GeneratorModel_model_lr_0.0001_epochs_500_itr_150_batchsize_300')
            # model.eval()
            # model.eval()
            # model = GenerativeModel()
            model.load_state_dict(torch.load('GeneratorModel_model_lr_0.0001_epochs_500_itr_150_batchsize_300'))
            # model = torch.load('GeneratorModel_model_lr_0.0001_epochs_500_itr_150_batchsize_300')
            # model.eval()
            converted_embedding = model(sample_embedding)
        else:
            converted_embedding = G_model(sample_embedding)
        # Get the nearest word to the embedding
        converted_sentence = ""
        for single_converted_embedding in converted_embedding:
            converted_sentence += nearest_word_to_embedding(
                single_converted_embedding.detach().numpy()) + " "

        save_file_str += "Input Sentence: " + sample_sentence + "\n"
        save_file_str += "Converted Sentence: " + sample_sentence + "\n"
        save_file_str += "----------------------\n"

        print("Input Sentence: ", sample_sentence)
        print("Converted Sentence: ", converted_sentence)
        bleu_metric(sample_sentence, converted_sentence)
        print("-------")

    with open(save_path, "w") as output_converted_sentences_file:
        output_converted_sentences_file.write(save_file_str)
    
    del save_file_str

def bleu_metric(reference, candidate):
    smoothie = SmoothingFunction().method4
    print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0),smoothing_function=smoothie))
    print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0),smoothing_function=smoothie))
    print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0),smoothing_function=smoothie))
    print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=smoothie))
    pass

if __name__ == '__main__':
    # First read the input.
    POS_SAMPLES, NEG_SAMPLES = read_yelp_dataset()
    VOCAB, EMBEDDINGS, WORD2INDEX = read_embeddings()

    CORPUS.extend(POS_SAMPLES)
    CORPUS.extend(NEG_SAMPLES)

    # transform_sentences(test_samples_count=10, load_model=True)
    # exit()
    EMBEDDING_SIZE = EMBEDDINGS.shape[1]
    batches_done = 0

    D_model = DiscriminaterModel(EMBEDDING_SIZE)
    G_model = GenerativeModel(EMBEDDING_SIZE)

    g_optimizer = torch.optim.Adam(G_model.parameters(), lr=learning_rate)
    d_optimizer = torch.optim.Adam(D_model.parameters(), lr=learning_rate)

    for step_epoch in tqdm(range(0, num_epochs)):
        print("Starting Step:" + str(step_epoch))

        d_error = discriminator_training(optimizer=d_optimizer)
        g_error = generator_training(optimizer=g_optimizer)

        if step_epoch % 1 == 0:
            print("Step: " + str(step_epoch))
            print("Discriminator Error:" + str(d_error))
            print("Generator Error:" + str(g_error))
            print("-----------------")

    # Once the training is done, let's convert some samples
    transform_sentences(test_samples_count=100)

    # Save the models for later reuse. 
    torch.save(G_model.state_dict(), "GeneratorModel_" + MODEL_NAME)
    torch.save(D_model.state_dict(), "DiscriminatorModel_" + MODEL_NAME)

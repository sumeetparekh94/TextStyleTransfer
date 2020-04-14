#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: bhavsar.y, shah.h
"""

import torch
import torch.utils.data
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import re
import string

from generator import GenerativeModel
from discriminator import DiscriminaterModel

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
num_epochs = 500
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
        error_real = loss_function(discriminator_output, expected_output)
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
        error_fake = loss_function(discriminator_output, expected_output)
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
    return np.array(res)

def rec_loss(output, embeddings, margin = 1.0):
    squared_diff = (output ** 2 - embeddings ** 2)
    random_words = random_word_generator(output.shape[0])
    per_word_distance = np.linalg.norm((embeddings - output)) ** 2
    per_random_word_distance = np.linalg.norm((random_words - output)) ** 2

    loss = max(0.0, margin + per_word_distance - per_random_word_distance)
    print(loss)
    print(random_words.shape)
    exit()
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

        # Get the embeddings and convert to Variable
        embeddings = convert_sentences_to_embeddings(batch)
        embeddings = Variable(torch.from_numpy(embeddings)).float()

        # get the generator output
        output = G_model(embeddings)

        # Calculate the loss using mseLoss.
        loss_mse = nn.MSELoss()
        reconstruction_error = loss_mse(output, embeddings)
        rec_loss(output, embeddings)
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
        error = loss_bce(discriminator_output, expected_output)
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


def transform_sentences(test_samples_count=10, save_path = "output_" + MODEL_NAME + ".txt"):
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
        print("-------")

    with open(save_path, "w") as output_converted_sentences_file:
        output_converted_sentences_file.write(save_file_str)
    
    del save_file_str

if __name__ == '__main__':

    # First read the input.
    POS_SAMPLES, NEG_SAMPLES = read_yelp_dataset()
    VOCAB, EMBEDDINGS, WORD2INDEX = read_embeddings()

    CORPUS.extend(POS_SAMPLES)
    CORPUS.extend(NEG_SAMPLES)

    EMBEDDING_SIZE = EMBEDDINGS.shape[1]
    batches_done = 0

    D_model = DiscriminaterModel(EMBEDDING_SIZE)
    G_model = GenerativeModel(EMBEDDING_SIZE)

    g_optimizer = torch.optim.Adam(G_model.parameters(), lr=learning_rate)
    d_optimizer = torch.optim.Adam(D_model.parameters(), lr=learning_rate)

    for step_epoch in range(0, 2):
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

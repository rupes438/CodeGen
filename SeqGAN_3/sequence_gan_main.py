import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import pickle as cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import checksyntax

#########################################################################################
#  Generator  Hyper-parameters (LSTM)
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 30 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 400 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64

#########################################################################################
#  Discriminator  Hyper-parameters (CNN)
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 200
positive_file = 'save/real_data.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
generated_num = 5000
true_file = 'save/Cprogs_len30.txt'


# def tokenize(s):
#     return [c for c in ' '.join(s.split(" "))]

def load_positive(data_file, outfile, token2idx, pad_idx):
    with open(data_file, 'r') as f:
        with open(outfile, 'w') as f_out:
            for line in f.read().splitlines():
                if line.strip():
                    test = line.split()
                    line_tokens = [str(token2idx[token]) for token in line.split()]
                    for i in range(SEQ_LENGTH - len(line_tokens)):
                        line_tokens.extend([str(pad_idx)])
                    f_out.writelines(' '.join(line_tokens))
                    f_out.write("\n")


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file, idx2token = None):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
#	print(generated_samples)
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:

        if idx2token == None:
            for idxs in generated_samples:
                buffer = ' '.join([str(idx) for idx in idxs]) + '\n'
                fout.write(buffer)
        else:
            for idxs in generated_samples:
                buffer = ' '.join([idx2token[idx] for idx in idxs]) + '\n'
                fout.write(buffer)

def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        print("Target lstm vitra ko x:\n",sess.run(target_lstm.x, {target_lstm.x: batch}))
        print("yo chai predictions re x sanga compare gareko:\n", sess.run(target_lstm.g_predictions, {target_lstm.x: batch}))
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        print(g_loss)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    with open(true_file,'r') as f_pos:
        file_contents = f_pos.read().splitlines()
        file_contents = [content.split() for content in file_contents]
        tokens = set([item for sublist in file_contents for item in sublist])
        # tokens = set(file_contents)

    pad_idx = len(tokens)
    vocab_size = pad_idx + 1

    token2idx = dict((token, i) for i, token in enumerate(tokens))
    idx2token = dict((i, token) for i, token in enumerate(tokens))
    idx2token[pad_idx] = " "
    load_positive(true_file, positive_file, token2idx, pad_idx)

    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing

    dis_data_loader = Dis_dataloader(BATCH_SIZE)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    target_params = cPickle.load(open('save/target_params.pkl','rb'), encoding='latin1')
    target_params[0] = np.random.random([vocab_size,32]).astype(np.float32)
    target_params[13] = np.random.random([32, vocab_size]).astype(np.float32)
    target_params[14] = np.random.random([vocab_size,]).astype(np.float32)

    target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model

    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim,
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())


    gen_data_loader.create_batches(positive_file, SEQ_LENGTH)

    # log file that stores progress
    log = open('save/experiment-log.txt', 'w')

    #  pre-train generator
    print('Start pre-training...')
    log.write('pre-training...\n')


    all_pre_train_losses = []
    for epoch in range(PRE_EPOCH_NUM):
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        all_pre_train_losses.append(loss)

    plt.plot(all_pre_train_losses)
    plt.savefig('pre_train_losses_plot.png')

    gen_outfile = 'save/generated_by_generator_after_' + str(PRE_EPOCH_NUM) + '_' + str(datetime.datetime.now()) + '_epochs.txt'

    generate_samples(sess, generator, BATCH_SIZE, generated_num, gen_outfile, idx2token)

    checksyntax.check_code(log,gen_outfile)


        # if epoch % 5 == 0:
        #     generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
        #     likelihood_data_loader.create_batches(eval_file, SEQ_LENGTH)
        #     test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
        #     print('pre-train epoch ', epoch, 'test_loss ', test_loss)
        #     buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
        #     log.write(buffer)

    print('Start pre-training discriminator...')
    # Train 3 epoch on the generated data and do this for 50 times
    for i in range(50):
        print("discriminator pre train epoch : ", i)
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file, SEQ_LENGTH)
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):

                x_batch, y_batch = dis_data_loader.next_batch()

                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _ = sess.run(discriminator.train_op, feed)

    gen_outfile = 'save/generated_by_generator_after_discriminator_training_'+ str(datetime.datetime.now) + '.txt'

    generate_samples(sess, generator, BATCH_SIZE, generated_num, gen_outfile, idx2token)

    checksyntax.check_code(log, gen_outfile)

    rollout = ROLLOUT(generator, 0.8)

    print('#########################################################################')
    print('Start Adversarial Training...')
    log.write('adversarial training...\n')
    for total_batch in range(TOTAL_BATCH):
        print("total_batch : ", total_batch)
        if total_batch % 20 == 0:
            file_name = 'save/output_batch_' + str(total_batch) + '.txt'
            generate_samples(sess, generator, BATCH_SIZE, generated_num, file_name, idx2token)

            checksyntax.check_code(log, file_name)

        # Train the generator for one step
        for it in range(1):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)
        # Test
    #     if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
    #         generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
    #         likelihood_data_loader.create_batches(eval_file)
    #         test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
    #         buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
    #         print('total_batch: ', total_batch, 'test_loss: ', test_loss)
    #         log.write(buffer)
    #
        # Update roll-out parameters
        rollout.update_params()
    #
        # Train the discriminator
        for _ in range(1):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file, SEQ_LENGTH)

            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(discriminator.train_op, feed)
        final_gen_file = 'save/final_output.txt'
        generate_samples(sess, generator, BATCH_SIZE, generated_num, final_gen_file, idx2token)

        checksyntax.check_code(log, final_gen_file)

    #     with open('save/output.txt','r') as f:
    #         with open('save/output_word.txt','w') as fout:
    #             for line in f:
    #                 line = line.strip()
    #                 line = line.split()
    #                 word_line = ''.join([idx2token[int(x)] for x in line])
    #                 fout.write(word_line + '\n')
    #
    log.close()


if __name__ == '__main__':
    main()

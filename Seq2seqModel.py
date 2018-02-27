import tensorflow as tf


class Seq2SeqModel:
    def __init__(self):
        self.encoder_hidden_unit = 150
        self.decoder_hidden_unit = 150
        self.max_n_times_in = 3
        self.max_n_times_out = 4
        self.n_features = 12
        self.time_major = True
        self.is_attention = False
        self.batch_size = 100

        self.create_embeddings()
        encoder_outputs, encoder_state = self.create_encoder()
        decoder_outputs, decoder_state = self.create_decoder(encoder_outputs,
                                                             encoder_state)

        decoder_logits = self.create_fully_connected_layer(decoder_outputs)
        self.loss = self.create_loss(decoder_logits)

        self.train_op = self.create_optimizer()
        self.summaries()

    def create_embeddings(self):
        self.encoder_inputs = tf.placeholder(shape=[None, None],
                                             dtype=tf.int32,
                                             name='encoder_inputs')
        self.decoder_inputs = tf.placeholder(shape=[None, None],
                                             dtype=tf.int32,
                                             name='decoder_inputs')
        self.decoder_targets = tf.placeholder(shape=[None, None],
                                              dtype=tf.int32,
                                              name='decoder_targets')
        self.decoder_full_length = tf.placeholder(shape=(None,),
                                                  dtype=tf.int32,
                                                  name='decoder_full_length')

        embeddings = tf.Variable(tf.random_uniform(
                                    [self.n_features,
                                     self.encoder_hidden_unit],
                                    -1.0, 1.0), dtype=tf.float32)

        self.encoder_embeddings = tf.nn.embedding_lookup(embeddings,
                                                         self.encoder_inputs)
        self.decoder_embeddings = tf.nn.embedding_lookup(embeddings,
                                                         self.decoder_inputs)
        self.decoder_targets_embeddings = tf.one_hot(self.decoder_targets,
                                                     self.n_features,
                                                     dtype=tf.float32)

    def create_encoder(self):
        # Build RNN cell
        with tf.variable_scope("encoder") as scope:
            encoder_cell = tf.contrib.rnn.LSTMCell(self.encoder_hidden_unit)
            # encoder_cell = tf.contrib.rnn.GRUCell(self.encoder_hidden_unit)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell=encoder_cell,
                inputs=self.encoder_embeddings,
                dtype=tf.float32,
                time_major=self.time_major,
                swap_memory=True,
                scope=scope
            )

        return encoder_outputs, encoder_state

    def create_decoder(self, encoder_outputs, encoder_state):
        with tf.variable_scope("decoder") as decoder_scope:
            helper = tf.contrib.seq2seq.TrainingHelper(
                    self.decoder_embeddings,
                    self.decoder_full_length,
                    time_major=self.time_major)

            if self.is_attention:
                decoder_cell = tf.contrib.rnn.LSTMCell(self.decoder_hidden_unit)
                # decoder_cell = tf.contrib.rnn.GRUCell(self.decoder_hidden_unit)
                if self.time_major:
                    # [batch_size, max_time, num_nunits]
                    attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
                else:
                    attention_states = encoder_outputs

                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.decoder_hidden_unit, attention_states)

                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    decoder_cell, attention_mechanism,
                    attention_layer_size=self.decoder_hidden_unit)

                decoder_initial_state = (
                     decoder_attetion_cell.zero_state(tf.size(self.decoder_full_length),
                                                      dtype=tf.float32).clone(
                                                        cell_state=encoder_state))
            else:
                decoder_cell = tf.contrib.rnn.LSTMCell(self.decoder_hidden_unit)
                decoder_initial_state = encoder_state

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=helper,
                initial_state=decoder_initial_state)

            outputs, last_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=self.time_major)

        return outputs.rnn_output, last_state

    def create_fully_connected_layer(self, decoder_outputs):
        decoder_logits = (tf.contrib.layers.linear(decoder_outputs,
                                                   self.n_features))

        self.decoder_prediction = tf.argmax(decoder_logits, 2)

        return decoder_logits

    def create_loss(self, decoder_logits):
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
             labels=self.decoder_targets_embeddings,
             logits=decoder_logits)

        loss = (tf.reduce_mean(stepwise_cross_entropy))
        return loss

    def create_optimizer(self):
        return tf.train.AdamOptimizer().minimize(self.loss)

    def summaries(self):
        with tf.name_scope("sumaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram_loss", self.loss)

            self.summary_op = tf.summary.merge_all()

    def update(self, sess, encoder_inputs,
               decoder_inputs, decoder_targets, decoder_full_length):
        return sess.run([self.train_op, self.loss, self.summary_op],
                        feed_dict={
                            self.encoder_inputs: encoder_inputs,
                            self.decoder_inputs: decoder_inputs,
                            self.decoder_targets: decoder_targets,
                            self.decoder_full_length: decoder_full_length})

    def predict(self, sess, encoder_inputs,
                decoder_inputs, decoder_full_length):
        pred = sess.run([self.decoder_prediction],
                        feed_dict={self.encoder_inputs: encoder_inputs,
                                   self.decoder_inputs: decoder_inputs,
                                   self.decoder_full_length: decoder_full_length
                                   })

        return pred

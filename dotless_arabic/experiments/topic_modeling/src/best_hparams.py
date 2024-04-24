best_hparams = {
    "sanad": {
        "WordTokenizer": dict(
            rnn_hiddens=128,
            rnn_dropout=0.5,
            dropout_prob=0.5,
            embedding_size=128,
            learning_rate=0.001,
        ),
        "SentencePieceTokenizer": dict(
            rnn_hiddens=512,
            rnn_dropout=0.5,
            dropout_prob=0.5,
            embedding_size=128,
            learning_rate=0.001,
        ),
        "FarasaMorphologicalTokenizer": dict(
            rnn_hiddens=256,
            rnn_dropout=0.5,
            dropout_prob=0.5,
            embedding_size=128,
            learning_rate=0.001,
        ),
        "DisjointLetterTokenizer": dict(
            rnn_hiddens=256,
            rnn_dropout=0.25,
            dropout_prob=0.5,
            embedding_size=128,
            learning_rate=0.001,
        ),
        # "CharacterTokenizer": dict(
        #     num_layers=2,
        #     learning_rate=0.001,
        #     hidden_size=256,
        #     embedding_size=512,
        #     dropout_prop=0.2,
        # ),
    },
}

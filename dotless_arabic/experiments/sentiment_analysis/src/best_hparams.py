best_hparams = {
    "labr": {
        "WordTokenizer": dict(
            num_layers=2,
            rnn_hiddens=256,
            rnn_dropout=0.45,
            dropout_prob=0.45,
            embedding_size=256,
            learning_rate=0.001,
        ),
        # "FarasaMorphologicalTokenizer": dict(
        #     num_layers=2,
        #     learning_rate=0.001,
        #     hidden_size=256,
        #     embedding_size=512,
        #     dropout_prop=0.2,
        # ),
        # "DisjointLetterTokenizer": dict(
        #     num_layers=2,
        #     learning_rate=0.001,
        #     hidden_size=256,
        #     embedding_size=512,
        #     dropout_prop=0.2,
        # ),
        # "CharacterTokenizer": dict(
        #     num_layers=2,
        #     learning_rate=0.001,
        #     hidden_size=256,
        #     embedding_size=512,
        #     dropout_prop=0.2,
        # ),
    },
}

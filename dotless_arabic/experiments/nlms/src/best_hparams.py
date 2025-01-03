best_hparams = {
    "quran": {
        "WordTokenizer": dict(
            num_layers=2,
            learning_rate=0.01,
            hidden_size=256,
            embedding_size=256,
            dropout_prob=0.333,
        ),
        "FarasaMorphologicalTokenizer": dict(
            num_layers=3,
            learning_rate=0.01,
            hidden_size=256,
            embedding_size=512,
            dropout_prob=0.5,
        ),
        "DisjointLetterTokenizer": dict(
            num_layers=3,
            learning_rate=0.01,
            hidden_size=256,
            embedding_size=512,
            dropout_prop=0.3333,
        ),
        "CharacterTokenizer": dict(
            num_layers=2,
            learning_rate=0.01,
            hidden_size=256,
            embedding_size=512,
            dropout_prop=0.3333,
        ),
    },
    "sanadset_hadeeth": {
        "WordTokenizer": dict(
            num_layers=2,
            learning_rate=0.001,
            hidden_size=256,
            embedding_size=512,
            dropout_prob=0.2,
        ),
        "FarasaMorphologicalTokenizer": dict(
            num_layers=2,
            learning_rate=0.001,
            hidden_size=512,
            embedding_size=512,
            dropout_prob=0.2,
        ),
        "DisjointLetterTokenizer": dict(
            num_layers=2,
            learning_rate=0.001,
            hidden_size=512,
            embedding_size=512,
            dropout_prop=0.2,
        ),
        "CharacterTokenizer": dict(
            num_layers=2,
            learning_rate=0.001,
            hidden_size=512,
            embedding_size=512,
            dropout_prop=0.2,
        ),
    },
    "poems": {
        "WordTokenizer": dict(
            num_layers=2,
            learning_rate=0.001,
            hidden_size=256,
            embedding_size=512,
            dropout_prob=0.2,
        ),
        "FarasaMorphologicalTokenizer": dict(
            num_layers=2,
            learning_rate=0.01,
            hidden_size=256,
            embedding_size=256,
            dropout_prob=0.2,
        ),
        "DisjointLetterTokenizer": dict(
            num_layers=2,
            learning_rate=0.001,
            hidden_size=512,
            embedding_size=512,
            dropout_prop=0.2,
        ),
        "CharacterTokenizer": dict(
            num_layers=2,
            learning_rate=0.001,
            hidden_size=512,
            embedding_size=512,
            dropout_prop=0.2,
        ),
    },
    "wikipedia": {
        "WordTokenizer": dict(
            num_layers=2,
            learning_rate=0.001,
            hidden_size=512,
            embedding_size=512,
            dropout_prob=0.2,
        ),
        "FarasaMorphologicalTokenizer": dict(
            num_layers=2,
            learning_rate=0.001,
            hidden_size=512,
            embedding_size=512,
            dropout_prob=0.2,
        ),
        "DisjointLetterTokenizer": dict(
            num_layers=2,
            learning_rate=0.001,
            hidden_size=256,
            embedding_size=512,
            dropout_prop=0.2,
        ),
        "CharacterTokenizer": dict(
            num_layers=2,
            learning_rate=0.001,
            hidden_size=512,
            embedding_size=512,
            dropout_prop=0.2,
        ),
    },
    "news": {
        "WordTokenizer": dict(
            num_layers=2,
            learning_rate=0.001,
            hidden_size=512,
            embedding_size=512,
            dropout_prop=0.2,
        ),
        "FarasaMorphologicalTokenizer": dict(
            num_layers=2,
            learning_rate=0.001,
            hidden_size=512,
            embedding_size=512,
            dropout_prob=0.2,
        ),
        "DisjointLetterTokenizer": dict(
            num_layers=2,
            learning_rate=0.001,
            hidden_size=512,
            embedding_size=512,
            dropout_prop=0.2,
        ),
        "CharacterTokenizer": dict(
            num_layers=2,
            learning_rate=0.001,
            hidden_size=512,
            embedding_size=512,
            dropout_prop=0.2,
        ),
    },
}

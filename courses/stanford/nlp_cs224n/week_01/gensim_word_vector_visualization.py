import numpy as np
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# http://web.stanford.edu/class/cs224n/materials/Gensim%20word%20vector%20visualization.html
# https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
def main():
    glove_file = 'courses/stanford/nlp_cs224n/week_01/glove.6B/glove.6B.100d.txt'
    word2vec_glove_file = 'courses/stanford/nlp_cs224n/week_01/glove.6B.100d.word2vec.txt'
    # glove2word2vec(glove_file, word2vec_glove_file)

    model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
    print(model.most_similar('obama'))
    print(model.most_similar('banana'))
    print(model.most_similar(negative='banana'))

    result = model.most_similar(positive=['woman', 'king'], negative=['man'])
    print("{}: {:.4f}".format(*result[0]))

    def analogy(x1, x2, y1):
        result = model.most_similar(positive=[y1, x2], negative=[x1])
        return result[0][0]

    print(analogy('japan', 'japanese', 'australia'))
    print(analogy('australia', 'beer', 'france'))
    print(analogy('tall', 'tallest', 'long'))

    display_pca_scatterplot(model, ['movie', 'cinema', 'basketball', 'football', 'skating', 'youtube',
                                    'cat', 'dog'])


def display_pca_scatterplot(model, words=None, sample=0):
    if words is None:
        if sample > 0:
            words = np.random.choice(list(model.key_to_index.keys()), sample)
        else:
            words = [word for word in model.vocab]

    word_vectors = np.array([model[w] for w in words])

    twodim = PCA().fit_transform(word_vectors)[:, :2]

    plt.figure(figsize=(6, 6))
    plt.scatter(twodim[:, 0], twodim[:, 1], edgecolors='k', c='r')
    for word, (x, y) in zip(words, twodim):
        plt.text(x + 0.05, y + 0.05, word)
    plt.show()

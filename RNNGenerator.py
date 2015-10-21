__author__ = 'Ilkka'
from SoiniParser import SoiniParser
from MinCharRnn import MinCharRnn

if __name__ == '__main__':
    rnn = MinCharRnn('Soini/input.txt')
    rnn.runmodel()
    """
    parser = SoiniParser()
    posts = parser.getBlogPosts()
    with open('output.txt', 'w') as f:
        for post in posts:
            f.write("%s\n\n" % post.encode("UTF-8"))
    """
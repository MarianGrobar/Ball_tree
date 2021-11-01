import struct
import numpy as np
import sklearn.neighbors as sk
from timeit import default_timer as timer
from datetime import timedelta
import sys


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    vector_count = 1000000
    vector_size = 128
    mass = []
    f = open('sift1M.bin', 'rb')


    for i in range(vector_count):
        vector = []
        for j in range(vector_size):
            vector.append(struct.unpack('f', f.read(4))[0])
        mass.append(vector)
    f.close()

    massQ = []
    f = open('siftQ1M.bin', 'rb')
    for i in range(1000):
        vectorQ = []
        for j in range(vector_size):
            vectorQ.append(struct.unpack('f', f.read(4))[0])
        massQ.append(vectorQ)
    f.close()

    start = timer()
    ball_tree = sk.BallTree(mass, leaf_size=20)
    end = timer()
    print('Cas vkladani:' + timedelta(seconds=end - start).__str__())

    for k in range(3):
        query_results = []
        start = timer()
        ind = ball_tree.query(massQ, k=10,dualtree = True,return_distance = False)
        end = timer()
        print('Cas dotazovani:' + timedelta(seconds=end - start).__str__())

        for r in ind:
            query_results.append(r)

        textfile = open(f"query_results{k}.txt", "w")
        for query_result in query_results:
            textfile.write(','.join([str(elem) for elem in query_result.tolist()]) + "\n")
        textfile.close()
    print("Konec")

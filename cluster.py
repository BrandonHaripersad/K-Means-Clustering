import os
import re
import string
import json
import math
import sys
import random

stopwords = ["i", "a", "about", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what", "when",
"where", "who", "will", "with", "the"]
posting = {}
df = {}
terms = []
vectors = {}
document_titles = {}
#r'D:\Projects\CPS842-Project\bbcsport'

def create_tokens():
    print("Parsing Document...")
    directory = r'C:\Users\MV\Documents\GitHub\K-Means-Clustering\bbcsport'
    tokens = open("tokens.txt", "w")
    docid = 0
    for subdir, dirs, files in os.walk(directory):
        for filename in files:
            filepath = subdir + os.sep + filename
            category = subdir.rsplit('\\', 1)[-1]
            if filepath.endswith(".txt"):
                tokens.write("\n")
                with open((filepath), "r") as lines:
                    docid += 1
                    count = 0
                    for line in lines:
                        if line is "\n":
                            continue
                        else:
                            if count is 0:
                                document_titles[docid] = [category, line.strip()]
                                count += 1
                            line = line.strip()
                            line = line.replace("-", " ")
                            line = re.sub (r'([^a-zA-Z\s]+?)', '', line)
                            words = line.split()
                            words = [word.lower() for word in words]
                            table = str.maketrans('', '', string.punctuation)
                            stripped = [w.translate(table) for w in words]
                            for item in stripped:
                                if item in stopwords:
                                    continue
                                else:
                                    tokens.write("%s\n" % item)

def invert(tokens):
    print("Creating inverted index...")
    docid = 0
    with open(tokens) as lines:
        for line in lines:
            if line is "\n":
                docid += 1
                continue
            line = line.strip()
            if line not in terms:
                terms.append(line)
            if line in posting and docid in posting[line]: # if term exhists in posting list already, we do not want to increment df, just want to increment tf and add to postioning list
                posting[line][docid] += 1 # increase term frequency for the doc by 1
            else:
                if line in df:
                    df[line] += 1 # if not in posting list for that doc, then we increment df and add that docid to the posting list
                    posting[line][docid] = 1 # create the posting list entry for this doc and term
                else: # new term, needs to be added to df and posting list
                    df[line] = 1
                    posting[line] = {docid: 1}
    save = json.dumps(df, sort_keys=True)
    s = open("df.json", "w")
    s.write(save)
    s.close()
    terms.sort()


def create_vectors():
    print("Creating document vectors...")
    N = 737
    for docid in range(1,738):
        vectors[docid] = [] # create vectors
        for term in terms: # Go through terms of those documents
            if docid in posting[term].keys():
                f = posting[term][docid] # freq of term in relevant document
                docf = df[term] # document freq of that term
                tf = 1 + math.log(f,10) # calculate tf
                idf = math.log((N/docf),10) # calculate idf
                w = tf * idf # calculate weight of that term
                vectors[docid].append(w) # append weight of that term
            else:
                vectors[docid].append(0) # if the term is no in the document, then the weight is 0
        f = 0
        docf = 0
        tf = 0
        idf = 0
        w = 0
    save = json.dumps(vectors)
    s = open("vectors.json", "w")
    s.write(save)
    s.close()

def dot_product(a,b):
    return float(sum([x*y for x,y in zip(a,b)]))

def recalc_centroids(c):
    if len(c) == 0:
        return 0
    total = vectors[c[0]].copy()
    for i in range(1, len(c)-1):
        for p in range(len(vectors[c[i]])-1):
            vec = vectors[c[i]]
            vec_value = vectors[c[i]][p]
            total[p] = total[p] + vec_value
    for i in range(len(total)-1):
        total[i] = total[i] / len(c)
    return total
        

def max_sim(scores):
    highest = max(scores)
    index = 1
    for i in scores:
        if highest == i:
            return index
        else:
            index += 1

def e_distance(v1, v2):
    squares = [(p-q) ** 2 for p, q in zip(v1, v2)]
    return math.sqrt(sum(squares))

def cluster_tightness(cluster, centroid):
    total = 0
    for i in range(len(cluster)-1):
        vec = vectors[cluster[i]]
        total += e_distance(vec, centroid)
        for p in range(i+1, len(cluster)):
            v1 = vectors[cluster[i]]
            v2 = vectors[p]
            total += e_distance(v1, v2)
    return total

def get_document_titles(cluster):
    for i in cluster:
        print(document_titles[i][1] + " : " + document_titles[i][0])

def purity_test(cluster):
    classes = ["athletics", "cricket", "football", "rugby", "tennis"]
    class_counter = [0, 0, 0, 0, 0]
    for i in cluster:
        class_name = document_titles[i][0]
        if class_name == "athletics":
            class_counter[0] += 1
        elif class_name == "cricket":
            class_counter[1] += 1
        elif class_name == "football":
            class_counter[2] += 1
        elif class_name == "rugby":
            class_counter[3] += 1
        elif class_name == "tennis":
            class_counter[4] += 1
    
    max_value = max(class_counter)
    max_index = class_counter.index(max_value)
    return classes[max_index]


def clusters():
    intial_centroids = []
    for c in range(0, 5):
        n = random.randint(1,737)
        intial_centroids.append(n)
    centroids = [vectors[intial_centroids[0]], vectors[intial_centroids[1]], vectors[intial_centroids[2]], vectors[intial_centroids[3]], vectors[intial_centroids[4]]]
    prev_centroid = []
    clusters = {1: [], 2: [], 3: [], 4: [], 5: []}
    c = 0
    # intial clusters with "randomly" selected centroids
    for i in range(1, 738):
        if i in intial_centroids:
            continue
        else:
            s1 = dot_product(vectors[i],centroids[0])
            s2 = dot_product(vectors[i],centroids[1])
            s3 = dot_product(vectors[i],centroids[2])
            s4 = dot_product(vectors[i],centroids[3])
            s5 = dot_product(vectors[i],centroids[4])

            cluster_index = max_sim([s1, s2, s3, s4, s5])
            clusters[cluster_index].append(i)
    print("Intial Clusters:" + "\n")
    print(clusters)

    done = False

    while(done == False):

        c += 1
        prev_clusters = {}
        prev_clusters = clusters.copy()
        prev_centroid = []
        prev_centroid = centroids.copy()
        centroids.clear()
    
        # calculating new centroids then check similarity of ALL vectors to new centroids, repeat until no changes or 20 iterations
        centroids.append(recalc_centroids(clusters[1]))
        centroids.append(recalc_centroids(clusters[2]))
        centroids.append(recalc_centroids(clusters[3]))
        centroids.append(recalc_centroids(clusters[4]))
        centroids.append(recalc_centroids(clusters[5]))
        
        clusters.clear()
        clusters = {1: [], 2: [], 3: [], 4: [], 5: []}

        for p in range(1, 738):
            s1 = dot_product(vectors[p],centroids[0])
            s2 = dot_product(vectors[p],centroids[1])
            s3 = dot_product(vectors[p],centroids[2])
            s4 = dot_product(vectors[p],centroids[3])
            s5 = dot_product(vectors[p],centroids[4])
            cluster_index = max_sim([s1, s2, s3, s4, s5])
            clusters[cluster_index].append(p)
        
        print("Iteration: " + str(c) + "\n")
        print(clusters)

        if (c > 1) and prev_clusters == clusters and prev_centroid == centroids:
            print("Convergence Reached")
            print("Cluster #1")
            #get_document_titles(clusters[1])
            c1 = purity_test(clusters[1])
            print("Class: " + c1)
            print("Cluster #2")
            #get_document_titles(clusters[2])
            c2 = purity_test(clusters[2])
            print("Class: " + c2)
            print("Cluster #3")
            #get_document_titles(clusters[3])
            c3 = purity_test(clusters[3])
            print("Class: " + c3)
            print("Cluster #4")
            #get_document_titles(clusters[4])
            c4 = purity_test(clusters[4])
            print("Class: " + c4)
            print("Cluster #5")
            #get_document_titles(clusters[5])
            c5 = purity_test(clusters[5])
            print("Class: " + c5)
            #t1 = cluster_tightness(clusters[1], centroids[0])
            #print(t1)
            #t2 = cluster_tightness(clusters[2], centroids[1])
            #print(t2)
            #t3 = cluster_tightness(clusters[3], centroids[2])
            #print(t3)
            #t4 = cluster_tightness(clusters[4], centroids[3])
            #print(t4)
            #t5 = cluster_tightness(clusters[5], centroids[4])
            #print(t5)
            break
        #print("Centroids for Iteration " + str(c) + "\n")
        #print(centroids)



create_tokens()
print(document_titles)
invert("tokens.txt")
create_vectors()
save = json.dumps(posting, sort_keys=True)
s = open("posting.json", "w")
s.write(save)
s.close()
clusters()

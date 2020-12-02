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
    for subdir, dirs, files in os.walk(os.path.join(os.getcwd(), "bbcsport")):
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

def magnitude(vector):
    total = 0
    for i in vector:
        total = total + (i*i)
    return float(math.sqrt(total))

def normalize(vectors):
    print("Normilizing vectors...")
    for i in range(1,738):
        temp_list = []
        mag = magnitude(vectors[i])
        temp_list = [x / mag for x in vectors[i]]
        vectors[i] = temp_list
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
    result = []
    for i in cluster:
        result.append((document_titles[i][1],document_titles[i][0]))
    return result

def purity_test(clusters):
    classes = ["athletics", "cricket", "football", "rugby", "tennis"]
    result_list = []
    total = 0
    for key in clusters:
        class_counter = [0, 0, 0, 0, 0]
        for i in clusters[key]:
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
        class_name = classes[max_index]

        while (True):
            if (class_name in result_list):
                class_counter[max_index] = 0
                max_value = max(class_counter)
                max_index = class_counter.index(max_value)
                class_name = classes[max_index]
            else:
                break

        total += max_value
        result_list.append(class_name)
    
    purity = (1/737) * total
    result_list.append(purity)
    return result_list

def get_clostest_to_centroid(cluster, centroid):
    first_highest = 0
    second_higest = 0
    first_index = 0
    second_index = 0
    for i in cluster:
        s = dot_product(vectors[i], centroid)
        if (s > first_highest) and (s > second_higest):
            first_highest = s
            first_index = i
            continue
        elif (s > second_higest):
            second_higest = s
            second_index = i
            continue
        else:
            continue
    return [first_index, second_index]  

def clusters():
    print("Randomly selecting initial centroids...")
    intial_centroids = []
    for c in range(0, 5):
        n = random.randint(1,737)
        intial_centroids.append(n)
    centroids = [vectors[intial_centroids[0]], vectors[intial_centroids[1]], vectors[intial_centroids[2]], vectors[intial_centroids[3]], vectors[intial_centroids[4]]]
    prev_centroid = []
    clusters = {1: [], 2: [], 3: [], 4: [], 5: []}
    c = 0
    print("Initial Centroids: " + str(intial_centroids[0]) + ", "  + str(intial_centroids[1]) + ", " + 
    str(intial_centroids[2]) + ", " + str(intial_centroids[3]) + ", " + str(intial_centroids[4]))
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
    print("Intial Cluster created.")

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
        
        print("Cluster Iteration: " + str(c) + " created.")
        if (c > 1) and prev_clusters == clusters and prev_centroid == centroids:
            print("Convergence Reached")
            class_list = purity_test(clusters)
            d1 = get_document_titles(clusters[1])
            d2 = get_document_titles(clusters[2])
            d3 = get_document_titles(clusters[3])
            d4 = get_document_titles(clusters[4])
            d5 = get_document_titles(clusters[5])
            l1 = get_clostest_to_centroid(clusters[1], centroids[0])
            l2 = get_clostest_to_centroid(clusters[2], centroids[1])
            l3 = get_clostest_to_centroid(clusters[3], centroids[2])
            l4 = get_clostest_to_centroid(clusters[4], centroids[3])
            l5 = get_clostest_to_centroid(clusters[5], centroids[4])
            print("{: >40} {: >0} {: >0}".format("CLUSTER 1", "CLASS:", class_list[0]))
            print("{: >0}: {: >0}, {: >0}".format("Most similar to Centroid: ", str(document_titles[l1[0]]), str(document_titles[l1[1]])))
            print("\n")
            
            print("{: >40} {: >0} {: >0}".format("CLUSTER 2", "CLASS:", class_list[1]))
            print("{: >0}: {: >0}, {: >0}".format("Most similar to Centroid: ", str(document_titles[l2[0]]), str(document_titles[l2[1]])))
            print("\n")

            print("{: >40} {: >0} {: >0}".format("CLUSTER 3", "CLASS:", class_list[2]))
            print("{: >0}: {: >0}, {: >0}".format("Most similar to Centroid: ", str(document_titles[l3[0]]), str(document_titles[l3[1]])))
            print("\n")

            print("{: >40} {: >0} {: >0}".format("CLUSTER 4", "CLASS:", class_list[3]))
            print("{: >0}: {: >0}, {: >0}".format("Most similar to Centroid: ", str(document_titles[l4[0]]), str(document_titles[l4[1]])))
            print("\n")
            
            print("{: >40} {: >0} {: >0}".format("CLUSTER 5", "CLASS:", class_list[4]))
            print("{: >0}: {: >0}, {: >0}".format("Most similar to Centroid: ", str(document_titles[l5[0]]), str(document_titles[l5[1]])))
            print("\n")

            print("Purity Score: ", end="")
            print(class_list[5])

            show_all = input("Would you like to view all documents in each cluster? (y/n)")
            if (show_all == "y"):
                print("{: >40}".format("CLUSTER 1"))
                for i in d1:
                    print("{: >40}, {: >0}".format(i[0], i[1]))
                input("Press ENTER to print ClUSTER 2")
                print("{: >40}".format("CLUSTER 2"))
                for i in d2:
                    print("{: >40}, {: >0}".format(i[0], i[1]))
                input("Press ENTER to print ClUSTER 3")
                print("{: >40}".format("CLUSTER 3"))
                for i in d3:
                    print("{: >40}, {: >0}".format(i[0], i[1]))
                input("Press ENTER to print ClUSTER 4")
                print("{: >40}".format("CLUSTER 4"))
                for i in d4:
                    print("{: >40}, {: >0}".format(i[0], i[1]))
                input("Press ENTER to print ClUSTER 5")
                print("{: >40}".format("CLUSTER 5"))
                for i in d5:
                    print("{: >40}, {: >0}".format(i[0], i[1]))
                

            tightness = input("Would you like to calculate the Tightness of each cluster? (y/n)")
            if (tightness == "y"):
                print("Calculating tightness of Cluster #1")
                t1 = cluster_tightness(clusters[1], centroids[0])
                print("Tightness of Cluster #1: ", end="")
                print(str(t1))
                print("Calculating tightness of Cluster #2")
                t2 = cluster_tightness(clusters[2], centroids[1])
                print("Tightness of Cluster #2: ", end="")
                print(str(t2))
                print("Calculating tightness of Cluster #3")
                t3 = cluster_tightness(clusters[3], centroids[2])
                print("Tightness of Cluster #3: ", end="")
                print(str(t3))
                print("Calculating tightness of Cluster #4")
                t4 = cluster_tightness(clusters[4], centroids[3])
                print("Tightness of Cluster #4: ", end="")
                print(str(t4))
                print("Calculating tightness of Cluster #5")
                t5 = cluster_tightness(clusters[5], centroids[4])
                print("Tightness of Cluster #5: ", end="")
                print(str(t5))
            break



if __name__ == "__main__":
    print("K-MEANS CLUSTERING PROJECT - BRANDON HARIPERSAD")
    print("-"*47)
    create_tokens()
    invert("tokens.txt")
    create_vectors()
    normalize(vectors)
    save = json.dumps(posting, sort_keys=True)
    s = open("posting.json", "w")
    s.write(save)
    s.close()
    clusters()
    input("Press ENTER to exit")

##############################################################################
# This skeleton was created by Efehan Guner  (efehanguner21@ku.edu.tr)       #
# Note: requires Python 3.5+                                                 #
##############################################################################

import csv
from enum import unique
import glob
import os
import sys
from copy import deepcopy
import numpy as np
import datetime
import time

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)

##############################################################################
# Helper Functions                                                           #
# These functions are provided to you as starting points. They may help your #
# code remain structured and organized. But you are not required to use      #
# them. You can modify them or implement your own helper functions.          #
##############################################################################

def read_dataset(dataset_file: str):
    """ Read a dataset into a list and return.

    Args:
        dataset_file (str): path to the dataset file.

    Returns:
        list[dict]: a list of dataset rows.
    """
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    return result


def write_dataset(dataset, dataset_file: str) -> bool:
    """ Writes a dataset to a csv file.

    Args:
        dataset: the data in list[dict] format
        dataset_file: str, the path to the csv file

    Returns:
        bool: True if succeeds.
    """
    assert len(dataset)>0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True

class Queue:
    def __init__(self):
        self.list = []

    def push(self,item):
        self.list.insert(0,item)

    def pop(self):
        return self.list.pop()

    def isEmpty(self):
        return len(self.list) == 0

class Stack:
    def __init__(self):
        self.list = []

    def push(self,item):
        self.list.append(item)

    def pop(self):
        return self.list.pop()

    def isEmpty(self):
        return len(self.list) == 0

class Tree:
    def __init__(self, root_data):
        node = Node(root_data)
        self.root = node
        self.current_node = self.root
        node.level = 0

    #Adds a child to current node
    def addChild(self, data):
        node = Node(data)
        self.current_node.childs.append(node)
        node.parent = self.current_node
        node.level = node.parent.level + 1
        self.current_node = node
        return node

    #Return the parent of current node
    def getParent(self):
        return self.current_node.parent
    
    #Return the parent N levels up of the current node
    def getParentNLevel(self, level):
        node = self.current_node.parent
        for i in range(level):
            node = node.parent
        return node

    #Return the childs of the current node
    def getChilds(self):
        return self.current_node.childs

    #Return the node that contains the data
    #Returns a Empty Node if node is not present
    def getNode(self, data):
        queue = Queue()
        queue.push(self.root)
        while not queue.isEmpty():
            node = queue.pop()
            if node.data == data:
                return node
            else:
                for item in node.childs:
                    queue.push(item)
        return Node("")

    #Return the total number of leaves
    def getNumberOfLeaves(self):
        queue = Queue()
        queue.push(self.root)
        leaves = 0
        while not queue.isEmpty():
            node = queue.pop()
            if len(node.childs) == 0:
                leaves += 1
            for item in node.childs:
                queue.push(item)
        return leaves

    #Return the number of leaves from a specified node
    def getNumberOfDescendantLeaves(self, data):
        queue = Queue()
        if len(self.getNode(data).childs) == 0:
            return 1
        queue.push(self.getNode(data))
        leaves = 0
        while not queue.isEmpty():
            node = queue.pop()
            if len(node.childs) == 0:
                leaves += 1
            for item in node.childs:
                queue.push(item)
        return leaves

    #Return the level of the specific node that contains the data
    def getLevelOfNode(self, data):
        queue = Queue()
        queue.push((0, self.root))
        while not queue.isEmpty():
            level, node = queue.pop()
            if node.data == data:
                return level
            else:
                for item in node.childs:
                    queue.push((level + 1, item))
        return -1
    
    #Return the least common ancestor of two nodes in the tree
    def getCommonAncestor(self, data1, data2):
        node1 = self.getNode(data1)
        node2 = self.getNode(data2)
        
        if data1 == data2:
            return node1.parent.data
        while node1.level != node2.level:
            if node1.level > node2.level:
                node1 = node1.parent
            else:
                node2 = node2.parent
         
        while node1.data != node2.data:
            node1 = node1.parent
            node2 = node2.parent
            
        return node1.data

    #Print the tree in BFS manner
    def printTreeBFS(self):
        queue = Queue()
        if len(self.root.childs) == 0:
            print("No childs present. Root: ", self.root.data)
        queue.push((0, self.root))
        while not queue.isEmpty():
            level, node = queue.pop()
            for item in node.childs:
                queue.push((level + 1, item))
            print("Level: ", level, ", Node: ", node.data)

    #Print the tree in DFS manner
    def printTreeDFS(self):
        queue = Stack()
        if len(self.root.childs) == 0:
            print("No childs present. Root: ", self.root.data)
        queue.push((0, self.root))
        while not queue.isEmpty():
            level, node = queue.pop()
            for item in node.childs:
                queue.push((level + 1, item))
            print("Level: ", level, ", Node: ", node.data)
            
#Tree nodes          
class Node:
    def __init__(self, data):
        self.data = data
        self.parent = self
        self.childs = []
        self.level = -1

def read_DGH(DGH_file: str):
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file.
    """
    #TODO: complete this code so that a DGH file is read and returned
    # in your own desired format.
    fd = open(DGH_file, "r")
    content = fd.readlines()
    fd.close()
    DGH = Tree("")
    prev_tree_level = 0
    prev_group = ""
    for ln in content:
        tree_level = ln.count("\t")
        group = ln.replace("\t", "").replace("\n", "")
        if DGH_file.find("age") != -1:
            group = group.replace(",", " ")
        if tree_level == 0:
            DGH = Tree(group)
        if tree_level != 0:
            if prev_tree_level > tree_level:
                DGH.current_node = DGH.getParentNLevel(prev_tree_level - tree_level)
                DGH.addChild(group)
            elif prev_tree_level == tree_level:
                DGH.current_node = DGH.getParent()
                DGH.addChild(group)
            else:
                DGH.current_node = DGH.addChild(group)

        prev_tree_level = tree_level
    #print("BFS")
    #DGH.printTreeBFS()
    #print("DFS")
    #DGH.printTreeDFS()
    return DGH


def read_DGHs(DGH_folder: str) -> dict:
    """ Read all DGH files from a directory and put them into a dictionary.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.

    Returns:
        dict: a dictionary where each key is attribute name and values
            are DGHs in your desired format.
    """
    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file)

    return DGHs


##############################################################################
# Mandatory Functions                                                        #
# You need to complete these functions without changing their parameters.    #
##############################################################################


def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Distortion Metric (MD) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)
    #TODO: complete this function.
    cost = 0
    N = len(anonymized_dataset)
    for category in raw_dataset[0]:
        if category == "income" or category == "index":
            continue
        current_DGH = DGHs[category]
        for idx in range(N):
            cost += abs(current_DGH.getNode(anonymized_dataset[idx][category]).level - current_DGH.getNode(raw_dataset[idx][category]).level )
    return float(cost)



def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)
    #TODO: complete this function.
    cost = 0
    N = len(anonymized_dataset)
    for category in raw_dataset[0]:
        if category == "income" or category == "index":
            continue
        current_DGH = DGHs[category]
        number_of_leaves = current_DGH.getNumberOfLeaves()
        for idx in range(N):
            cost += abs(((current_DGH.getNumberOfDescendantLeaves(anonymized_dataset[idx][category]) - 1)) / (number_of_leaves - 1))
    return cost        
            
def distance_LM(record, categories, DGHs):
    cost = 0.0
    for category in categories:
        if category == "income" or category == "index":
            continue
        current_DGH = DGHs[category]
        number_of_leaves = current_DGH.getNumberOfLeaves()
        cost += abs(((current_DGH.getNumberOfDescendantLeaves(record[category]) - 1)) / (number_of_leaves - 1))
    return cost

#Anonymizes record1 and record2 and returns the calculated LM cost of record2
def anonymized_records_LM(record1, record2, categories, DGHs):
    for category in categories:
        if category == "income" or category == "index":
            continue
        current_DGH = DGHs[category]
        least_common_ancestor = current_DGH.getCommonAncestor(record1[category], record2[category])
        record2[category] = least_common_ancestor
    return distance_LM(record2, categories, DGHs)
def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str, s: int):
    """ K-anonymization a dataset, given a set of DGHs and a k-anonymity param.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
        s (int): seed of the randomization function
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)    

    for i in range(len(raw_dataset)): ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i

    raw_dataset = np.array(raw_dataset)
    np.random.seed(s) ## to ensure consistency between runs
    np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize

    clusters = []

    D = len(raw_dataset)
    
    #TODO: START WRITING YOUR CODE HERE. Do not modify code in this function above this line.
    # Store your results in the list named "clusters". 
    # Order of the clusters is important. First cluster should be the first EC, second cluster second EC, ...

    #Clustering of random shuffled data set
    last_idx = 0
    for idx in range(k, len(raw_dataset) + k, k):
        if len(raw_dataset) - idx >= k + 1 and len(raw_dataset) - idx <= (2 * k) - 1:
            clusters.append(list(raw_dataset[last_idx:idx]))
            clusters.append(list(raw_dataset[idx:]))
            break
        clusters.append(list(raw_dataset[last_idx:idx]))
        last_idx = idx
    
    #Anonymization with
    raw_dataset = list(raw_dataset)
    #iteration = 0
    for cluster in clusters:
        for category in raw_dataset[0]:
            if category == "income" or category == "index":
                continue
            current_DGH = DGHs[category]
            least_common_ancestor = ""
            for idx in range(len(cluster)):
                for idx2 in range(len(cluster)):
                    if idx == idx2:
                        continue
                    if least_common_ancestor != "ANY":
                        least_common_ancestor = current_DGH.getCommonAncestor(cluster[idx][category], cluster[idx2][category])
                    cluster[idx][category] = least_common_ancestor
                    cluster[idx2][category] = least_common_ancestor
            #iteration += 1
            #print("Iteration #", iteration)
                

    

    # END OF STUDENT'S CODE. Do not modify code in this function below this line.

    anonymized_dataset = [None] * D

    for cluster in clusters:        #restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)



def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Clustering-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)
    
    N = len(raw_dataset)
    
    clusters = []
    
    for i in range(len(raw_dataset)): ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i
    
    D = np.array(raw_dataset.copy())
    np.random.shuffle(D)
    D = list(D)
    milestone = 0
    while len(D) > 2 * k:
        record1 = D.pop(0)
        cluster = [record1]
        if len(D) < 0.75 * len(raw_dataset) and milestone == 0:
            print("%75 left.")
            milestone += 1
        if len(D) < 0.50 * len(raw_dataset) and milestone == 1:
            print("%50 left.")
            milestone += 1
        if len(D) < 0.25 * len(raw_dataset) and milestone == 2:
            print("%25 left.")
            milestone += 1
        for idx in range(k - 1):
            record2 = {'index': 999999}
            record_distance = 999999
            for record in D:
                dst = anonymized_records_LM(record1.copy(), record.copy(), raw_dataset[0], DGHs)
                if record_distance == dst:
                    if record['index'] > record2['index']:
                        continue
                    else:
                        record_distance = dst
                        record2 = record
                if record_distance > dst:
                    record_distance = dst
                    record2 = record
            cluster.append(record2)
            D.remove(record2)
        
        for category in raw_dataset[0]:
            if category == "income" or category == "index":
                continue
            current_DGH = DGHs[category]
            least_common_ancestor = ""
            for idx in range(len(cluster)):
                for idx2 in range(len(cluster)):
                    if idx == idx2:
                        continue
                    if least_common_ancestor != "ANY":
                        least_common_ancestor = current_DGH.getCommonAncestor(cluster[idx][category], cluster[idx2][category])
                    cluster[idx][category] = least_common_ancestor
                    cluster[idx2][category] = least_common_ancestor
        clusters.append(cluster)
        
    for category in raw_dataset[0]:
        if category == "income" or category == "index":
            continue
        current_DGH = DGHs[category]
        least_common_ancestor = ""
        for idx in range(len(D)):
            for idx2 in range(len(D)):
                if idx == idx2:
                    continue
                if least_common_ancestor != "ANY":
                    least_common_ancestor = current_DGH.getCommonAncestor(D[idx][category], D[idx2][category])
                D[idx][category] = least_common_ancestor
                D[idx2][category] = least_common_ancestor
    clusters.append(D)


    # Finally, write dataset to a file
    anonymized_dataset = [None] * N

    for cluster in clusters:        #restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)



def bottomup_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Bottom up-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """

    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.

    # Finally, write dataset to a file
    #write_dataset(anonymized_dataset, output_file)


# Command line argument handling and calling of respective anonymizer:
if len(sys.argv) < 6:
    print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
    print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
    sys.exit(1)

algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'bottomup']:
    print("Invalid algorithm.")
    sys.exit(2)

start_time = datetime.datetime.now() ##
print(start_time) ##

dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])

function = eval(f"{algorithm}_anonymizer");
if function == random_anonymizer:
    if len(sys.argv) < 7:
        print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
        print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
        sys.exit(1)
        
    seed = int(sys.argv[6])
    function(raw_file, dgh_path, k, anonymized_file, seed)
else:    
    function(raw_file, dgh_path, k, anonymized_file)

cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
print (f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")

end_time = datetime.datetime.now() ##
print(end_time) ##
print(end_time - start_time)  ##

sys.exit(0)

# Sample usage:
# python3 code.py clustering DGHs/ adult-hw1.csv result.csv 300 5
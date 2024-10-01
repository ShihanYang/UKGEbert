"""
================================================================================
@In Project: UKGEbert
@File Name: bayesianet.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2024/09/18
@Update Date: 
@Version: 0.2.0
@Functions: 
    1. To create Bayesian Net and infer on it
    2. Notes:
================================================================================
"""
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # only ERROR messages

import numpy as np
import itertools
import matplotlib.pyplot as plt
import pickle

class Node:
    # Nodes are distinguished by name, and
    # Nodes with the same name are not allowed, and
    # Relation is also a node.
    name = ''
    parent = set()
    child = set()
    cptable = np.ndarray(shape=(0,))

    def __init__(self, name='nothing', cptable=[]):
        self.name = name
        self.parent = set()
        self.child = set()
        cpt = np.array(cptable)
        if len(cptable) == 0:
            p = np.random.random()
            self.cptable = np.array([p, 1 - p])
        else:
            self.cptable = cpt

    def neighbors(self):
        return ('Node %s:\n    parents: %s \n   children: %s') % \
               (self.name, [p.name for p in self.parent], [c.name for c in self.child])

    def CPT(self):
        numberOfParent = len(self.parent)
        if numberOfParent == 0:
            print('\n')
            print('================ CPT of Node %s =================' % self.name)
            print('               TRUE                   FALSE      ')
            print('--------------------------------------------------')
            print("   NoP.     %.12f  |  %.12f      " % (self.cptable[0], self.cptable[1]))
        else:
            combine = itertools.product({0, 1}, repeat=numberOfParent)  # number of parameters can be changed
            print('\n')
            print('================ CPT of Node %s =================' % self.name)
            print('     with parent nodes: %s' % [p.name for p in self.parent])
            print('================================================')
            print('               TRUE                   FALSE      ')
            print('--------------------------------------------------')
            for i in list(combine):
                rowName = list()
                for j in range(len(i)):
                    rowName.append(i[j])
                if np.array(self.cptable).ndim == 1:
                    p = self.cptable[0]
                else:
                    str_i = [str(k) for k in i]
                    m = int(''.join(str_i), 2)
                    p = self.cptable[m][0]
                print('%s    %.12f  |  %.12f        ' % (rowName, p, 1 - p))
                print('--------------------------------------------------')
        print('\n')
        return 2 ** len(self.parent)

    def __str__(self):
        return self.name


class Edge:
    # Edges are distinguished by (from name, label, to name), i.e.
    # Edges with the same label but different from or to node name are allowed.
    label = ''
    fro = Node('from')
    to = Node('to')

    def __init__(self, fro, to, label='edge'):
        self.label = label
        self.fro = fro
        self.to = to

    def __str__(self):
        return 'Edge: %s = (%s -> %s)' % (self.label, self.fro, self.to)


class Fact:
    head = Node()
    relation = Node()
    tail = Node()
    confidence = 1E-4  # 1E-12, The initial value must be sufficiently small.

    def __init__(self, head=Node(), relation=Node(), tail=Node(), confidence=1E-4):
        self.head = head
        self.relation = relation
        self.tail = tail
        self.confidence = confidence

    def factFromTriple(self, triple=('head', 'relation', 'tail')):
        head = Node(name=triple[0])
        tail = Node(name=triple[2])
        rel = Node(name=triple[1])
        f = Fact(head, rel, tail)
        return f

    def __str__(self):
        return 'Fact: %s, %s, %s, %.12f' % \
               (self.head.name, self.relation.name, self.tail.name, self.confidence)


class BN:
    mode = 't2t'  # h2h: head-> r <-tail  or  t2t: head<- r ->tail
    nodes = list()
    edges = list()
    __facts = set()
    net = [nodes, edges]

    def __init__(self, nodes=[], edges=[]):
        self.nodes = nodes
        self.edges = edges
        # build chain
        for e in edges:
            e.fro.child.add(e.to)
            e.to.parent.add(e.fro)
        self.net[0] = nodes
        self.net[1] = edges

    def namesOfNodes(self):
        names = set()
        for n in self.nodes:
            names.add(n.name)
        return names

    def nodeByName(self, name):
        for n in self.nodes:
            if n.name == name:
                return n

    def degreesOfNodes(self):
        degrees = dict()
        for n in self.nodes:
            degrees[n.name] = [0, 0]  # [out-degree, in-degree]
        for e in self.edges:
            fro = e.fro.name
            to = e.to.name
            degrees[fro][0] += 1
            degrees[to][1] += 1
        return degrees

    def chooseZeroIn(self, degrees):
        for n in degrees.keys():
            if degrees[n][1] == 0:
                return n
        return None

    def labelsOfEdges(self):
        labels = set()
        for e in self.edges:
            labels.add(e.label)
        return labels

    def pairsOfEdges(self):
        pairs = dict()
        for e in self.edges:
            pairs[e.label] = (e.fro, e.to)
        return pairs

    def allEdges(self):
        alledges = list()
        for e in self.edges:
            alledges.append((e.fro.name, e.label, e.to.name))
        return alledges

    def addNode(self, node=Node()):
        if node.name not in self.namesOfNodes():
            self.nodes.append(node)
            # print('Added a node successfully.')
        # else:
        #     print('Node "%s" is already existed.' % node)
        return node

    def hasCircle(self):
        retVal = False
        # topological sorted algorithm
        sorted = list()
        degrees = self.degreesOfNodes()
        # choose a node with zero in-degree
        for i in range(len(self.nodes)):
            zeroIn = self.chooseZeroIn(degrees)
            if zeroIn is not None:
                sorted.append(zeroIn)
                degrees.pop(zeroIn)
                # update degrees
                for e in self.edges:
                    if e.fro.name == zeroIn:
                        degrees[e.to.name][1] -= 1
        if len(sorted) != len(self.nodes):
            retVal = True
        return retVal

    def addEdge(self, edge=Edge(Node(), Node())):
        fro = edge.fro
        to = edge.to
        if fro.name in self.namesOfNodes():
            fro = self.nodeByName(fro.name)
        if to.name in self.namesOfNodes():
            to = self.nodeByName(to.name)
        if (fro, to) not in self.pairsOfEdges().values():  # only one edge between same two nodes
            if len(fro.child) < 2:  # relation node
                if edge.label not in self.pairsOfEdges().keys():
                    newedge = Edge(fro, to, label=edge.label)
                    self.edges.append(newedge)
                    if self.hasCircle():
                        self.edges.remove(newedge)
                        print('Failed adding edge: There is a circle when adding this edge.')
                    else:
                        # build chain
                        newedge.fro.child.add(to)
                        newedge.to.parent.add(fro)
                else:
                    print('Failed adding edge: Label name "%s" repeated.' % edge)
            else:
                print('Failed adding edge: A relation node has only two out-degrees.')
        else:
            print('Failed adding edge: Edge from %s to %s is already existed.' % (fro, to))
        return edge

    def createBNfromKG(self, file):
        numbersOfFacts = 0
        kg = list()
        with open(file, 'r', encoding='utf-8') as kgfile:
            lines = kgfile.readlines()
            rcount = 0
            for l in lines:
                rcount += 1
                if file[-3:] == 'csv':
                    line = [s.strip() for s in l.split(',')]
                else:
                    line = [s.strip() for s in l.split()]
                h = Node(line[0])
                if h.name not in self.namesOfNodes():
                    self.addNode(h)
                else:
                    h = self.nodeByName(h.name)
                t = Node(line[2])
                if t.name not in self.namesOfNodes():
                    self.addNode(t)
                else:
                    t = self.nodeByName(t.name)
                r = Node(line[1] + '~' + str(rcount))
                self.addNode(r)
                rh = Edge(r, h, label=r.name + '#2#' + h.name)
                self.addEdge(rh)
                rt = Edge(r, t, label=r.name + '#2#' + t.name)
                self.addEdge(rt)
                c = line[3]
                f = Fact(h, r, t, float(c))
                self.__facts.add(f)
                kg.append(f)
        return numbersOfFacts, kg

    def save(self, file):
        with open(file, 'wb') as bnf:
            pickle.dump(self, bnf)  # save BN object
        return file

    @staticmethod
    def load(file):
        with open(file, 'rb') as bnf:
            bayesianNet = pickle.load(bnf)
        return bayesianNet  # BN type

    def facts(self):
        if len(self.__facts) == 0:  # generate all facts from the Bayesian Network
            handled = []
            self.__facts = set()
            for e in self.edges:
                if e.fro not in handled:  # avoiding repeated facts
                    children = list(e.fro.child)
                    handled.append(e.fro)
                    if len(children) < 2:  # each relation node has just only two children
                        continue
                    else:
                        h = children[1]
                        r = e.fro
                        t = children[0]
                        c = np.random.random()  # TODO: read confidence from BERT model
                        f = Fact(h, r, t, c)
                        self.__facts.add(f)
        return self.__facts

    def findFactByTriple(self, head, relation, tail):
        for f in self.__facts:
            if f.head.name == head and \
                    f.relation.name == relation and \
                    f.tail.name == tail:
                return f
        return None

    def sumConfidenceOfFacts(self):
        s = 0
        for f in self.facts():
            s += f.confidence
        return s

    def numberOfFacts(self):
        self.facts()
        return len(self.__facts)

    def entities(self):
        self.facts()
        ent = set()
        for f in self.__facts:
            ent.add(f.head)
            ent.add(f.tail)
        return ent

    def namesOfEntities(self):
        names = set()
        ents = self.entities()
        for ent in ents:
            names.add(ent.name)
        return names

    def relations(self):
        self.facts()
        rel = set()
        for f in self.__facts:
            rel.add(f.relation)
        return rel

    def namesOfRelations(self):
        names = set()
        rels = self.relations()
        for rel in rels:
            names.add(rel.name)
        return names

    def typesOfRelations(self):
        types = set()
        for name in self.namesOfRelations():
            loc = name.find('~')
            if loc != -1:
                types.add(name[:loc])
            else:
                types.add(name)
        return types

    def numberOfEdges(self):
        return len(self.edges)

    def numberOfNodes(self):
        return len(self.nodes)

    def inferEntity(self, n=Node(), mode='t2t'):
        count = 1
        if mode == 't2t':
            for e in self.edges:
                if e.to == n:
                    count += 1
        else:  # 'h2h'
            for e in self.edges:
                if e.fro == n:
                    count += 1
        return count / self.numberOfEdges()

    def condition(self, node=Node()):  # computing p(t|r) or p(h|r)
        cpt = node.cptable
        t_sum = sum([t[0] for t in cpt])
        f_sum = sum([f[1] for f in cpt])
        cp = t_sum / (t_sum + f_sum)
        return cp

    def inferRfromFact(self, fact=Fact()):
        p = 1E-4
        h = fact.head
        t = fact.tail
        r = fact.relation
        c = fact.confidence
        nc = c / self.sumConfidenceOfFacts()  # confidence is not same as probability
        if self.mode == 't2t':
            # p(fact) = p(r)p(h|r)p(t|r)
            hr = self.condition(h)
            tr = self.condition(t)
            p = nc / (hr * tr)
            p = 1 / (1 + math.exp(-p))  # sigmoid function to normalize probability
        else:  # 'h2h
            # p(fact) = p(h)p(t)p(r|h,t)
            hp = self.inferEntity(h, mode='h2h')
            tp = self.inferEntity(t, mode='h2h')
            rht = nc / (hp * tp)  # TODO : for h2h mode, generateCPT function should be rewrite!
            p = 1 / (1 + math.exp(-rht))
            pass
        return p

    def generateCPT(self, mode='t2t'):
        # generate Conditional Probabilistic Table for all nodes
        cptables = dict()
        for e in self.entities():
            table = list()
            p = self.inferEntity(e)
            parents = e.parent
            for i in range(2 ** len(parents)):  # 根据1的个数计算概率
                binary_row = bin(i)
                one_number = binary_row.count('1')
                if one_number == 0:
                    row_p = 1 / (1 + len(parents))
                else:
                    row_p = p * (one_number / len(parents))
                row = [row_p, 1 - row_p]
                table.append(np.array(row))
            e.cptable = table
            cptables[e.name] = table
        for r in self.relations():  # find the fact by a relation
            for f in self.__facts:
                if f.relation == r:
                    p = self.inferRfromFact(f)
                    r.cptable = np.array([p, 1 - p])
                    cptables[r.name] = r.cptable
        return cptables

    def inferFact(self, fact=Fact()):
        # for 't2t' mode, p(fact) = p(r)p(h|r)p(t|r), and r is not in the set of relations
        confidence = fact.confidence
        r = fact.relation
        temp_name = r.name
        if r.name.find('~') != -1:
            temp_name = r.name[:r.name.find('~')]
        h = fact.head
        t = fact.tail
        case = (temp_name in self.typesOfRelations(),  # special handling for relationship
                h.name in self.namesOfEntities(),
                t.name in self.namesOfEntities())
        # print('Is (head, relation, tail) visible = (%s, %s, %s)' % (case[1], case[0], case[2]))
        if case == (1, 1, 1):  # trivial case
            for f in self.__facts:
                if f.head.name == h.name and f.tail.name == t.name and \
                        f.relation.name[:f.relation.name.find('~')] == r.name:
                    # confidence = f.confidence
                    confidence = f.relation.cptable[0] * self.condition(f.head) * self.condition(f.tail)
        if case == (1, 0, 0):
            for rel in self.relations():
                if rel.name[:rel.name.find('~')] == r.name:
                    r = rel
            confidence = r.cptable[0]
        if case == (1, 0, 1):
            t = self.nodeByName(t.name)
            for rel in self.relations():
                if rel.name[:rel.name.find('~')] != r.name:
                    continue
                if t in rel.child:
                    r = rel
            confidence = r.cptable[0] * self.condition(t)
        if case == (1, 1, 0):
            h = self.nodeByName(h.name)
            for rel in self.relations():
                if rel.name[:rel.name.find('~')] != r.name:
                    continue
                if h in rel.child:
                    r = rel
            confidence = r.cptable[0] * self.condition(h)
        if case == (0, 0, 0):  # invisible relationship 1 : all r,h,t are invisible
            # confidence = np.random.random()
            pe_sum = 0.0
            for e in self.entities():
                pe_sum += self.condition(e)
            ph = pt = pe_sum / (2 * len(self.entities()))
            pr_sum = 0.0
            for r in self.relations():
                pr_sum += r.cptable[0]
            pr = pr_sum / len(self.relations())
            confidence = pr * ph * pt
        if case == (0, 1, 1):  # invisible relationship 2 : only r is invisible
            h = self.nodeByName(h.name)
            t = self.nodeByName(t.name)
            ph = self.condition(h)
            pt = self.condition(t)
            pr_sum = list()
            for r in self.relations():
                if h in r.child:
                    pr_sum.append(r.cptable[0])
                if t in r.child:
                    pr_sum.append(r.cptable[0])
            pr = 1 - sum(pr_sum) / len(pr_sum) + math.prod(pr_sum)
            confidence = pr * ph * pt
        if case == (0, 1, 0):  # invisible relationship 3 : r and t are invisible
            h = self.nodeByName(h.name)
            ph = pt = self.condition(h) / 2
            pr_sum = list()
            for r in self.relations():
                if h in r.child:
                    pr_sum.append(r.cptable[0])
            pr = 1 - sum(pr_sum) / (2 * len(pr_sum)) + math.prod(pr_sum)
            confidence = pr * ph * pt
        if case == (0, 0, 1):  # invisible relationship 4 : r and h are invisible
            t = self.nodeByName(t.name)
            pt = ph = self.condition(t) / 2
            pr_sum = list()
            for r in self.relations():
                if t in r.child:
                    pr_sum.append(r.cptable[0])
            pr = 1 - sum(pr_sum) / (2 * len(pr_sum)) + math.prod(pr_sum)
            confidence = pr * ph * pt
        fact.confidence = confidence
        return fact.confidence

    def addFact(self, fact=Fact()):
        h = fact.head
        if h.name not in self.namesOfNodes():
            self.addNode(h)
        else:
            h = self.nodeByName(h.name)

        t = fact.tail
        if t.name not in self.namesOfNodes():
            self.addNode(t)
        else:
            t = self.nodeByName(t.name)

        r = fact.relation
        r_flag = False
        if r.name in self.typesOfRelations():
            for f in self.__facts:
                if f.head == h and f.tail == t and \
                        f.relation.name[:f.relation.name.find('~')] == r.name:
                    r = f.relation
                    r_flag = True
                    break
            if not r_flag:
                r.name = r.name + '~' + str(len(self.__facts) + 1)
                self.addNode(r)
                rh = Edge(r, h, label=r.name + '#2#' + h.name)
                self.addEdge(rh)
                rt = Edge(r, t, label=r.name + '#2#' + t.name)
                self.addEdge(rt)
        else:
            r.name = r.name + '~' + str(len(self.__facts) + 1)
            self.addNode(r)
            rh = Edge(r, h, label=r.name + '#2#' + h.name)
            self.addEdge(rh)
            rt = Edge(r, t, label=r.name + '#2#' + t.name)
            self.addEdge(rt)

        fact.head = h
        fact.tail = t
        fact.relation = r
        temp = self.findFactByTriple(h.name, r.name, t.name)
        if temp:
            fact = temp
        else:
            self.__facts.add(fact)
        self.propagation(mode='t2t')
        return fact

    def propagation(self, mode='t2t'):
        # when adding a fact into the knowledge graph, the CPT should be updated
        self.generateCPT(mode=mode)  # Todo : cost too much !

    def __str__(self):
        return '<Class>BN : (%d, %d)' % (len(self.nodes), len(self.edges))


if __name__ == "__main__":
    # Step 1. Build a new Bayesian Network object.
    bayesn = BN()

    # Step 2. Create a real Bayesian Network from a Knowledge Graph file.
    bayesn.createBNfromKG('../data/bio.csv')

    # Step 3. Generate Conditional Probabilistic Table for the Bayesian network.
    bayesn.generateCPT()  # After building the BN, you have to generate a CPT, or else they're random values.

    # Step 4. Check the Bayesian Network.
    # print(bayesn, 'has', bayesn.numberOfFacts(), 'facts.')

    # Step 5. Perform the approximate inference on the Bayesian Network.
    triple = ('cat', 'eat', 'apple')  # a triple denoted by (head, relation, tail)
    fact = Fact().factFromTriple(triple)  # build a fact object from the triple
    bayesn.inferFact(fact)  # infer the confidence of the triple as the prior probability of it

    # Step 6. After inferring, the invisible fact should be added into the KB!
    fact = bayesn.addFact(fact)  # adding a fact and updating cptable of its nodes; must reassign the fact!

    # Checking whether everything is correct or not?
    # print(bayesn, 'has', bayesn.numberOfFacts(), 'facts.')
    # print(triple, bayesn.inferFact(fact))
    # fact.head.CPT()
    # fact.tail.CPT()
    # fact.relation.CPT()

    # Step 7. Inferring invisible facts based on the Bayesian network.
    # scenario: To infer 'animal is pet', firstly, the confidence is very small. After adding facts, such
    #           as 'dog is animal' and 'dog is pet', 'fish is animal' and 'fish is pet', etc., the confidence
    #           of it should become bigger. And the more these facts added, the bigger confidence of it is
    #           inferred. Of course, there exists a up limit confidence that should be determined.
    invisible_fact = ('animal', 'is', 'pet')
    new_facts = [('dog', 'is', 'pet'), ('dog', 'is', 'animal'),
             ('fish', 'is', 'pet'), ('fish', 'is', 'animal'),
             ('rabit', 'is', 'pet'), ('rabit', 'is', 'animal'),
             ('duck', 'is', 'pet'), ('duck', 'is', 'animal'),
             ('kitty', 'is', 'pet'), ('kitty', 'is', 'animal'),
             ('puppy', 'is', 'pet'), ('puppy', 'is', 'animal'),
             ('turtle', 'is', 'pet'), ('turtle', 'is', 'animal')]

    confidences = list()
    fact0 = Fact().factFromTriple(invisible_fact)
    conf = bayesn.inferFact(fact0)
    confidences.append(conf)
    fact0 = bayesn.addFact(fact0)

    for f in new_facts:
        tempf = Fact().factFromTriple(f)
        bayesn.inferFact(tempf)
        tempf = bayesn.addFact(tempf)

        # print(tempf.tail, bayesn.condition(tempf.tail))
        # "As the number of similar fact instances increases, their probability will increase."
        # So the invisible facts are!
        # is this a kind of learning?
        fact0.relation.name = fact0.relation.name[:fact0.relation.name.find('~')]
        fact0conf = bayesn.inferFact(fact0)

        confidences.append(fact0conf)

    print(confidences)

    X = np.array([i for i in range(4, len(confidences))])  # drop the first 4 trival values
    Y = np.array([100*i for i in confidences[4:]])

    coefficients = np.polyfit(X, Y, 1)
    polynomial = np.poly1d(coefficients)
    Y_ = polynomial(X)  # linear fitting

    plt.plot(X, Y_)
    plt.scatter(X, Y, color='red', marker='o')
    plt.ylabel('$\times 10^{-2}$')
    plt.xlabel('Scale of factual instances')
    plt.show()


"""
================================================================================
@In Project: UKGEbert
@File Name: bayesianet.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2024/09/18
@Update Date: 
@Version: 0.1.0
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
            self.cptable = np.array([p, 1-p])
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
                print('%s    %.12f  |  %.12f        ' % (rowName, p, 1-p))
                print('--------------------------------------------------')
        print('\n')
        return None

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
    confidence = 1E-12

    def __init__(self, head=Node(), relation=Node(), tail=Node(), confidence=1E-12):
        self.head = head
        self.relation = relation
        self.tail = tail
        self.confidence = confidence

    def __str__(self):
        return 'Fact: %s, %s, %s, %.12f' % (self.head.name, self.relation.name, self.tail.name, self.confidence)


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
        if (edge.fro, edge.to) not in self.pairsOfEdges().values():  # only one edge between same two nodes
            if len(edge.fro.child) < 2:  # relation node
                if edge.label not in self.pairsOfEdges().keys():
                    self.edges.append(edge)
                    if self.hasCircle():
                        self.edges.remove(edge)
                        print('Failed adding edge: There is a circle when adding this edge.')
                    else:
                        # build chain
                        edge.fro.child.add(edge.to)
                        edge.to.parent.add(edge.fro)
                else:
                    print('Failed adding edge: Label name "%s" repeated.' % edge)
            else:
                print('Failed adding edge: A relation node has only two out-degrees.')
        else:
            print('Failed adding edge: Edge from %s to %s is already existed.' % (edge.fro, edge.to))
        return edge

    def createBNfromKG(self, file):
        numbersOfFacts = 0
        kg = list()
        with open(file, 'r', encoding='utf-8') as kgfile:
            lines = kgfile.readlines()
            rcount = 0
            for l in lines:
                rcount += 1
                line = [s.strip() for s in l.split(',')]
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
                r = Node(line[1]+str(rcount)); self.addNode(r)
                rh = Edge(r, h, label=r.name+'2'+h.name); self.addEdge(rh)
                rt = Edge(r, t, label=r.name+'2'+t.name); self.addEdge(rt)
                c = line[3]
                f = Fact(h, r, t, float(c))
                self.__facts.add(f)
                kg.append(f)
        return numbersOfFacts, kg

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

    def relations(self):
        self.facts()
        rel = set()
        for f in self.__facts:
            rel.add(f.relation)
        return rel

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
        p = 1E-12
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
            for i in range(2**len(parents)):  # 根据1的个数计算概率
                binary_row = bin(i)
                one_number = binary_row.count('1')
                if one_number == 0:
                    row_p = 1 / (1 + len(parents))
                else:
                    row_p = p * (one_number / len(parents))
                row = [row_p, 1-row_p]
                table.append(np.array(row))
            e.cptable = table
            cptables[e.name] = table
        for r in self.relations():  # find the fact by a relation
            for f in self.__facts:
                if f.relation == r:
                    p = self.inferRfromFact(f)
                    r.cptable = np.array([p, 1-p])
                    cptables[r.name] = r.cptable
        return cptables

    def inferFact(self, fact = Fact()):
        # for 't2t' mode, p(fact) = p(r)p(h|r)p(t|r), and r is not in the set of relations
        confidence = 1E-12
        r = fact.relation
        h = fact.head
        t = fact.tail
        if r in self.relations():
            confidence = r.cptable[0] * self.condition(h) * self.condition(t)
        elif h not in self.entities() and t not in self.entities():
            confidence = np.random.random()
        elif h in self.entities() and t in self.entities():
            confidence = 1 - 0  # TODO : computing confidence
        elif h in self.entities() and t not in self.entities():
            confidence = 1  # TODO : computing confidence
            pass
        elif h not in self.entities() and t in self.entities():
            pass  # TODO : computing confidence
        return confidence

    def propagation(self, mode='t2t', node=Node()):
        pass  # TODO

    def __str__(self):
        return '<Class>BN : (%d, %d)' % (len(self.nodes), len(self.edges))


if __name__ == "__main__":
    # Step 1. Build a new Bayesian Network object
    bayesn = BN()
    # Step 2. Create a real Bayesian Network from a Knowledge Graph file
    bayesn.createBNfromKG('../data/bio.csv')
    # Step 3. Generate Conditional Probabilistic Table for the Bayesian network
    bayesn.generateCPT()  # After building the BN, you have to generate a CPT, or else they're random values.
    # Step 4. Check the Bayesian Network
    print(bayesn, 'has facts', bayesn.numberOfFacts())
    for f in bayesn.facts():
        # f.head.CPT()
        f.relation.CPT()
        # f.tail.CPT()
        pass
    # Step 5. Perform the approximate inference on the Bayesian Network












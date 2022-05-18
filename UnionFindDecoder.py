from qecsim.model import Decoder, cli_description
from qecsim.model import DecoderFTP
from qecsim.models.rotatedplanar import RotatedPlanarCode
import math
from enum import Enum
from copy import deepcopy
import numpy as np

class edgeState(Enum):
    Unoccupied = 0
    HalfGrown = 0.5
    Grown = 1

class Vertex():
    def __init__(self, position):
        self.x = position[0]
        self.y = position[1]
        self.root = self
        self.edges = []
        self.rank = 1 # size of the cluster tree
        self.syndrome = 0 # is error on it
        self.belong = None # belong to which cluster
        self.child = [] # for spanning tree and peeling algorithm
        self.used = 0  # for spanning tree
    
    def getX(self):
        return self.x
    
    def getY(self):
        return self.y

    def position(self):
        return (self.x, self.y)
    
    def flip(self):
        self.syndrome = 1-self.syndrome
    
    def getChild(self):
        return self.child
    
    def addChild(self,c):
        self.child.append(c)
    
    def isSyndrome(self):
        return self.syndrome==1
    
    def rootbelong(self):
        return self.find().belong
    
    def addedge(self, e):
        self.edges.append(e)
    
    def calMidPosition(self, v):
        if isinstance(v, ImaginaryBoundaryVertex):
            return v.calMidPosition(self)
        return ((self.x+v.x)/2,(self.y+v.y)/2)    
    
    def plaquette2node(self,v): # to change not in this class
        return (v[0]+0.5,v[1]+0.5)
    
    def union(self, v):
        """union two vertex

        Args:
            v (Vertex): vertex to union with self

        Returns:
            int: 0 => union to self 
                 1 => union to v
        """
        # return to tell which rank bigger
        r1 = self.find()
        r2 = v.find()
        if r1.rank > r2.rank:
            r2.root = r1
            r1.rank += r2.rank
            return 0
        else:
            r1.root = r2
            r2.rank += r1.rank
            return 1

    def find(self):
        if self != self.root:
            self.root = self.root.find()
        return self.root
    
    def growHalf(self):
        """grow each edge self connect

        Returns:
            list: edge which become Grown
            list: new boundary vertex
        """
        fusionList = []
        newBoundaryList = []
        for edge in self.edges:
            if edge.growHalf():
                fusionList.append(edge)
                newBoundaryList.append(edge.anotherVertex(self))
        return fusionList, newBoundaryList
    
    def isInsideVertex(self):
        for e in self.edges:
            if e.state != edgeState.Grown:
                return False
        return True
        
    def __eq__(self, __o):
        if isinstance(__o, Vertex):
            return (self.x == __o.x and self.y == __o.y)
        return False
    
    def __hash__(self):
        return hash(self.position())
    
    def __str__(self):
        return "("+str(self.x)+","+str(self.y)+")" # +":"+str(self.syndrome)
    
    def __repr__(self):
        return str(self)
    
class ImaginaryBoundaryVertex(Vertex):
    """This class aims to handle odd syndrome in decoder
    * when even syndrome, we need a trivial boundary vertex
    * when odd syndrome, we need a nontrivial imaginary syndrome vertex
    * all boundary vertex will merge into **one** imaginary vertex
    For example, stabilizer plaquette types and indices on a 3 x 3 lattice:
                 -------
                /   Z   \
               |  (0,2)  |
               +---------+---------+-----
               |    X    |    Z    |  X  \
               |  (0,1)  |  (1,1)  |(2,1) |
               |         |         |     /
          -----+---------+---------+-----
         /  X  |    Z    |    X    |
        |(-1,0)|  (0,0)  |  (1,0)  |
         \     |         |         |
          -----x---------+---------+
               \         |    Z    |\
                \         \ (1,-1)/  \
                 \         -------    \
                  b                    b
                  (0,-1)
    if 'X' error happen, (-1,0) is the only syndrome, we need add syndrome (0,-1) so that the decoder
    can decode by (-1,0) to (0,-1) successfully
    * Notice: there are several imaginary vertex, when we need its position to decode, we choose the nearest
      boundary vertex
    """
    
    def __init__(self, position,t):
        super().__init__(position)
        self.type = t
    # overide
    def calMidPosition(self, v):
        # print(v,self.type)
        if self.type == 'X':
            if v.getY() == 0: # low line
                return (v.getX()+0.5,v.getY()-0.5)
            else: # high line
                return (v.getX()-0.5,v.getY()+0.5)
        if self.type == 'Z':
            if v.getX() == 0: # left line
                return (v.getX()-0.5,v.getY()-0.5)
            else: # right line
                return (v.getX()+0.5,v.getY()+0.5)
            
    def __str__(self):
        return "BoundaryVertex"    
        
class Edge(): 
    def __init__(self, a:Vertex, b:Vertex): # u->v
        self.u = a
        self.v = b
        self.state = edgeState.Unoccupied
        a.addedge(self)
        b.addedge(self)
        
    def fstVertex(self):
        return self.u
    def lstVertex(self):
        return self.v
    
    def anotherVertex(self, v):
        if self.v == v:
            return self.u
        if self.u == v:
            return self.v
        return None

    def growHalf(self):
        """grow half edge

        Returns:
            bool: is it become Grown
        """

        if self.state == edgeState.Unoccupied:
            self.state = edgeState.HalfGrown
            return False
        elif self.state == edgeState.HalfGrown:
            self.state = edgeState.Grown
            return True
        elif self.state == edgeState.Grown:
            return False
    
    def __eq__(self,o):
        if isinstance(o, Edge):
            return (self.u == o.u and self.v == o.v) or (self.u == o.v and self.v == o.u)
        return False
    
    def __str__(self):
        return str(self.u)+"->"+str(self.v)
    
    def __repr__(self):
        return str(self)
    
class DecoderGraph():
    def __init__(self,code,type):
        """buildGraph
        * build vertex according to surface code
        * build edge connected by vertex according to surface code
        Args:
            code (RotatedPlanarCode): Surface Code
            type (function): decoder 'Z' or 'X' error
        """
        self.vertexs = dict()
        self.edges = []
        self.code = code
        if isinstance(code,  RotatedPlanarCode):
            plaquettes = code._plaquette_indices
            # set vertexs
            for p in plaquettes:
                if type(p):
                    self.vertexs[p] = Vertex(p)
            # set edges some triky thing
            for pos in self.vertexs.keys():
                v = self.vertexs[pos]
                pos = v.position()
                x = v.getX()
                assitant = [(1,1),(1,-1),(-1,1),(-1,-1)]
                if (x+1)%2==0:
                    for ass in assitant:
                        otherPos = (ass[0]+pos[0], ass[1]+pos[1])
                        if otherPos in self.vertexs.keys():
                            other = self.vertexs[otherPos]
                            self.edges.append(Edge(v, other))
            self.createBoundaryVertex(type)
            # for i in self.vertexs.values():
            #     print(i.edges)
    def createBoundaryVertex(self, type):
        """Create imaginary vertex
        * See class:ImaginaryBoundaryVertex for detail
        Args:
            type (function): 'Z' or 'X' error lead to different special boundary vertex
        """
        if isinstance(self.code,  RotatedPlanarCode):
            dim = self.code.n_k_d[2]
            if type == self.code.is_x_plaquette:
                # print('x',self.edges)
                l1 = [(i,0) for i in range(-1,dim,2)]
                l2 = [(i, dim-2) for i in range(0, dim, 2)]
                # print(dim)
                imagVertex = ImaginaryBoundaryVertex((-666,-666), 'X')
                for i in l1:
                    self.edges.append(Edge(self.vertexs[i], imagVertex))
                for i in l2:
                    self.edges.append(Edge(self.vertexs[i], imagVertex))                    
                self.vertexs[((-666,-666))] = imagVertex
            if type == self.code.is_z_plaquette:
                l3 = [(0,i) for i in range(0,dim,2)]
                l4 = [(dim-2, i) for i in range(-1,dim,2)]
                imagVertex = ImaginaryBoundaryVertex((-666,-666), 'Z')
                for i in l3:
                    self.edges.append(Edge(self.vertexs[i], imagVertex))
                for i in l4:
                    self.edges.append(Edge(self.vertexs[i], imagVertex))    
                self.vertexs[((-666, -666))] = imagVertex
    
    def pos2vertex(self, pos):
        return self.vertexs[pos]
        
    def __str__(self):
        s = ''
        # for v in self.v:
        #     s += str(v)
        # s+='\n'
        for e in self.edges:
            s += str(e)+' '
        return s
        
class Cluster():
    """represent cluster in unionfind algorithm
    Attributes:
        treeroot: a representation vertex of the cluster
        edges: all Grown edges in this cluster
        boundaryList: represent boundary in unionfind algorithm
        parity: there are odd/even syndrome in this cluster
        alive: when this cluster merge by another cluster or be even, alive = False
    """
    def __init__(self, point):
        
        self.treeroot = point # vertex
        self.edges = [] # edge list
        self.boundaryList = [point]
        # self.support = dict()
        self.parity = 1
        self.alive = True
        point.belong = self
        point.flip() # set syndrome
        
        
    def isEven(self):
        return self.parity % 2 == 0
        
    def growHalf(self):
        self.treeroot = self.treeroot.find()
        edgeList = []
        newBoundaryList = []
        for boundary in self.boundaryList:
            edgelist, boundarylist = boundary.growHalf()
            edgeList += edgelist
            newBoundaryList += boundarylist
            # update boundary
        self.edges += edgeList # update edges belong to the cluster
        # print('edges', self.edges)
        self.boundaryList += newBoundaryList
        return edgeList
    
    def merge(self, c):
        r1: Vertex = self.treeroot.find()
        r2: Vertex = c.treeroot.find()
        c1 = r1.belong
        c2 = r2.belong
        flag = r1.union(r2)

        if flag == 0: # c->self
            c2.alive = False # update vertex belong
            r2.belong = c1
            c1.parity += c2.parity # update syndrome num
            # update boundary
            c1.boundaryList += c2.boundaryList
            c1.edges += c2.edges
            # c = self
            # print('parity', c1.parity)
            # print('belong', r1.rootbelong())
        else: # self->c
            # print(c1, '->', c2)
            c1.alive = False
            r1.belong = c2
            c2.parity += c1.parity
            # update boundary
            c2.boundaryList += c1.boundaryList
            c2.edges += c1.edges 
            # self = c
            # print('parity',c2.parity)
            # print('belong', r2.rootbelong())
            
    def cleanBoundary(self):
        self.boundaryList = list(set(self.boundaryList)) # to do!!!
        for i in range(len(self.boundaryList)-1,-1,-1):
            boundary = self.boundaryList[i]
            if boundary.isInsideVertex():
                self.boundaryList.remove(boundary)

                
    def cluster2Tree(self):
        edges = self.edges
        vertexDict = dict()
        for edge in edges:
            u = edge.fstVertex()
            v = edge.lstVertex()
            if vertexDict.__contains__(u):
                vertexDict[u].append(edge)
            else:
                vertexDict[u] = [edge]
            if vertexDict.__contains__(v):
                    vertexDict[v].append(edge)
            else:
                vertexDict[v] = [edge]        

        root = v
        # Build a tree by root
        buildList = [root]
        root.used = 1
        while len(buildList)>0:
            v = buildList.pop()
            for edge in vertexDict[v]:
                another = edge.anotherVertex(v)
                if another.used == 0:
                    buildList.append(another)
                    v.addChild(another)
                    another.used = 1
        # print(vertexDict)
        return root
    
    def __eq__(self, __o):
        if isinstance(__o, Cluster):
            return self.treeroot == __o.treeroot
        return False
    

    
    def __str__(self):
        return str(self.treeroot)+" parity:"+str(self.parity)+" alive "+str(self.alive)
    
    def __repr__(self):
        return str(self)

@cli_description('unionfind decoder')
class UnionFindDecoder(Decoder):
    def decode(self, code, syndrome, debug=False, **kwargs):
        self.debug = debug
        self.code = code
        self.syndrome = syndrome
        recovery = []
        syndromex = []
        syndromez = []
        syndromeSet = code.syndrome_to_plaquette_indices(syndrome)
        for syn in syndromeSet:
            if(code.is_x_plaquette(syn)):
                syndromex.append(syn)
            if(code.is_z_plaquette(syn)):
                syndromez.append(syn)
        graph = self.buildGraph(code, code.is_z_plaquette)
        recovery += self._decoder(syndromez, graph)
        graph = self.buildGraph(code, code.is_x_plaquette)
        recovery += self._decoder(syndromex, graph)
        # print(syndromex)

        return np.array(recovery) 
    
    def _decoder(self, syndromeList, graph):
        """main procedure to decoder

        Args:
            syndromeList (list): list contains syndromes in (x, y)
            graph (Graph): abstarct graph built by self.buildGraph
            
        Returns:
            recoveryList (list): list contains recovery in (x, y)
        """
        clusterList = self.initialCluster(syndromeList, graph)
        evenClusterList = []
        while len(clusterList)>0: # odd cluster not empty      
            if self.debug:      
                print('all',clusterList)
                print('even', evenClusterList)
            # clusterList.sort(key=lambda x: len(x.boundaryList)) # emm
            fusionList = []
            for cluster in clusterList:
                if cluster.isEven()==False:
                    fusionList += cluster.growHalf()
            # print(fusionList)
            for fusionEdge in fusionList:
                # print(fusionEdge)
                u = fusionEdge.fstVertex()
                v = fusionEdge.lstVertex()
                if u.rootbelong() == None: # new vertex do not belong any cluster
                    u.belong = v.rootbelong
                    u.root = v
                    continue
                if v.rootbelong() == None:
                    v.root = u
                    v.belong = u.rootbelong
                    continue
                if u.find() != v.find():
                    u.rootbelong().merge(v.rootbelong())
                    
            for i in range(len(clusterList)-1,-1,-1):
                cluster = clusterList[i]
                if cluster.alive is False:
                    clusterList.remove(cluster)
                elif cluster.isEven(): # find even cluster
                    evenClusterList.append(cluster)
                    clusterList.remove(cluster)

            for i in range(len(evenClusterList)-1,-1,-1):
                cluster = evenClusterList[i]
            # for cluster in evenClusterList:
                if cluster.alive and cluster.isEven() is False:
                    clusterList.append(cluster)
                    evenClusterList.remove(cluster)
                if cluster.alive == False:
                    evenClusterList.remove(cluster)
            
            for cluster in clusterList:  # remove not boundary vertex in boundaryList
                cluster.cleanBoundary()

        if self.debug:
            print('eveneven', evenClusterList)
        treeList = self.spanningTree(evenClusterList)
        recoveryList = self.peeling(treeList)
        return self.pos2binary(recoveryList)
                
    def buildGraph(self, code,type):
        """build abstract graph by given suface code 

        Args:
            code: surface code
            type: a function  'Z' error or 'X' error

        Returns:
            Graph: abstract graph
        """
        G = DecoderGraph(code,type)
        return G
     
    def initialCluster(self, syndromeList, graph:DecoderGraph):
        """initial cluster induce by syndrome in graph"""
        clusterList = []
        for syn in syndromeList:
            clusterList.append(Cluster(graph.pos2vertex(syn)))         
        if len(syndromeList) %2 == 1: # odd syndrome->boudary vertex nontrivial
            clusterList.append(Cluster(graph.pos2vertex((-666,-666))))
        return clusterList
    
    def spanningTree(self, clusterList):
        """generate spanning tree

        Args:
            clusterList (list): list of even syndrome cluster

        Returns:
            list: list of spanning tree according to each cluster
        """
        # print(clusterList)
        treeList = []
        for cluster in clusterList:
            if cluster.alive:
            # cluster.cleanBoundary() # to do !!!
            # print(cluster.edges, cluster.boundaryList)
            # print(cluster)
                tree = cluster.cluster2Tree()
                treeList.append(tree)
        
        # for tree in treeList: # test Depth First Search
        #     temp = [tree]
        #     while len(temp)>0:
        #         node = temp.pop()
        #         print('par',node,'child',node.getChild())
        #         for child in node.getChild():
        #             temp.append(child)
        return treeList

    def peeling(self, treeList):
        """peeling algorithm to decode
        * see https://arxiv.org/abs/1703.01517 for details
        Args:
            treeList (list): list of spanning tree

        Returns:
            list: list of recovery in format [(x_1,y_1),...,]
        """
        recoveryList = []
        for tree in treeList:
            recoveryOne = []
            self._peeling_(tree,recoveryOne)
            recoveryList += recoveryOne
        return recoveryList
    
    def _peeling(self, treeNode:Vertex, l:list):
        """subroute of peeling, a recursive implementation

        Args:
            treeNode (Vertex): current node
            l (list): list of node need recovery
        """
        for child in treeNode.getChild():
            self._peeling(child, l)
        for child in treeNode.getChild():
            if child.isSyndrome():
                treeNode.flip()
                # print('flip', treeNode)
                mid = treeNode.calMidPosition(child)
                # l.append(treeNode)
                l.append(treeNode.plaquette2node(mid))
                
    def _peeling_(self, root:Vertex, l:list):
        """subroute of peeling, an iterative implementation

        Args:
            treeNode (Vertex): current node
            l (list): list of node need recovery
        """
        levelTraversal = []
        usedList = [root]
        while len(usedList)>0:
            node = usedList.pop()
            levelTraversal.append(node)
            for child in node.getChild():
                usedList.append(child)
        # print(levelTraversal)
        while len(levelTraversal)>0:
            node = levelTraversal.pop()
            # print(node)
            for child in node.getChild():
                if child.isSyndrome():
                    node.flip()
                    mid = node.calMidPosition(child)
                    l.append(node.plaquette2node(mid))
        
    def pos2binary(self, recoveryList):
        """position list to binary list

        Args:
            recoveryList (list): [(x_1,y_1)]

        Returns:
            list: [0,1,1,0,...]
        """
        length = int(self.code.stabilizers.T.shape[0]/2)
        dim = int(math.sqrt(length))
        l = [0]*length
        for r in recoveryList:
            l[int(r[1])*dim+int(r[0])] = 1
        # print(recoveryList)
        return l

    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        return 'UnionFind Decoder'

    def __repr__(self):
        return "UnionFind Decoder"

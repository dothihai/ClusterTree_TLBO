import numpy as np
import networkx as nx
import numpy as np
import time
from Algorithm import *
from itertools import combinations
class Instance(object):
	"""docstring for Insta"""
	def __init__(self, name):
		self.name = name
		filename = 'test/'+name
		self.resultname = 'result/_'+name+'_.gen'
		self.resultnameopt = 'result/_'+name+'_.opt'

		with open(filename) as f:
			if "NODE_COORD_SECTION" not in f.read():
				self.use = False
				return
			else:
				self.use = True

		with open(filename) as f:
			v_n = 0 # so vertex
			c_n = 0 # so cluster
			self.cluster = [] # cac cluster
			coor = [] # coor
			line = f.readline()
			while "NODE_COORD_SECTION" not in line:
				line = f.readline()
				if 'DIMENSION' in line:
					v_n = int(line.split()[-1])
				if 'NUMBER_OF_CLUSTERS' in line:
					c_n = int(line.split()[-1])
			self.distance = np.empty([v_n,v_n])		
			for x in range(v_n):
				a,b = f.readline().split()[-2:]
				coor.append(np.array([float(a),float(b)])) #float 
			f.readline()
			f.readline()
			for x in range(c_n):
				cluster = [int(i) for i in f.readline().split()[1:-1]]
				self.cluster.append(cluster)
			for x in range(v_n):
				for y in range(v_n):
					self.distance[x,y]=np.linalg.norm(coor[x]-coor[y])
			# print(self.distance)
		
		self.dim = v_n
		
	def evaluate(self,learner):
		prufer,err = self.decode(learner.subjects,learner.node)
		if err:
			learner.fitness =  1e20
			return
		tree = self.decode_tree(prufer) # actually not prufer tho, well whatever :)
		# cost1 = 0
		# for x1,x2 in combinations(range(self.dim),2):
		# 	route = list(nx.all_simple_paths(tree, source=x1, target=x2))[0]
		# 	route_cost = 0
		# 	for x in range(len(route)-1):
		# 		n1 = route[x]
		# 		n2 = route[x+1]
		# 		route_cost = route_cost + self.distance[n1,n2]
		# 	cost1 = cost1 + route_cost
		# learner.fitness = cost1
		cost = 0
		w = np.ones(self.dim)
		k = list(nx.bfs_predecessors(tree,0))
		l = dict(k)
		# print(k)
		h = [a for a,b in k][::-1]
		for z in h:
			w[l[z]] = w[l[z]]+w[z]
			cost = cost + w[z] * (self.dim - w[z]) * self.distance[z,l[z]]

		# cost2 = 0
		# ctree = tree.copy()
		# edge_list = ctree.edges
		# for edge in edge_list:
		# 	a,b = edge
		# 	ctree.remove_edge(a,b)
		# 	w = len(list(nx.dfs_preorder_nodes(ctree,a)))
		# 	cost2 = cost2 + w * (self.dim - w) * self.distance[a,b]
		# 	ctree.add_edge(a,b)
		learner.fitness = cost
		# print(len(tree.edges))
		return cost

	def decode(self,subjects,node): 
		# node array of array int 
		# subject (float) de dung tlbo
		seq = []
		start = 0
		err = False
		for x in node[:-1]:
			if len(x) == 0:
				seq.append([])
				continue
			pos = np.argsort(subjects[start:start+len(x)])
			seq.append([x[i] for i in pos])
			start = start + len(x)
		seq.append(node[-1])
		# for x in range(len(self.cluster)):
		# 	if seq[-1][x] >= len(self.cluster[x]):
		# 		err = True
		return seq,err

	def decode_tree(self,seq):
		tree = nx.Graph()
		start = 0
		vl = []
		for x in range(len(seq)-2):
			if len(self.cluster[x]) == 2:
				tree.add_edge(self.cluster[x][0],self.cluster[x][1])
				continue
			if len(self.cluster[x]) == 1:
				continue
			t_t = nx.from_prufer_sequence(seq[x])
			for e in list(t_t.edges):
				tree.add_edge(self.cluster[x][e[0]],self.cluster[x][e[1]])
				# print(self.cluster[x][e[0]],self.cluster[x][e[1]])
			# vl.append(self.cluster[x][seq[-c_n+x]])
		t_t = nx.from_prufer_sequence(seq[-2])
		mseq = seq[-1]
		for e in list(t_t.edges):
			tree.add_edge(self.cluster[e[0]][mseq[e[0]]],self.cluster[e[1]][mseq[e[1]]])
		# print(tree.edges)
		# print([self.distance[a,b] for a,b in tree.edges])
		return tree

	def init(self,pop):
		population = []
		for x in range(pop):
			c_l = [len(x) for x in self.cluster] # do dai tung cluster
			c_n = len(self.cluster)
			node = []
			for c in c_l:
				if c < 3:
					node.append([])
					continue
				node.append(list(np.random.randint(c,size=c-2)))


			node.append(list(np.random.randint(c_n,size=c_n-2)))
			# node.append(list(np.random.randint(min(c_l),size=c_n)))
			# node.append(list(np.random.randint(max(c_l),size=c_n)))
			node.append([np.random.randint(c) for c in c_l])

			subjects = np.random.rand(self.dim-2)
			ind = Learner(subjects,node)
			self.evaluate(ind)
			population.append(ind)
		return population
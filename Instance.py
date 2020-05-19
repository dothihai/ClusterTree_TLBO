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
		self.coor = []
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
				self.coor.append(np.array([float(a),float(b)])) #float 
			f.readline()
			f.readline()
			for x in range(c_n):
				cluster = [int(i) for i in f.readline().split()[1:-1]]
				self.cluster.append(cluster)
			for x in range(v_n):
				for y in range(v_n):
					self.distance[x,y]=np.linalg.norm(self.coor[x]-self.coor[y])
			# print(self.distance)
		
		self.dim = v_n
		
	def evaluate(self,learner):
		prufer,err = self.decode(learner.subjects)
		# print(prufer)
		learner.seq = prufer
		if err:
			learner.fitness =  1e20
			return
		tree = self.decode_tree(prufer) # actually not prufer tho, well whatever :)
		learner.tree = tree
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

		learner.fitness = cost
		# print(len(tree.edges))
		return cost

	def decode(self,subjects): 
		# node array of array int 
		# subject (float) de dung tlbo
		seq = []
		start = 0
		err = False
		c_l = [len(x) for x in self.cluster] # do dai tung cluster
		c_n = len(self.cluster)
		for x in c_l:
			if x < 3:
				seq.append([])
				continue
			sequence = subjects[start:start+x-2] 
			seq.append([int(np.floor(i*x)) for i in sequence])
			start += (x - 2)

		sequence = subjects[start:start+c_n-2] 
		seq.append([int(np.floor(i*c_n)) for i in sequence])
		start += c_n - 2
		sequence = subjects[start:] 
		seq.append([int(np.floor(sequence[i]*c_l[i])) for i in range(c_n)])
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
			length = c_n*2-2
			for c in c_l:
				if c < 3:
					continue
				length += c-2

			subjects = np.random.rand(length)
			ind = Learner(subjects)
			self.evaluate(ind)
			population.append(ind)
		return population
from Instance import *
from os import listdir
from functools import reduce
from operator import attrgetter
import time
listfile = listdir("test")
pop= 100
gen = 500
#listfile = ['6st70-2x3.clt']
#listfile = ['10eil76.clt']
def secondsToStr(t):
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(t*1000,),1000,60,60])

for file in listfile:
	start_time = time.time()

	instance = Instance(file)

	if not instance.use:
		continue
	print(instance.name)
	population = instance.init(pop)
	result_file = open(instance.resultname,'w')
	result_file.write("Generation \t"+instance.name+'\n')
	for x in range(gen):
		teacher = min(population, key=attrgetter('fitness'))
		#write to file
		result_file.write(str(x)+' \t'+str(teacher.fitness)+'\n')
		diff = np.random.rand()*(teacher.subjects - np.random.randint(1,3)*np.mean([x.subjects for x in population],axis=0))
		# buoc 1
		for y in range(pop):
			new = Learner(population[y].subjects+diff*np.random.rand(),population[y].node)
			instance.evaluate(new)
			if new.fitness < population[y].fitness:
				population[y] = new
		#buoc 2
		for y in range(pop):
			l2 = population[np.random.randint(pop)]
			while population[y]==l2:
				l2 = population[np.random.randint(pop)]
			if population[y].fitness < l2.fitness:
				new = Learner(population[y].subjects-(l2.subjects-population[y].subjects)*np.random.rand(),population[y].node)
				instance.evaluate(new)
			else:
				new = Learner(population[y].subjects+(l2.subjects-population[y].subjects)*np.random.rand(),population[y].node)
				instance.evaluate(new)
			if new.fitness < population[y].fitness:
				population[y] = new
	
	result_file.close()
	best = min(population, key=attrgetter('fitness'))
	with open(instance.resultnameopt,'w') as f:
		f.write('filename: '+instance.name+'\n')
		f.write("Seed: 0\n")
		f.write("Fitness: "+str(best.fitness)+'\n')
		f.write("Time: "+secondsToStr(time.time()-start_time)+'\n')

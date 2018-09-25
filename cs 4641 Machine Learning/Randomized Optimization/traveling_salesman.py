import sys
import os
import time
import csv

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.TwoPointCrossOver as TwoPointCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.TravelingSalesmanEvaluationFunction as TravelingSalesmanEvaluationFunction
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
from array import array
import sys




def fixedIterationMultipleSaveTraining(trainer, ef, max_iter, samples, repeat, output, N, parameters):
	run_time = [0] * samples
	call_count = [0] * samples
	value = [0] * samples
	iteration = range(max_iter/samples, max_iter + max_iter/samples, max_iter/samples)
	for n in range(0, repeat):
		start_time = time.time()
		ef.clearCount()
		for i in range(0, samples):
			fit =  FixedIterationTrainer(trainer[n], max_iter/samples)
			fit.train()
			run_time[i] += time.time() - start_time
			call_count[i] += ef.getTotalCalls()
			value[i] += (ef.value(trainer[n].getOptimal()))
			sys.stdout.write("=")
			sys.stdout.flush()

	call_count = [x/repeat for x in call_count]
	run_time = [x/repeat for x in run_time]
	value = [x/repeat for x in value]
	for i in range(0, samples):
		output.append([N, iteration[i], run_time[i], call_count[i], value[i]] + parameters)
	sys.stdout.write("\n")



csvfile_rhc = open("out/traveling_salesman_RHC.csv", "w+", 0)
writer_rhc = csv.writer(csvfile_rhc, delimiter=",")

csvfile_sa = open("out/traveling_salesman_SA.csv", "w+", 0)
writer_sa = csv.writer(csvfile_sa, delimiter=",")

csvfile_ga = open("out/traveling_salesman_GA.csv", "w+", 0)
writer_ga = csv.writer(csvfile_ga, delimiter=",")

csvfile_mimic = open("out/traveling_salesman_MIMIC.csv", "w+", 0)
writer_mimic = csv.writer(csvfile_mimic, delimiter=",")



"""
Commandline parameter(s):
   none
"""
runs = 10

scf = SingleCrossOver()
dcf = TwoPointCrossOver()
ucf = UniformCrossOver()

t = 1E12
cooling = 0.999
gap_list = [None, None, "traveling"]
populationSize = 2000
toMate = 0.75
toMutate = 0.125
samples = 500
tokeep = 0.2

n=5


for N  in [20,40,60,80]:
	rhc_data = []
	sa_data = []
	ga_data = []
	mimic_data = []
	print("--------", time.strftime("%X"))
	print("N:", N)
	random = Random()
	points = [[0 for x in xrange(2)] for x in xrange(N)]
	for i in range(0, len(points)):
	    points[i][0] = random.nextDouble()
	    points[i][1] = random.nextDouble()

	ef = TravelingSalesmanRouteEvaluationFunction(points)
	gap_list[1] = TravelingSalesmanCrossOver(ef)
	odd = DiscretePermutationDistribution(N)
	nf = SwapNeighbor()
	mf = SwapMutation()

	hcp = GenericHillClimbingProblem(ef, odd, nf)
	gap = GenericGeneticAlgorithmProblem(ef, odd, mf, gap_list[1])

	print("RHC")
	rhc = []
	for i in range(0, n):
		rhc.append(RandomizedHillClimbing(hcp))
	fixedIterationMultipleSaveTraining(rhc, ef, 10000, 10, 5, rhc_data, N, [])
	for data in rhc_data:
		writer_rhc.writerow(data)

	print("SA")
	for t in [1E2, 1E5, 1E10]:
		for cooling in [0.6, 0.8, 0.95]:
			sa = []
			for i in range(0, n):
				sa.append(SimulatedAnnealing(t, cooling, hcp))
			fixedIterationMultipleSaveTraining(sa, ef, 10000, 10, 5, sa_data, N, [t, cooling])
	for data in sa_data:
		writer_sa.writerow(data)

	print("GA")
	for populationSize in [20, 200, 300]:
		for toMate in [0.5, 0.75, 1]:
			ga = []
			for i in range(0, n):
				ga.append(StandardGeneticAlgorithm(populationSize, int(toMate*populationSize), int(toMutate*populationSize), gap))
			fixedIterationMultipleSaveTraining(ga, ef, 10000, 10, 5, ga_data, N, [populationSize, toMate, toMutate, gap_list[2]])
	for data in ga_data:
		writer_ga.writerow(data)

	ef = TravelingSalesmanSortEvaluationFunction(points);
	fill = [N] * N
	ranges = array('i', fill)
	odd = DiscreteUniformDistribution(ranges);
	df = DiscreteDependencyTree(.1, ranges);
	pop = GenericProbabilisticOptimizationProblem(ef, odd, df);

	print("MIMIC")
	for samples in [50 ,100,100]:
		for tokeep in [0.1, 0.2, 0.5]:
			mimic = []
			for i in range(0, n):
				mimic.append(MIMIC(samples, int(tokeep*samples), pop))
			fixedIterationMultipleSaveTraining(mimic, ef, 5000, 5, 5, mimic_data, N, [samples, tokeep])
	for data in mimic_data:
		writer_mimic.writerow(data)
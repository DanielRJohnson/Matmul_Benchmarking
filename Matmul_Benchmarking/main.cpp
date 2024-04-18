#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <functional>
#include <chrono>
#include "matrix.h"
#include "algorithms.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

static std::vector<int> parseOption(int argc, char** argv, std::string option) {
	/*
	* Searches for a command line option (e.g. "--test") and retrieves a vector
	* of integer options until it runs into a non-int or the end.
	*/
	std::vector<int> values;
	bool found = false;
	for (int i = 0; i < argc; i++) {
		if (found) {
			int value = atoi(argv[i]);
			if (value != 0) {
				values.push_back(value);
			}
			else {
				break;
			}
		}
		if (argv[i] == option) {
			found = true;
		}
	}
	return values;
}

static void usageAndDie() {
	/*
	* Prints usage and exits with error status 1
	*/
	std::cout << "Usage: exe <vanilla|parfor|tiled|DAC_notemp|DAC_temp|strassens> <n_runs>" <<
		" --ps <list of num_procs> --Ns <list of matrix sizes> --params <list of alg params>\n";
	std::exit(1);
}

int main(int argc, char** argv)
{
	/*
	* Runs data generation for a given algorithm over the following cross product of inputs:
	*		thread limits X matrix sizes X algorithm parameters
	* 
	* Prints a CSV-style output for data analysis.
	*/

	if (argc < 3) {
		usageAndDie();
	}

	// get options from command line arguments
	std::string algorithm = argv[1];
	int n_runs = atoi(argv[2]);
	std::vector<int> ps = parseOption(argc, argv, "--ps");
	std::vector<int> Ns = parseOption(argc, argv, "--Ns");
	std::vector<int> params = parseOption(argc, argv, "--params");

	if ( ps.size() < 1 ) {
		std::cerr << "Need to specify number of processors!" << std::endl;
		usageAndDie();
	}

	if ( Ns.size() < 1 ) {
		std::cerr << "Need to specify matrix dimensions!" << std::endl;
		usageAndDie();
	}

	if (algorithm == "vanilla" || algorithm == "parfor") {
		if ( params.size() != 0 ) {
			std::cerr << "Algorithm \"" + algorithm + "\" requires 0 parameters!" << std::endl;
			usageAndDie();
		}
		params.push_back(1); // hack so that a later for loop runs once instead of zero times
	} else if ( params.size() < 1 ) {
		std::cerr << "Algorithm \"" + algorithm + "\" requires parameters!" << std::endl;
		usageAndDie();
	}

	auto chosenAlgorithm = (algorithm == "vanilla") ? vanillaMatmul :
		(algorithm == "parfor") ? parforMatmul :
		(algorithm == "tiled") ? tiledMatmul :
		(algorithm == "DAC_notemp") ? noTempDACMatmul :
		(algorithm == "DAC_temp") ? tempDACMatmul :
		(algorithm == "strassens") ? strassensMatmul : nullptr;

	if (chosenAlgorithm == nullptr) {
		std::cerr << "Invalid algorithm: " << algorithm << std::endl;
		usageAndDie();
	}

	auto eqchSize = (algorithm != "tiled" || params[0] <= 4) ? 4 : params[0] * 2;

	// check that the written algorithm is correct
	Matrix eqCheckA = generateMatrix(eqchSize, true);
	Matrix eqCheckB = generateMatrix(eqchSize, true);
	Matrix eqCheckC1 = generateMatrix(eqchSize, false);
	Matrix eqCheckC2 = generateMatrix(eqchSize, false);
	vanillaMatmul(eqCheckA, eqCheckB, eqCheckC1, 1);
	omp_set_dynamic(0); // always use the exact number of threads we say to
	omp_set_num_threads(1);
	chosenAlgorithm(eqCheckA, eqCheckB, eqCheckC2, params[0]);

	if (!matrixEquality(eqCheckC1, eqCheckC2)) {
		std::cerr << "Algorithm produced an incorrect answer.\n";
		eqCheckC1.print(std::cerr);
		eqCheckC2.print(std::cerr);
		std::exit(1);
	}

	// begin data generation 
	std::vector<float> data_times;
	std::vector<int> data_ps;
	std::vector<int> data_Ns;
	std::vector<int> data_params;

	for (auto th: ps) {
		omp_set_num_threads(th); // limit to th threads
		//std::cout << th;
		for (const auto& size: Ns) {
			Matrix A = generateMatrix(size, true); // random matrix
			Matrix B = generateMatrix(size, true); // random matrix
			Matrix C = generateMatrix(size, false); // empty matrix
			for (auto param : params) {
				for (int i = 0; i < n_runs; i++) {
					// time one algorithm run and output information to data vectors
					auto t1 = high_resolution_clock::now();
					chosenAlgorithm(A, B, C, param);
					auto t2 = high_resolution_clock::now();
					duration<double, std::milli> ms_double = t2 - t1;
					data_times.push_back(ms_double.count());
					data_ps.push_back(th);
					data_Ns.push_back(size);
					data_params.push_back(param);
				}
			}
			//std::cout << ".";
		}
	}

	// print data
	std::cout << "threads,N,runtime(ms),param\n";
	for (int i = 0; i < data_times.size(); i++) {
		std::cout << data_ps[i] << ", " << data_Ns[i] << ", " << data_times[i] << ", " << data_params[i] << "\n";
	}

	return 0;
}


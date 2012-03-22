/*
 *       Filename:  caveboy.c
 *    Description:  Video Sequence Analyzer
 *         Author:  Javier Santacruz (), <francisco.santacruz@estudiante.uam.es>
 *         Author:  Alberto Montes (), <alberto.montes@estudiante.uam.es>
 *
 *        Created:  20/02/12 16:55:39
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <errno.h>
#include <getopt.h>

#include <mpi.h>

#include "perceptron.h"	

/*  Handy macros */
#ifndef printerr
  #define printerr(...) fprintf(stderr, __VA_ARGS__)
#endif

#ifndef printdbg
 #ifdef DEBUG
   #define printdbg(...) printerr(__VA_ARGS__)
 #else
   #define printdbg(...) (void)0;
#endif
#endif

#ifndef FALSE
 #define FALSE 0
#endif
#ifndef TRUE
 #define TRUE !FALSE
#endif

const char * usage = "Usage: %s PATDIR [-irhoam N] [-wez FILE] [-vnt]\n"\
					  "\t-i N\tInput neurons [from pattern]\n"\
					  "\t-h N\tHidden layer neurons [=input]\n"\
					  "\t-o N\tOutput neurons [from pattern]\n"\
					  "\t-a N\tLearning rate [0.001]\n"\
					  "\t-m N\tMax epoch [2000]\n"\
					  "\t-f N\tVideo fps [10]\n"\
					  "\t-r N\tNeuron radio [0.1]\n"\
					  "\t-e FILE\tLog training ECM [error.dat]\n"\
					  "\t-w FILE\tWeights file (wil be written if training) [weights.dat]\n"\
					  "\t-z FILE\tSave/Read training info [tinfo.dat]\n"\
					  "\t-n\tNormalize values [NO]\n"\
					  "\t-t\tTraining [NO]\n"\
					  "\t-v\tVerbose mode [NO]\n";

/* Share perceptron parameters */
int broadcast_sizes(int * nin, int * nh, int * nout, int * npats, int rank) {
	int vals[4];

	if( rank == 0 ){
		vals[0] = *nin;
		vals[1] = *nh;
		vals[2] = *nout;
		vals[3] = *npats;
	}

	MPI_Bcast(vals, 4, MPI_INT, 0, MPI_COMM_WORLD);

	if( rank != 0 ) {
		/* Receive and set perceptron sizes */
		*nin  = vals[0];
		*nh   = vals[1];
		*nout = vals[2];
		*npats = vals[3];
	}

	return TRUE;
}

/* Set weights from master to all other nodes */
int broadcast_weights(perceptron per){

	MPI_Bcast(&(per->w[0][0][0]), per->w_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	set_cube_pointers(per->w, &(per->w[0][0][0]), per->n);

	return TRUE;
}


/* Gets deltas from all processors and computes new weights */
int compute_new_weights(perceptron per, int rank){

	MPI_Reduce( MPI_IN_PLACE,              /* Use current root deltas */
			        &(per->dw[0][0][0]),   /* Current deltas */
			        per->w_size,            /* weights cube sizes */
			        MPI_DOUBLE,
			        MPI_SUM,
			        0,
			        MPI_COMM_WORLD);

	/* Sum weights on root */
	if( rank == 0 ){
		size_t n = per->w_size;
		double * w = &(per->w[0][0][0]);
		double * dw = &(per->dw[0][0][0]);

		while( n-- )
			*(w + n) += *(dw + n);
	}

	/* Fix w and dw pointers */
	set_cube_pointers(per->w, &(per->w[0][0][0]), per->n);
	set_cube_pointers(per->dw, &(per->dw[0][0][0]), per->n);

	return TRUE;
}

/* Get the result codes from all processors */
int return_codes(int * codes, int n, int * allcodes, int nall){

	MPI_Gather(codes, n, MPI_DOUBLE, allcodes, nall, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	return TRUE;
}

/* Splits patterns within all available nodes.
 * Creates a new patternset to store the patterns.
 *
 * @param pset Original patternset with right sizes. (And data if root)
 * @param newpset Uninitialized patternset by reference.
 * @param rank Processor rank.
 * @param size Comm size. */
int distribute_patterns(patternset pset, patternset * newpset, int rank, int size) {

	/* Each process receives an equal slice of patterns.
	 * Root gets the slice plus the spare patterns. */
	int i = 0;
	int patsize  = pset->ni + 1;
	int partsize = pset->npats / size;
	int rootsize = partsize + (pset->npats % size);
	int partsize_units = partsize * patsize;
	int rootsize_units = rootsize * patsize;

	/* How many doubles are to be read */
	/* How far away is the next value */
	int * scounts = (int *) malloc (size * sizeof(int));
	int * strides = (int *) malloc (size * sizeof(int));

	/* Create new patternset */
	patternset_create(newpset,
			rank == 0 ? rootsize : partsize,
			patsize,
			pset->npsets);
	patternset_init(newpset);

	/* Set how many doubles are to be sent/received */
	scounts[0] = rootsize_units;
	strides[0] = 0;

	for(i = 1; i < sizes; ++i) {
		scounts[i] = partsize_units;
		strides[i] = rootsize_units + (i-1) * partsize_units;
	}

	/* Send patterns to each new perceptron on net */
	MPI_Scatterv(pset->input_raw, scounts, strides, MPI_DOUBLE,
			newpset->input_raw, scounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	/* Adapt memory pointers on new perceptrons */
	for(i = 0; i < scounts[rank]; ++i)
		newpset->input[i] = &(newpset->input_raw[i * scounts[rank] ]);

	return TRUE;
}

/* Distributes codes within all processors.
 * Follows the same scheme as distribute_patterns.
 * To be used after distribute_patterns.
 * newpset is supposed to be already initialized.
 *
 * @param pset Original initialized pset with right sizes (Data only on root).
 * @param newpset Initialized patternset with distribute_patterns.
 * @param rank Processor rank.
 * @param size Comm size  */
int distribute_codes(patternset pset, patternset newpset, int rank, int size) {
	/* Each process receives an equal slice of codes.
	 * Root gets the slice plus the spare codes. */
	int partsize = pset->npats / size;
	int rootsize = partsize + (pset->npats % size);
	int i = 0;

	/* Set how many doubles are to be sent/received */
	scounts[0] = rootsize;
	strides[0] = 0;

	for(i = 1; i < sizes; ++i) {
		scounts[i] = partsize;
		strides[i] = rootsize + (i-1) * partsize;
	}

	/* Send patterns to each new perceptron on net */
	MPI_Scatterv(pset->codes, scounts, strides, MPI_UNSIGNED_LONG,
			newpset->codes, scounts[rank], MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

	return TRUE;
}

/* Train the network in a parallelized manner
 *
 * All available patterns are split in different subsets and delivered within nodes.
 * Each nodes computes only a subset of patterns and calculates deltas for those.
 * Deltas are sent back to master and joined with current weights.
 * New wegiths are sent to all nodes and a new epoch begins.
 *
 * 1. Distribute patterns and codes to each node.
 * 2. For each epoch:
 * 		1. Master broadcasts current neuron weights within nodes.
 * 		2. All nodes compute next epoch for given patterns.
 * 		3. All nodes returns computed deltas to master.
 * 		4. Master applies deltas to current weights.
 * 		5. [Future] Master calibrates workload per node
 * 		      and adjusts pattern distribution.
 */
int training(perceptron per, patternset pset, int max_epoch, double alpha,
		char * weights_path, int rank, int size;
		char * tinfo_path, char * error_path){
	FILE * error_file = NULL;

	/* Save obtained training info. Basically, patternsets names */
	if( rank == 0 )
		patternset_print_traininginfo(pset, tinfo_path);

	for(epoch = 0; epoch < max_epoch; ++epoch) {

		/* Send root weights in each processor */
		broadcast_weights(pset);

		/* Compute backpropagation for each available pattern.
		 * Do not update weights, get only deltas in per->dw */
		for(pat = 0; pat < pset->npats; ++pat)
			perceptron_backpropagation_raw(per, pset->input[pat], pset->codes[pat], alpha, 0);

		/* Get deltas back from each processor and compute new weights */
		compute_new_weights(per, rank) */

		/* TODO: get performance stats and adjust pattern distribution */
	}

	/* Master saves current weights  */
	if( rank == 0 )
		perceptron_printpath(per, weights_path);

	return TRUE;
}

/* Exploiting network in a parallelized manner.
 *
 * The testing phase uses an already trained net
 * to classificate new unknown presented datterns.
 *
 * 1. Distribute patterns withing nodes.
 * 2. Pass all nodes through the net in each node and save the output.
 * 3. Return obtained codes for each node.
 * 4. Join all codes into a single result.
 */
int testing(perceptron per, patternset pset, int npatsall, double radio, char * weights_path, char * tinfo_path) {
	size_t pat = 0, n = 0, matches = 0;
	int chosen = 0;
	int * codes = NULL;
	int * allcodes = NULL;
	double min = 1.0 - radio;

	/* Testing phase uses an already trained net to try to clasificate
	 * new unknown patterns.
	 *
	 * 1. Recuperate patterns code-name associations from training.
	 * 2. Recuperate trained perceptron weights.
	 * 3. Pass each pattern through the net and save the output.
	 * 4. Calculate stats.
	 */

	if( rank == 0 ){
		/* Recuperate training patterns info. */
		if( patternset_read_traininginfo(pset, tinfo_path) == FALSE)
			return FALSE;

		/* Recuperate trained net. Read weights. */
		if( perceptron_readpath(&per, weights_path) == FALSE )
			return FALSE;
	}

	/* Distribute patterns within nodes for testing
	 * We get a prepared pset with this
	 * distribute_patterns_testing() */

	/* Broadcast trained net weights from master
	 * broadcast_weights */

	if( pset->npats <= 0 )
		return FALSE;

	codes = (int *) malloc (sizeof(int) * pset->npats);


	/* Alloc extra codes for root, which has to deal with the result */
	if( rank == 0 )
		allcodes = (int *) malloc (npatsall * sizeof(int));

	/* Use trained net per each input pattern
	 * and put the output in a vector */
	for(pat = 0; pat < pset->npats; ++pat){
		perceptron_feedforward(per, pset->input[pat]);

		/* Find the most excited neuron
		 *
		 * Undecidible (not found, -1) if:
		 * - Most excited neuron doesn't get close enough (> min)
		 * - More than 1 neuron has been activated (matches > 1).
		 */
		matches = 0;
		chosen = -1;
		for(n = 0; n < pset->no; ++n){
			if( per->net[2][n] > min ) {
				chosen = n;
				++matches;

				/* 1 active neuron at most */
				if( matches > 1 ) {
					chosen = -1;
					break;
				}
			}
		}

		codes[pat] = chosen;
	}

	/* Send all codes and then join them */
	 return_codes(codes, pset->npats, allcodes, npatsall);

	 /* Print results */
	 if( rank == 0 ){
		 for(pat = 0; pat < npatsall; ++pat) {
			 printf("Output for %i: %i\n", pat, allcodes[pat]);
		 }
	 }

	 if( codes != NULL )
		free(codes);

	 if( allcodes != NULL )
		free(allcodes);

	return TRUE;
}

/* Cleans all program resources */
void clean_resources(perceptron * per, patternset * pset){
	if( per != NULL )
		perceptron_free(per);

	if( pset != NULL )
		patternset_free(pset);

	MPI_Finalize();
}

int main(int argc, char * argv[] ) {
	double alpha = 0.001,
		    radio = 0.1;

	int nin = 1, nh = 1, nout = 1,
		max_epoch = 2000, fps = 10,
		verbose = FALSE, errorflag = FALSE,
		do_training = FALSE, normalize = FALSE,
		mpi_rank = 0, mpi_size = 0, npats = 0;

	char c = 0,
		 * dir_path = NULL,
		 * errorlog_path = "error.dat",
		 * weights_path = "weights.dat",
		 * traininginfo_path = "tinfo.dat";

	perceptron per = NULL;
	patternset pset = NULL,  /* General patternset */
			   wpset = NULL; /* Working patternset */

	/* Check arguments */
	if( argc > 20 ) {
		printerr("ERROR: Too many arguments\n");
		printerr(usage, argv[0]);
		exit(EXIT_FAILURE);

	} else if( argc < 2 ) {
		printerr("ERROR: Patternset directory needed\n");
		printerr(usage, argv[0]);
		exit(EXIT_FAILURE);
	}

	/* Parse MPI arguments
	 * MPI should remove it's own arguments from argv */
	MPI_Init(&argc, &argv);

	/* Read arguments */
	dir_path = argv[1];  /* Patternset directory */

	while( (c = getopt(argc, argv, "vnti:h:o:a:e:m:w:")) != EOF ){
		switch(c) {
			/* Args */
			case 'i': nin = atoi(optarg); break;    /* Input neurons */
			case 'h': nh = atoi(optarg); break;     /* Hidden layer neurons */
			case 'o': nout = atoi(optarg); break;   /* Output neurons */
			case 'm': max_epoch = atoi(optarg); break;   /* Max epoch */
			case 'f': fps = atoi(optarg); break;   /* Video fps */
			case 'a': alpha = atof(optarg); break;  /* Learning rate */
			case 'r': radio = atof(optarg); break;  /* Neuron radio */
			case 'e': errorlog_path = optarg; break;   /* Error logging */
			case 'w': weights_path = optarg; break;   /* Weights */
			case 'z': traininginfo_path = optarg; break;   /* Training names */

			/* Flags */
			case 't': do_training = 1; break;      /* Train net */
			case 'v': verbose = 1; break;       /* Verbose */
			case 'n': normalize = 1; break;     /* Previous data normalization */

			default: printerr("WARNING: Unkown arg '-%c'\n", c);
					 errorflag = TRUE;
					 break;
		}
	}

	if( errorflag ){
		clean_resources(&per, &pset);
		exit(EXIT_FAILURE);
	}

	/* Check arguments read */
	if( nout <= 0 || nin <= 0 || nh <= 0 || alpha <= 0 ){
		printerr("ERROR: Invalid net sizes or learning rate\n");
		clean_resources(&per, &pset);
		exit(EXIT_FAILURE);
	}

	/* Get some info about MPI */
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

	/* Root loads the samples and gets the perceptron size */
	if( rank == 0 ){

		/* Read patterns.
		 * BEWARE IO operations and lots of memory being allocated here.  */
		if( patternset_readpath(&pset, dir_path) == FALSE ) {
			printerr("ERROR: Failed to load patternset: '%s'\n", dir_path);
			clean_resources(&per, &pset);
			exit(EXIT_FAILURE);
		}

		/* Set net sizes from patterns if not provided by user */
		if( nin == 1)
			nin = pset->ni;
		if( nout == 1)
			nout = pset->no;
		if( nh == 1 )
			nh = pset->no * 2;
	}

	/* Receive perceptron sizes nin, nout, nh
	 * If we're rank 0, broadcast sizes, if we're not, receive them. */
	 broadcast_sizes(&nin, &nout, &nh, &npats, rank);

	 /* Create an empty patternset with the right sizes but no memory */
	if( rank != 0 )
		patternset_create(&pset, npats, nin, nout);

	/* Create perceptron */
	if( perceptron_create(&per, nin, nh, nout) == 0 ) {
		printerr("ERROR: Couldn't create perceptron.\n");
		clean_resources(&per, &pset);
		exit(EXIT_FAILURE);
	}

	/* Distribute patterns to all processors */
	distribute_patterns(pset, &wpset, rank, size);

	if( do_training ) {
		/* Also distribute output codes when training, they're necessary */
		distribute_codes(pset, wpset, rank, size);
		training(per, pset, max_epoch, alpha, weights_path, traininginfo_path, errorlog_path);
	} else {
		testing(per, pset, pset->npats, radio, weights_path, traininginfo_path);
	}

	clean_resources(&per, &pset);

	return EXIT_SUCCESS;
}

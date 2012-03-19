/*
 *       Filename:  caveboy.c
 *    Description:  Video Sequence Analyzer
 *         Author:  Javier Santacruz (), <francisco.santacruz@estudiante.uam.es>
 *         Author:  Alberto Montes (), <alberto.montes@estudiante.uam.es>
 *
 *        Created:  20/02/12 16:55:39
 */

/* #define _POSIX_C_SOURCE 1; *//* Activate the use of getopt */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <errno.h>
#include <getopt.h>

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

int training(perceptron per, patternset pset, int max_epoch, double alpha,
		char * weights_path, char * tinfo_path, char * error_path){
	FILE * error_file = NULL;

	/* Save obtained training info  */
	patternset_print_traininginfo(pset, tinfo_path);

	if( error_path != NULL )
		if( (error_file = fopen(error_path, "w")) == NULL )
			printerr("ERROR: Couldn't open file %s; error %s\n",
					error_path, strerror(errno));

	/* Train and print epoch info to outfile */
	perceptron_trainingprint(per, pset, alpha, 0, max_epoch, error_file);

	/* Save weights  */
	perceptron_printpath(per, weights_path);

	/* Close errorlog file */
	if( error_file != NULL )
		if( fclose(error_file) == EOF ) {
			printerr("ERROR: Couldn't close file %s; error %s\n",
					error_path, strerror(errno));
		}

	return TRUE;
}

int testing(perceptron per, patternset pset, double radio, char * weights_path, char * tinfo_path){
	size_t pat = 0, n = 0, matches = 0;
	int chosen = 0;
	int * codes = NULL;
	double min = 1.0 - radio;

	/* Testing phase uses an already trained net to try to clasificate
	 * new unknown patterns.
	 *
	 * 1. Recuperate patterns code-name associations from training.
	 * 2. Recuperate trained perceptron weights.
	 * 3. Pass each pattern through the net and save the output.
	 * 4. Calculate stats.
	 */

	/* Recuperate training patterns info. */
		if( patternset_read_traininginfo(pset, tinfo_path) == FALSE)
			return FALSE;

		/* Recuperate trained net. Read weights. */
		if( perceptron_readpath(&per, weights_path) == FALSE )
			return FALSE;

		if( pset->npats <= 0 )
			return FALSE;

		/* Alloc space for all patterns output */
		codes = (int *) malloc (sizeof(int) * pset->npats);

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
					chosen = n
					++matches;

					/* 1 active neuron at most */
					if( matches > 1 ) {
						chosen = -1;
						break;
					}
				}
			}

			codes[pat] = chosen;

			printf("\nRaw output\n");
			for(n = 0; n < pset->no; ++n){
				printf("%f\t", per->net[2][n]);
			}
			printf("%d\t", codes[pat]);
		}

		/* Output to stdout to check */
		for(pat = 0; pat < pset->npats; ++pat){
			printf("Pattern %zd ", pat);
			if( codes[pat] != -1 )
				printf("recognized as %s (%d)\n",
						pset->names[codes[pat]], codes[pat]);
			else
				printf("is undecidible\n");
		}

		return TRUE;
}

int main(int argc, char * argv[] ) {
	double alpha = 0.001,
		    radio = 0.1;

	int nin = 1, nh = 1, nout = 1,
		max_epoch = 2000, fps = 10,
		verbose = FALSE,
		do_training = FALSE,
		normalize = FALSE;

	char c = 0,
		 * dir_path = NULL,
		 * errorlog_path = "error.dat",
		 * weights_path = "weights.dat",
		 * traininginfo_path = "tinfo.dat";

	perceptron per = NULL;
	patternset pset = NULL;

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

			default: if(verbose) printerr("WARNING: Unkown arg '-%c'\n", c);
		}
	}

	/* Check arguments read */
	if( nout <= 0 || nin <= 0 || nh <= 0 || alpha <= 0 ){
		printerr("ERROR: Invalid net sizes or learning rate\n");
		exit(EXIT_FAILURE);
	}

	/* Read training patterns.
	 * BEWARE IO operations and lots of memory being allocated here.  */
	if( patternset_readpath(&pset, dir_path) == FALSE ) {
		printerr("ERROR: Failed to load patternset: '%s'\n", dir_path);
		patternset_free(&pset);
		exit(EXIT_FAILURE);
	}

	/* Set net sizes from patterns if not provided by user */
	if( nin == 1)
		nin = pset->ni;
	if( nout == 1)
		nout = pset->no;
	if( nh == 1 )
		nh = pset->no * 2;

	/* Create perceptron */
	if( perceptron_create(&per, nin, nh, nout) == 0 ) {
		printerr("ERROR: Couldn't create perceptron.\n");
		perceptron_free(&per);
		patternset_free(&pset);
		exit(EXIT_FAILURE);
	}

	if( do_training )
		training(per, pset, max_epoch, alpha, weights_path, traininginfo_path, errorlog_path);
	else
		testing(per, pset, radio, weights_path, traininginfo_path);

	patternset_free(&pset);
	perceptron_free(&per);

	return EXIT_SUCCESS;
}

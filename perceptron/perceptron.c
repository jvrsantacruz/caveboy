/*
 *       Filename:  perceptron.c
 *    Description:  Multilayer Perceptron implementation
 *         Author:  Javier Santacruz <francisco.santacruz@estudiante.uam.es>
 *
 */

#include "perceptron.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <time.h>

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

static double perceptron_mean_square_error(double * actual, size_t code, int n){
	int i = 0;
	double dif,sum; 
	dif = sum = 0.0;

	if( n == 0 ){
		printerr("perceptron_mean_square_error: Lenght 0\n");
		return 0;
	}

	for(; i < n; ++i) {
		dif = actual[i] - (ISACTIVE(i, code));
		sum += dif * dif;
	}

	return 0.5 * (sum / n);
}

/** 
 * Perceptron default weight initializer.
 *
 * @return Random double value in the interval [-1,1]
 */
static double perceptron_rand(){
	return rand()/(RAND_MAX/2.0) - 1;
}

/**
 * Perceptron transition function (bipolarsigmoid)
 *
 * @param Value
 * @return Filtered value between [-1,1]
 */
static double perceptron_bipolarsigmoid(double x){
	return 2.0/(1 + exp(-x)) - 1;
}

/**
 * Perceptron transition function prima (bipolarsigmoid)
 *
 * @param Value
 * @return Filtered value between [-1,1]
 */
static double perceptron_bipolarsigmoid_prima(double x){
	double fx = perceptron_bipolarsigmoid(x);
	return 0.5 * (1 + fx) * (1 - fx);
}


/** 
 * Computes forward feeding for perceptron given a pattern.
 *
 * @param per Initialized perceptron
 * @param pat Initialized pattern
 * @return 0 if unsuccessful, 1 otherwise
 */
int perceptron_feedforward(perceptron per, pattern pat){
	int i, j, k, n;
	double sum = 0;

	/* Set input pattern */
	per->net[0] = pat;

	/* Calculate output layer value */

	/* For the input and hidden layers */
	for(i = 0; i < 2; ++i)
		/* For all neurons in layer (+ bias) */
		for(j = 0; j < per->n[i] + 1; ++j)
			/* For all neurons in next layer (no bias) */
			for(k = 0; k < per->n[i+1]; ++k) {
				/* Sum all neurons (j=n[i]) in layer by its
				 * weight w[j][k] to a certain neuron (k) */

				/* sum = perceptron_weighted_sum(per->net[i], per->w[i], k, per->n[i] + 1); */
				sum = 0;
				n = per->n[i] + 1;
				while( n-- )
					sum += per->net[i][n] * per->w[i][n][k];

				per->net[i+1][k] = perceptron_bipolarsigmoid(sum);
			}

	return 1;
}

/**
 * Computes backpropagation for a perceptron and a given pattern.
 * Raw version which perform the calculations and 
 *
 * @param per Initialized perceptron
 * @param pat Initialized pattern
 * @param code Active neuron in output pattern
 * @param lrate Learning rate 
 * @return 0 if unsuccessful, 1 otherwise
 */
int perceptron_backpropagation_raw(perceptron per, pattern pat, size_t code,
		double lrate){
	int n = 0, i = 0, j = 0, k = 0, err = 1;
	double Dj_in, Dj, sum;

	/* Rename temp delta vectors */
	double ** d = per->d,     /* Deltas */
		   ** rin = per->rw, /* Raw neuron inputs */
		   *** dw = per->dw;  /* Weight Deltas */

	/* Set input layer values 
	 * We just make net[0] to point to the pattern so we don't have to copy all
	 * of it each time. */
	per->net[0] = pat;

	/* Compute feed forward */

	/* For the input and hidden layers */
	for(i = 0; i < 2; ++i)
		/* For all neurons in layer (+ bias) */
		for(j = 0; j < per->n[i] + 1; ++j)
			/* For all neurons in next layer (no bias) */
			for(k = 0; k < per->n[i+1]; ++k) {
				/* Sum all neurons (j=n[i]) in layer by its
				 * weight w[j][k] to a certain neuron (k) */

				/* sum = perceptron_weighted_sum(per->net[i], per->w[i], k, per->n[i] + 1); */
				sum = 0;
				n = per->n[i] + 1;
				while( n-- )
					sum += per->net[i][n] * per->w[i][n][k];

				rin[i][k] = sum;  /* Save raw input to be used later */
				per->net[i+1][k] = perceptron_bipolarsigmoid(sum);
			}


	/* Calculate output layer (i = 2) backpropagation */
	for(k = 0; k < per->n[2]; ++k){
		/* Get the already computed Yk in */
		/* Yk_in = perceptron_weighted_sum(per->net[1], per->w[1], k, per->n[1] + 1); */
		/* Yk_in = rin[1][k]; */

		/* Calculate dk against desired output (neuron k should match the code) */
		d[1][k] = (ISACTIVE(k, code) - per->net[2][k]) * perceptron_bipolarsigmoid_prima(rin[1][k]);

		/* Calculate weight deltas for all weights to this neuron from the previous layer */
		for(j = 0; j < per->n[1] + 1; ++j)
			dw[1][j][k] = lrate * d[1][k] * per->net[1][j];
	}

	/* Calculate hidden layer (i = 1) backpropagation */
	for(j = 0; j < per->n[1]; ++j){
		/* Get the already computed Zj_in */ 
		/* Zj_in = perceptron_weighted_sum(per->net[0], per->w[0], j, per->n[0] + 1); */
		/* Zj_in = rin[0][j]; */

		/* Calculate Dj_in based on output layer deltas and the hidden layer weights */
		Dj_in = 0;
		for(k = 0; k < per->n[2]; ++k)
			Dj_in += d[1][k] * per->w[1][j][k];

		/* Calculate delta */
		Dj = Dj_in * perceptron_bipolarsigmoid_prima(rin[0][j]);

		/* Calculate weight deltas for all weights to this neuron from the previous layer */
		for(i = 0; i < per->n[0] + 1; ++i)
			dw[0][i][j] = lrate * Dj * per->net[0][i];
	}

	/* Update weights */
	for(i = 0; i < per->w_size; ++i)
		per->w_raw[i] += per->dw_raw[i];

	/* For the weighted layers
	for(i = 0; i < 2; ++i)
		* For each neuron (+ bias) *
		for(j = 0; j < per->n[i] + 1; ++j)
			* To all neurons in the next layer *
			for(k = 0; k < per->n[i + 1]; ++k)
				per->w[i][j][k] += dw[i][j][k]; */

	return err;
}

/**
 * Computes backpropagation for a perceptron and a given pattern.
 *
 * @param per Initialized perceptron
 * @param pat Initialized pattern
 * @param code Output pattern
 * @param lrate Learning rate 
 * @return 0 if unsuccessful, 1 otherwise
 */
int perceptron_backpropagation(perceptron per, pattern pat, size_t code, double lrate){
	int ret;
	double ** d = per->d;     /* Neuron deltas */
	double *** dw = per->dw;   /* Weight deltas */

	/* Check allocations */
	if( d == NULL || dw == NULL ){
		ret = 0;
		printerr("perceptron_backpropagation: Couldn't alloc space for deltas.\n");
	} else {
		ret = perceptron_backpropagation_raw(per, pat, code, lrate);
	}

	return ret;
}

/**
 * Reads perceptron weights and structure from stream.
 *
 * NI NH NO
 * I Layer weights (length NH)
 * I Layer weights
 * ... NI + 1 ...
 * H Layer weights (length NO)
 * H Layer weights
 * ... NH + 1 ...
 *
 * @param per Uninitialized perceptron by reference.
 * @param perfile Input stream.
 * @return 0 if unsuccessful, 1 otherwise
 */
int perceptron_read(perceptron * per_ptr, FILE * perfile){
	int i, j, k, ret;
	double tmp = 0;
	perceptron per = NULL;

	/*  Grab header */
	ret = fscanf(perfile, "%i %i %i", &i, &j, &k);

	/*  Check header */
	if( ret != 3 ){
		printerr("perceptron_read: Couldn't get file header.\n");
		return 0;
	}

	if( i <= 0 || j <= 0 || k <= 0 ) {
		printerr("perceptron_read: Incorrect header values input: %i, hidden: %i, output: %i\n", 
				i, j, k);
		return 0;
	}

	/* Initialize perceptron */
	if( perceptron_create(per_ptr, i, j, k) == 0 ){
		printerr("perceptron_read: Couldn't initialize perceptron correctly.\n");
		return 0;
	}

	per = *per_ptr;

	/*  For input and hidden layers */
	for(i = 0; i < 2; ++i)

		/* For all neuron in layer + bias weights */
		for(j = 0; j < per->n[i] + 1; ++j)

			/* For all neuron (no bias) in next layer */
			for(k = 0; k < per->n[i + 1]; ++k){
				ret = fscanf(perfile, "%lf", &tmp);

				if( ret != 1 ) {
					printerr("perceptron_read: Couldn't finish reading hidden layer values.");
					return 0;
				}

				per->w[i][j][k] = tmp;
			}

	return 1;
}

/** 
 * Reads perceptron weights and structure from file.
 *
 * @param per Uninitialized perceptron by reference.
 * @param perfile_path Path to output file.
 * @return 0 if unsuccessful, 1 otherwise
 */
int perceptron_readpath(perceptron * per_ptr, const char * perfile_path) {

	int ret = 0;
	FILE * perfile = NULL;

	perfile	= fopen( perfile_path, "r" );

	if ( perfile == NULL ) {
		printerr("perceptron_readpath: Couldn't open file '%s'; %s\n", perfile_path, strerror(errno));
		return 0;
	}

	ret = perceptron_read(per_ptr, perfile);

	if( fclose(perfile) == EOF ) {
		printerr("perceptron_readpath: Couldn't close file '%s'; %s\n", perfile_path, strerror(errno));
		return 0;
	}

	return ret;
}

/**
 * Dumps perceptron weights and structure to stream.
 *
 * NI NH NO
 * I Layer weights (length NH)
 * I Layer weights
 * ... NI + 1 ...
 * H Layer weights (length NO)
 * H Layer weights
 * ... NH + 1 ...
 *
 * @param per Initialized perceptron.
 * @param perfile Output stream.
 * @return 0 if unsuccessful, 1 otherwise
 */
int perceptron_print(perceptron per, FILE * perfile){
	int i, j, k;

	fprintf(perfile, "%i %i %i\n", 
			per->n[0], per->n[1], per->n[2]);

	/*  Weights for input and hidden layer */
	for(i = 0; i < 2; ++i){

		/*  For all neurons + bias weights in layer */
		for(j = 0; j < per->n[i] + 1; ++j){

			/* For all neurons weights (no bias) in next layer */
			for(k = 0; k < per->n[i + 1]; ++k){
				fprintf(perfile, "%lf ", per->w[i][j][k]);
			}

			fprintf(perfile, "\n");
		}
	}

	return 1;
}

/**
 * Dumps perceptron weights and structure to file.
 *
 * @param per Initialized perceptron.
 * @param perfile_path Path to file.
 * @return 0 if unsuccessful, 1 otherwise
 */
int perceptron_printpath(perceptron per, const char * perfile_path){
	int ret = 0;
	FILE * perfile = NULL;

	perfile	= fopen( perfile_path, "w" );

	if ( perfile == NULL ) {
		printerr("perceptron_printpath: Couldn't open file '%s'; %s\n", perfile_path, strerror(errno));
		return 0;
	}

	ret = perceptron_print(per, perfile);

	if( fclose(perfile) == EOF ) {
		printerr("perceptron_printpath: Couldn't close file '%s'; %s\n", perfile_path, strerror(errno));
		return 0;
	}

	return ret;
}

/**
 * Resets all net values and weights of a perceptron.
 *
 * @param per Initialized perceptron.
 * @return 0 if unsuccessful, 1 otherwise
 */
int perceptron_reset(perceptron per){
	int i, j, k, n;

	/*  Reset net values */
	/* Net values in all layers including bias */

	/* For all layers */
	for(i = 0; i < 3; ++i){
		n = i < 2 ? per->n[i] + 1 : per->n[i]; /* bias in this layer? */

		/* For all neuron + bias (if any) */
		for(j = 0; j < n; ++j)
			/* Set all to rand, bias (at last pos + 1) to 1 */
			per->net[i][j] = (j == per->n[i] + 1) ? 1 : (*(per->init))();  
	}

	/*  Reset weights */

	/* For the input and hidden layers */
	for(i = 0; i < 2; ++i)
		/* For all neuron and bias weights */
		for(j = 0; j < per->n[i] + 1; ++j)
			/* Each of them are related to a neuron in the next layer (no bias) */
			for(k = 0; k < per->n[i + 1]; ++k)
				per->w[i][j][k] = (*(per->init))();

	return 1;
}

/** 
 * Sets a given pattern as input.
 *
 * @param per Initialized perceptron.
 * @param in Vector of double with the in pattern. Length must be the same as perceptron->n[0]
 * @return 0 if unsuccessful, 1 otherwise
 */
int perceptron_setpattern(perceptron per, pattern pat){
	/*
	int i = 0;

	for(; i < per->n[0]; ++i)
		per->net[0][i] = pat[i];
		*/
	per->net[0] = pat;

	return 1;
}

/* Forward declaration */
static int perceptron_backpropagation_alloc_rw(perceptron per, double * **dw_ptr);
static int perceptron_backpropagation_alloc_dw(perceptron per, double * ***dw_ptr,
		double ** dw_raw_ptr);
static int perceptron_backpropagation_alloc_d(perceptron per, double * **d_ptr);

/**
 * Frees the perceptron structure
 *
 * @param per Initialized perceptron
 * @returns 0 if unsuccessful, 1 otherwise
 */
int perceptron_free(perceptron * per_ptr){
	perceptron per = *per_ptr;

	if( per == NULL ) {
		printerr("perceptron_free: Perceptron already freed.");
		return 0;
	}

	/* Free Net */
	if( per->net_raw != NULL )
		free(per->net_raw);

	if( per->net != NULL )
		free(per->net);

	/* Free Weights */
	if( per->w_raw != NULL )
		free(per->w_raw);

	if( per->w != NULL )
		free(per->w);

	/* Free temporal mem */
	if( per->d != NULL && per->d[0] != NULL )
		free(per->d[0]);

	if( per->d != NULL )
		free(per->d);

	if( per->dw_raw != NULL )
		free(per->dw_raw);

	if( per->dw )
		free(per->dw);

	if( per->rw != NULL && & per->rw[0] != NULL )
		free(per->rw[0]);

	if( per->rw != NULL )
	free(per->rw);

	free(per);
	*per_ptr = NULL;

	return 1;
}

/**
 * Initializes a perceptron given by reference
 *
 * @parak per Uninitialized perceptron by reference
 * @param nin Number of neurons in the input layer
 * @param nhidden Number of neurons in the hidden layer
 * @param nout Number of neurons in the output layer
 * @return 0 if unsuccessful, 1 otherwise
 * */
int perceptron_create(perceptron * per_ptr, int nin, int nhidden, int nout){

	int ni, nh, no, w_size, net_size;
	int i, j;
	double * raw = NULL;
	perceptron per = NULL;

	if((per = (perceptron) malloc (sizeof(perceptron_t))) == NULL)
		return 1;

	*per_ptr = per;

	/* Set perceptron dimensions */
	ni = per->n[0] = nin;
	nh = per->n[1] = nhidden;
	no = per->n[2] = nout;
	/* 1 extra bias value for in and h layers */
	net_size = ni + 1 + nh + 1 + no;
	w_size = (ni + 1) * nh + (nh + 1) * no;

	per->net = NULL;
	per->w = NULL;
	per->d = NULL;
	per->rw = NULL;
	per->dw = NULL;
	per->net_raw = NULL;
	per->w_raw = NULL;
	per->dw_raw = NULL;

	/* Set perceptron default functions */
	perceptron_setfunc_init(per, perceptron_rand);
	perceptron_setfunc_error(per, perceptron_mean_square_error);
	perceptron_setfunc_trans(per, perceptron_bipolarsigmoid);
	perceptron_setfunc_trans_prima(per, perceptron_bipolarsigmoid_prima);

	/*  Net: Neuron values */
	per->net = (double **) malloc (3 * sizeof(double*));
	if( per->net == NULL )
		return 1;

	/* Alloc contiguous memory for net and split it in layers */
	raw = (double *) malloc (net_size * sizeof(double));
	per->net[0] = &(raw[0]);
	per->net[1] = &(raw[ni]);
	per->net[2] = &(raw[ni+nh]);
	per->net_raw = raw;

	if( raw == NULL ){

		printerr("perceptron_create: Couldn't alloc space for net.\n");
		perceptron_free(&per);
		return 0;
	}

	/*  Bias value its always 1.
	 *  Will be set at last position in both layers by convention */
	per->net[0][ni-1] = 1;
	per->net[1][nh-1] = 1;

	/*  Weights */
	per->w = (double ***) malloc (2 * sizeof(double**));

	if( per->w == NULL ){
		printerr("perceptron_create: Couldn't alloc space for weights.");
		perceptron_free(&per);
		return 0;
	}

	/* Input and hidden layers
	 * ninput neurons + bias to nhidden neurons  */

	/* Alloc contiguous memory for weights and split it within the cube */
	raw = (double *) malloc (w_size * sizeof(double));
	if( raw == NULL ){
		printerr("perceptron_create: Couldn't alloc space for weight values.");
		return 0;
	}
	per->w_raw = raw;

	/* For all neuron and bias weight in the input and hidden layer */
	for(i = 0; i < 2; ++i) {
		per->w[i] = (double **) malloc ((per->n[i] + 1) * sizeof(double *));

		/* For all neuron (no bias) in the next layer */
		for(j = 0; j < per->n[i] + 1; ++j)
			per->w[i][j] = &(raw[ (i * ni * (nh-1)) + (j * per->n[i+1]) ]);
	}

	/* Init delta temporal matrix and cube */
	perceptron_backpropagation_alloc_d(per, &(per->d));
	perceptron_backpropagation_alloc_dw(per, &(per->dw), &(per->dw_raw));
	perceptron_backpropagation_alloc_rw(per, &(per->rw));

	/* Set all neurons and weights */
	return perceptron_reset(per);
}

/**
 * Computes backpropagation for a perceptron and a given pattern.
 *
 * @param per Initialized perceptron
 * @param pset Initialized patternset
 * @param lrate Learning rate 
 * @param thres Error threshold. Iteration stop condition.
 * @return 0 if unsuccessful, 1 otherwise
 */
int perceptron_training(perceptron per, patternset pset, double lrate, double thres, int limit) {
	int epoch, i;
	double error = thres + 1;

	if( per->n[2] > pset->no ) {
		printerr("perceptron_training: Incompatible output layer sizes for perceptron and patterns.\n");
		return 0;
	}

	/* Until error reaches threshold or epoch limit is reached */
	for(epoch = 0; error > thres && epoch < limit; ++epoch){
		error = 0;

		/* Calculate epoch */
		for(i = 0; i < pset->npats; ++i) {
			if( perceptron_backpropagation_raw(per, pset->input[i], pset->codes[i], lrate) == 0 ) {
				printerr("perceptron_training: Error applying backpropagation at pattern:%i epoch:%i\n", i, epoch);
				return 0;
			}

			/* Calculate error */
			error += (*(per->error))(per->net[2], pset->codes[i], per->n[2]);
		}
	}

	return 1;
}

/**
 * Computes backpropagation for a perceptron and a given pattern.
 * Logs the error per epoch to stream.
 *
 * @param per Initialized perceptron
 * @param pset Initialized patternset
 * @param lrate Learning rate 
 * @param thres Error threshold. Iteration stop condition.
 * @param stream Output stream
 * @return 0 if unsuccessful, 1 otherwise
 */
int perceptron_trainingprint(perceptron per, patternset pset, double lrate, double thres, int limit, FILE * stream) {
	int epoch, i;
	double error = thres + 1;
	double prev_error = error + 1;

	if(pset->npats == 0){
		printerr("perceptron_training: empty patternset\n");
		return 0;
	}

	if( per->n[2] > pset->ni ) {
		printerr("perceptron_training: Incompatible output layer sizes for perceptron and patterns.\n");
		return 0;
	}

	/* Print header */
	if( stream ) fprintf(stream, "#epoch\tneurons\talpha\terror\n");

	/* Until error reaches threshold or epoch limit is reached */
	for(epoch = 0; error >= thres && epoch < limit; ++epoch){
		prev_error = error;
		error = 0;

		printf("Epoch: %d\n", epoch);

		/* Calculate epoch */
		for(i = 0; i < pset->npats; ++i) {
			if( perceptron_backpropagation(per, pset->input[i], pset->codes[i], lrate) == 0 ) {
				printerr("perceptron_training: Error applying backpropagation at pattern:%i epoch:%i\n", i, epoch);
				return 0;
			}

			/* Calculate error */
			error += (*(per->error))(per->net[2], pset->codes[i], per->n[2]);

			printf("Pattern %d\n", i);
		}

		error /= pset->npats;  /* Error per pattern */

		if( stream )
			fprintf(stream, "%i\t%i\t%f\t%f\n", epoch, per->n[1], lrate, error);
	}

	return 1;
}

/**
 * Sets init function
 * @param fun Function to set.
 * @return old function
 */
perceptron_fun_init perceptron_setfunc_init(perceptron per, double(*fun)()) {
	double (*tmp)() = per->init;
	per->init = fun;
	return tmp;
}

/**
 * Sets error function
 * @param fun Function to set.
 * @return old function
 */
perceptron_fun_error perceptron_setfunc_error(perceptron per, double(*fun)(double*,size_t,int)) {
	double(*tmp)(double*,size_t,int) = per->error;
	per->error = fun;
	return tmp;
}

/**
 * Sets trans function
 * @param fun Function to set.
 * @return old function
 */
perceptron_fun_trans perceptron_setfunc_trans(perceptron per, double(*fun)(double)) {
	double (*tmp)(double) = per->trans;
	per->trans = fun;
	return tmp;
}

/**
 * Sets trans_prima function * @param fun Function to set.
 * @return old function
 */
perceptron_fun_trans perceptron_setfunc_trans_prima(perceptron per, double(*fun)(double)) {
	double (*tmp)(double) = per->trans_prima;
	per->trans_prima = fun;
	return tmp;
}

static int perceptron_backpropagation_alloc_rw(perceptron per, double * **d_ptr){
	double ** rw = (double **) malloc (2 * sizeof(double *));
	double * rw_raw = (double *) malloc ((per->n[1] + per->n[2]) * sizeof(double));
	if( rw != NULL ){
		rw[0] = &(rw_raw[0]);
		rw[1] = &(rw_raw[per->n[1]]);
	}

	*d_ptr = rw;

	return rw != NULL;
}

static int perceptron_backpropagation_alloc_d(perceptron per, double * **d_ptr){
	/* Allocation for Neuron deltas */
	double ** d = (double **) malloc (3 * sizeof(double *));
	double * d_raw = (double *) calloc (per->n[1] + per->n[2] * 2, sizeof(double));

	if( d != NULL ) {
		d[0] = &(d_raw[0]); /* Hidden layer neurons (no bias) */
		d[1] = &(d_raw[per->n[1]]); /* Output layer neurons (no bias) */
		d[2] = &d_raw[per->n[1] + per->n[2]]; /* Difference between pattern and output */
	}

	*d_ptr = d;

	return d != NULL;
}

static int perceptron_backpropagation_alloc_dw(perceptron per, double * ***dw_ptr, double ** dw_raw_ptr){
	/* Allocation for Weight corrections */
	int i = 0, j = 0, size1, size2;
	double *** dw = (double ***) malloc (2 * sizeof(double **));
	double * dw_raw = NULL;

	if( dw == NULL )
		return 0;

	dw[0] = (double **) malloc ((per->n[0] + 1) * sizeof(double*)); /* Input layer weights (+ bias) */
	dw[1] = (double **) malloc ((per->n[1] + 1) * sizeof(double*)); /* Hidden layer weights (+ bias) */

	/* Alloc contiguous memory and split it afterwards */
	size1 = per->n[1] * (per->n[0] + 1);  /* First layer weights */
	size2 = per->n[2] * (per->n[1] + 1);  /* Second layer weights */

	dw_raw = (double *) calloc (size1 + size2, sizeof(double));

	if( dw_raw == NULL ){
		printerr("ERROR: Couldn't alloc space for weight deltas.\n");
		free(dw[0]);
		free(dw[1]);
		free(dw);

		return 1;
	}

	/* Associate layers */
	for(i = 0; i < 2; ++i)
		for(j = 0; j < per->n[i] + 1; ++j)
			dw[i][j] = &(dw_raw[ i * size1 + j * per->n[i+1] ]);

	*dw_ptr = dw;
	*dw_raw_ptr = dw_raw;

	return dw != NULL;
}

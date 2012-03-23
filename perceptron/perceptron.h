/*
 *       Filename:  perceptron.h
 *    Description:  Multilayer Perceptron
 *         Author:  Javier Santacruz <francisco.santacruz@estudiante.uam.es>
 *
 */

#include <stdio.h>

#include "pattern.h"

/**
 * n: Layers lengths. 0 for input, 1 for hidden and 2 for output layer.
 * net: Neuron values along iterations (matrix: [ni+1, nh+1, no])
 * w: Neuron weighted conections (cube: [ni+1 x [nh], nh+1 x [no]])
 *
 * All this functions can be setted to modify the way the perceptron
 * works.
 *
 * init: Initialize net weights. (default is rand in [-1,1])
 * error: Net error. (default is mean square error)
 * trans: Transition function. (default is bipolar sigmoid)
 * trans_prima: Transition function for bkpr. (default is bipolar sigmoid prima)
 *
 * Perceptron description file format
 *
 * NI NH NO
 * I Layer weights (length NH)
 * I Layer weights
 * ... NI + 1 ...
 * H Layer weights (length NO)
 * H Layer weights
 * ... NH + 1 ...
 *
 */
typedef struct {
	int n[3];

	double ** net; /* neuron values */
	double *** w;  /* weights */

	/* Internals */
	double ** d;   /* output delta */
	double ** rw;   /* neuron raw inputs */
	double *** dw;  /* delta weights */

	int wsize;  /* Total length of w */

	double(*init)();              /* initialization function */
	double(*trans)(double);       /* transition function */
	double(*trans_prima)(double); /* prima transition function */
	double(*error)(double*,size_t,int); /* error function */
} perceptron_t;

typedef perceptron_t * perceptron;

/* Perceptron function types  */
typedef double(*perceptron_fun_init)();
typedef double(*perceptron_fun_trans)(double);
typedef double(*perceptron_fun_error)(double*,size_t,int);

/**
 * Initializes a perceptron passed by reference
 *
 * @parak per Uninitialized perceptron by reference
 * @param nin Number of neurons in the input layer
 * @param nhidden Number of neurons in the hidden layer
 * @param nout Number of neurons in the output layer
 * @return 0 if unsuccessful, 1 otherwise
 * */
int perceptron_create(perceptron * per, int nin, int nhidden, int nout);

/**
 * Reads perceptron weights and structure from stream.
 *
 * @param per Initialized perceptron.
 * @param perfile Output stream.
 * @return 0 if unsuccessful, 1 otherwise
 */
int perceptron_read(perceptron * per, FILE * perfile);

/**
 * Reads perceptron weights and structure from file.
 *
 * @param per Initialized perceptron.
 * @param perfile_path Path to perceptron description file.
 * @return 0 if unsuccessful, 1 otherwise
 */
int perceptron_readpath(perceptron * per, const char * perfile_path);

/**
 * Dumps perceptron weights and structure to stream.
 *
 * @param per Initialized perceptron.
 * @param perfile Output stream.
 * @return 0 if unsuccessful, 1 otherwise
 */
int perceptron_print(perceptron per, FILE * perfile);

/**
 * Dumps perceptron weights and structure to file.
 *
 * @param per Initialized perceptron.
 * @param perfile_path Path to file.
 * @return 0 if unsuccessful, 1 otherwise
 */
int perceptron_printpath(perceptron per, const char * perfile_path);

/**
 * Sets a given pattern as input.
 *
 * @param per Initialized perceptron.
 * @param pat Pattern to set the input.
 * @return 0 if unsuccessful, 1 otherwise
 */
int perceptron_setpattern(perceptron per, pattern pat);

/**
 * Resets all net values and weights of a perceptron.
 *
 * @param per Initialized perceptron.
 * @return 0 if unsuccessful, 1 otherwise
 */
int perceptron_reset(perceptron per);

/**
 * Frees the perceptron structure
 *
 * @param per Initialized perceptron
 * @returns 0 if unsuccessful, 1 otherwise
 */
int perceptron_free(perceptron * per_ptr);

/**
 * Computes backpropagation for a perceptron and a given pattern.
 *
 * @param per Initialized perceptron.
 * @param pset Initialized patternset.
 * @param lrate Learning rate.
 * @param thres Error threshold. Iteration stop condition.
 * @param limit Max number of epochs allowed for the learning.
 * @return 0 if unsuccessful, 1 otherwise
 */
int perceptron_training(perceptron per, patternset pset, double lrate, double thres, int limit);

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
int perceptron_trainingprint(perceptron per, patternset pset, double lrate, double thres, int limit, FILE * stream);

/**
 * Computes backpropagation for a perceptron and a given pattern.
 *
 * @param per Initialized perceptron
 * @param pat Initialized pattern
 * @param code Output pattern
 * @param lrate Learning rate
 * @return 0 if unsuccessful, 1 otherwise
 */
int perceptron_backpropagation(perceptron per, pattern pat, size_t code, double lrate);


/**
 * Full parametrized call to backpropagation
 *
 * Computes backpropagation for a perceptron and a given pattern.
 * Raw version which perform the calculations and
 *
 * @param per Initialized perceptron
 * @param pat Initialized pattern
 * @param code Active neuron in output pattern
 * @param lrate Learning rate
 * @param update Update weights (!=0) or not (0)
 * @return 0 if unsuccessful, 1 otherwise
 */
int perceptron_backpropagation_raw(perceptron per, pattern pat, size_t code, double lrate, int update);

/**
 * Computes forward feeding for perceptron given a pattern.
 *
 * @param per Initialized perceptron
 * @param pat Initialized pattern
 * @return 0 if unsuccessful, 1 otherwise
 */
int perceptron_feedforward(perceptron per, pattern pat);

/**
 * Fixes a cube of data from its raw data.
 *
 * @param cube Matrix of double pointers.
 * @param raw Raw vector of doubles.
 * @param sizes Vector of 3 sizes.
 */
int set_cube_pointers(double *** cube, double * raw, int * sizes);

/**
 * Sets init function
 * @param fun Function to set.
 * @return old function
 */
perceptron_fun_init perceptron_setfunc_init(perceptron per, double(*fun)());

/**
 * Sets error function
 * @param fun Function to set.
 * @return old function
 */
perceptron_fun_error perceptron_setfunc_error(perceptron per, double(*fun)(double*,size_t, int));

/**
 * Sets trans function
 * @param fun Function to set.
 * @return old function
 */
perceptron_fun_trans perceptron_setfunc_trans(perceptron per, double(*fun)(double));

/**
 * Sets trans_prima function
 * @param fun Function to set.
 * @return old function
 */
perceptron_fun_trans perceptron_setfunc_trans_prima(perceptron per, double(*fun)(double));

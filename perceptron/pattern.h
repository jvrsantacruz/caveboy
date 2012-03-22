/*
 *       Filename:  imgpattern.h
 *    Description:  Image patterns
 *         Author:  Javier Santacruz (), <francisco.santacruz@estudiante.uam.es>
 *         Author:  Alberto Montes (), <alberto.montes@estudiante.uam.es>
 *
 *        Created:  20/02/12 16:55:39
 *
 *   A patternset holds a list of patterns with its associated output.
 */

typedef struct {
	size_t npats, npsets, w, h, bpp, size, ni, no;
	char ** names;     /* Name for each patternset. name[code] */

	double ** input;   /* All input patterns */
	double * input_raw;   /* All input patterns in contiguous memory */
	size_t * codes;    /* Code for each pattern. codes[npat] */
} patternset_t;

typedef patternset_t * patternset;
typedef double * pattern;

/* Macro to know if a neuron should be active or not depending on the
 * code */
#ifndef ISACTIVE
 #define ISACTIVE(i, code) ((i == code) ? 1.0 : -1.0)
#endif

/* 
 * Reads image patterns from dir path
 * @param pset_ptr Uninitialized patternset by reference.
 * @param dir_path Path to the root patternsets directory.
 * @return != 0 on success.
 */

/* Creates an empty patternset
 * No memory is allocated. */
int patternset_create(patternset * pset_ptr, int npats, int patsize, int npsets);


/* Frees an allocated patternset */
int patternset_free(patternset * pset_ptr); 

/* Convert uchar raw image data to double pattern 
 * Convert each pixel in a double number 
 *
 * @param pattern Uninitialized double pattern vector.
 * @param upattern Initialized raw image data.
 * @param size Total size of upattern.
 * @param bpp Bytes per pixel.
 * @return the final size of pattern double vector.
 */
int pattern_create(pattern * pat, unsigned char * upattern, size_t size, size_t bpp); 

/* 
 * Returns the code marked in a given pattern.
 * @param pattern Initialized output pattern.
 * @param npsets Pattern length
 * @param min Min val to consider a number as 1.
 */
size_t pattern_to_code(double * pattern, size_t npsets, double min);

/* Sets needed info obtained in training phase in the test patternset 
 * Basically copy the names for consulting the output net codes.
 *
 * @param training The trained patternset.
 * @param test Filled but not trained patternset.
 * @return 0 if something went wrong, 1 otherwise.
 */ 
int patternset_set_traininginfo(patternset training, patternset test);

/* Sets needed info obtained in training phase in the test patternset 
 * Basically copy the names for consulting the output net codes.
 *
 * @param path The path to the file where the names are.
 * @param test Filled but not trained patternset.
 * @return 0 if something went wrong, 1 otherwise.
 */ 
int patternset_read_traininginfo(patternset test, const char * path);

/* Dumps the patternset training info to a file.
 *
 * @param path The path to the file where the names are.
 * @param training Trained patternset.
 * @return 0 if something went wrong, 1 otherwise.
 *
 * The format is plaintext like the following:
 * n
 * name0
 * name1
 * ...
 */ 
int patternset_print_traininginfo(patternset training, const char * path);

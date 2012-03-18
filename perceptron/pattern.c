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

#define _BSD_SOURCE 1       /* Allows dirent.h scandir() */
#define _POSIX_C_SOURCE 200809   /* Allows stdio.h getline() */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <limits.h>

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include "pattern.h"
#include "../pnglite/pnglite.h"

/*  Handy macros */
#ifndef printerr
  #define printerr(...) fprintf(stderr, __VA_ARGS__)
#endif

#ifndef printdbg
 #ifdef DEBUG
   #include <stdio.h>
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

static int dir_select(const struct dirent * dire);
static int png_select(const struct dirent * dire);

/* 
 * Returns the code marked in a given pattern.
 * pattern: Initialized output pattern.
 * npsets: Pattern length
 * dist: Min val to consider a number as 1.
 */
size_t pattern_to_code(double * pattern, size_t npsets, double min){
	size_t pos = npsets - 1;

	/* Find the 1 in pattern */
	while( pos > 0  && pattern[pos] < min) --pos;

	return pos;
}

/* Convert uchar raw image data to double pattern 
 * Convert each pixel in a double number 
 *
 * pattern: Uninitialized double pattern vector.
 * upattern: Initialized raw image data.
 * size: Total size of upattern.
 * bpp: Bytes per pixel.
 *
 * Returns the final size of pattern double vector.
 *         0 in case of error;
 */
int pattern_create(pattern * pat, unsigned char * upattern, size_t size, size_t bpp) {
	size_t i = 0, b = 0, n = 0;

	/* Check sizes */
	if( size % bpp != 0 ) {
		printerr("ERROR: Unaligned raw data and bpp.\n");
		return FALSE;
	}

	if( bpp > sizeof(size_t) ){
		printerr("ERROR: Pixel value overflow (%ld bpp)\n", bpp);
		return FALSE;
	}

	if( bpp > sizeof(size_t) ) {
		printerr("ERROR: Too many bytes per pixel (%ld) to perform"\
				"conversion\n", bpp);
		return FALSE;
	}

	/* Alloc 1 extra 'hidden' double for the bias. 
	 * To be used later in the perceptron so it will fit sizes */
	*pat = (double *) malloc ((sizeof(double) * size/bpp) + 1);
	if( *pat == NULL ){
		printerr("ERROR: Out of memory\n");
		return FALSE;
	}
	
	/* Set each bpp bytes together as a single
	 * number and convert it to double */
	for(n = 0, b = 0; i < size; i += bpp) {
		while( b < bpp ) {
			/* Add each byte in its position */
			n |= upattern[i+b] << (b * 8);
			++b;  /* Next byte */
		}

		/* One double value per pixel */
		(*pat)[i/bpp] = (double) n;
	}

	return size/bpp;
}

static int patternset_init(patternset * pset_ptr, size_t npsets, size_t npats) {
	patternset pset = (patternset) malloc (sizeof(patternset_t));
	int i = 0;

	if( pset == NULL )
		return FALSE;

	/* Init values */
	pset->npsets = npsets;
	pset->npats = npats;
	pset->size = pset->w = pset->h = pset->bpp = 0;
	pset->names = (char **) malloc (sizeof(char *) * npsets);
	pset->input = (double **) malloc (sizeof(double *) * npats);
	pset->codes = (size_t *) malloc (sizeof(size_t) * npats);

	if( pset->names == NULL 
			|| pset->input == NULL
			|| pset->codes == NULL )
		return FALSE;

	/* Set unused names to NULL */
	for(; i < npsets; ++i) 
		pset->names[i] = NULL;

	*pset_ptr = pset;

	return TRUE;
}

static int patternset_expand(patternset pset, size_t size){
	pset->input = (double **) realloc (pset->input, size);
	pset->codes = (size_t *) realloc (pset->codes, size);

	if(pset->input == NULL || pset->codes == NULL )
		return FALSE;

	return TRUE;
}

/* Sets needed info obtained in training phase in the test patternset 
 * Basically copy the names for consulting the output net codes.
 *
 * @param training The trained patternset.
 * @param test Filled but not trained patternset.
 * @return 0 if something went wrong, 1 otherwise.
 */ 
int patternset_set_traininginfo(patternset training, patternset test){
	int i = 0;

	/* Check there is something to do */
	if( training->npsets == 0 )
		return FALSE;

	/* Free useless names from test first */
	if( test->names != NULL ){
		for(; i < test->npsets; ++i)
			if( test->names[i] != NULL )
				free(test->names[i]);
		free(test->names);
	}

	test->names = (char **) malloc (sizeof(char *) * training->npsets);
	if( test->names == NULL )
		return FALSE;

	/* Copy all names in training to test */
	for(i = 0; i < training->npsets; ++i) {
		test->names[i] = (char *) malloc (strlen(training->names[i]) + 1);
		strcpy(test->names[i], training->names[i]);
	}

	/* Set the same output size for both nets */
	test->npsets = training->npsets;
	test->no = training->no;

	return TRUE;
}

/* Sets needed info obtained in training phase in the test patternset 
 * Basically copy the names for consulting the output net codes.
 *
 * @param path The path to the file where the names are.
 * @param test Filled but not trained patternset.
 * @return 0 if something went wrong, 1 otherwise.
 */ 
int patternset_read_traininginfo(patternset test, const char * path) {
	size_t i = 0, n = 0, l = 0, npsets = 0;
	char * buf = NULL;
	FILE * stream = fopen(path, "r");

	if( stream == NULL )
		return FALSE;

	/* Read header  */
	if( fscanf(stream, "%zu\n", &n) == 0 || n == 0) {
		fclose(stream);
		return FALSE;
	}

	/* Free useless names from test first */
	if( test->names != NULL ){
		for(; i < test->npsets; ++i)
			if( test->names[i] != NULL )
				free(test->names[i]);
		free(test->names);
	}

	/* Alloc for n patternset names */
	test->names = (char **) malloc (sizeof(char *) * n);
	if( test->names == NULL )
		return FALSE;

	/* Read subsequent lines */
	for(i = 0; i < n; ++i) {
		if( getline(&buf, &l, stream) == -1 ) {
			test->names[i] = NULL;
		} else {
			test->names[i] = buf;
			++npsets;
		}

		/* Reset buffer so getline will alloc new space */
		buf = NULL;
		l = 0;
	}

	/* Set the same output size for both nets */
	test->npsets = npsets;
	test->no = npsets;

	return TRUE;
}

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
int patternset_print_traininginfo(patternset training, const char * path){
	size_t i = 0;
	FILE * stream = NULL; 

	if( training->npsets == 0 )
		return FALSE;
	
	stream = fopen(path, "w");
	if( stream == NULL )
		return FALSE;

	fprintf(stream, "%zd\n", training->npsets);

	for(i = 0; i < training->npsets; ++i)
		fprintf(stream, "%s\n", training->names[i]);

	fclose(stream);

	return TRUE;
}

int patternset_readpath(patternset * pset_ptr, const char * dir_path) {
	size_t ndirs = 0, npngs = 0, allocpats = 256, npats = 0, npsets = 0, 
		   i = 0, j = 0, k = 0, readpngs = 0; 
	int ret = 0, exit_error = FALSE;
	struct dirent ** dirs = NULL,
				  ** files = NULL;
	unsigned char * rawdata = NULL;
	char full_dir_path[PATH_MAX],
		 full_png_path[PATH_MAX]; 
	png_t image; 
	patternset pset = NULL;

	/* 
	 * A patternset is a directory containing png images, which are the
	 * actual patterns. We list the different directories in the given
	 * path and list them one at time to get all the png images inside.
	 * Each directory name containing pngs its associated to a code, so
	 * the patterns within the directory are also associated to the code.
	 *
	 * - pattern: png image and a number
	 * - patternset: directory with png images.
	 *
	 * Open the given dir and look for directories (patternsets) containing png
	 * images (patterns).
	 * Associate a number (npsets) to each patternset dir name.
	 * Read all png images and associate them with the npsets.
	 *
	 * Tables:
	 * codes: Patternset code for each pattern. codes[pat] = pset
	 * names: Patternset names. names[pset] = psetname
	 * input: Double input pattern.
	 */

	/* Read top directory and get patternsets names */
	if( (ndirs = scandir(dir_path, &dirs, dir_select, NULL)) < 0 ) {
		printerr("ERROR: Couldn't open patternset directory at '%s'\n",
				dir_path);
		return TRUE;
	} 

	if( ndirs == 0 ) {
		printerr("ERROR: No patternsets present in patternset directory at '%s'\n", 
				dir_path);
		exit_error = TRUE;
	}

	if( !exit_error )
		patternset_init(&pset, ndirs, allocpats);

	/* Initialize pnglite */
	png_init(NULL, NULL);

	/* Open each directory and read all pngs, assigning them different codes */
	npats = 0;
	npsets = 0;
	for(i = 0; !exit_error && i < ndirs; ++i) {

		readpngs = 0;  /* n of successfully read pngs within this dir */
		sprintf(full_dir_path, "%s/%s", dir_path, dirs[i]->d_name);

		/* List root directory and check wether there is pngs inside. */
		if( (npngs = scandir(full_dir_path, &files, png_select, NULL)) == -1 ) {
			printerr("WARNING: Couldn't open patterns dir: '%s'\n", 
					full_dir_path);
			continue;
		}

		if( npngs == 0 ) {
			printerr("WARNING: Ignoring empty patterns dir: '%s'\n", 
					full_dir_path);
			continue;
		}

		/* Once we have the png files list, load them.  
		 * At first png load, store the sizes and ignore all following
		 * images that doesn't match.
		 *
		 * We cannot know in advance the n of images we have to read, so
		 * we dinamically alloc entries in blocks and check sizes after
		 * every loaded image .
		 */
		for(j = 0; !exit_error && j < npngs; ++j) {

			/* Extra check for file name length */
			if( strlen(files[j]->d_name) < 5 )
				continue;

			/* Obtain full subdirectory path */
			sprintf(full_png_path, "%s/%s", full_dir_path, files[j]->d_name);

			/* Load png image */
			if( (ret = png_open_file(&image, full_png_path)) != PNG_NO_ERROR) {
				printerr("WARNING: Couldn't open PNG image: '%s': %s\n", 
						full_png_path, png_error_string(ret));
				continue;
			}

			/* We get size info from the first loaded image.
			 * Once we know the sizes, we can alloc space for patterns.
			 * We must alloc for:
			 * - rawdata: Temp memory for storing raw image data.
			 * - pset->codes: Each pattern has a code.
			 * - pset->input: Each pattern has a double vector as input.
			 * - pset->names: Copy dir name when we successfully read a png.
			 */
			if( !pset->size ) {
				pset->w = image.width;
				pset->h = image.height;
				pset->bpp = image.bpp;
				pset->size = pset->w * pset->h * pset->bpp;

				/* Temp buffer to store raw image data 
				 * before converting it to double in the pattern */
				rawdata = (unsigned char *) 
					malloc (sizeof(unsigned char) * pset->size);

				printf("INFO: First PNG loaded. "\
						"Sizes: %ldx%ld (%ld Bpp) (pattern %ld KB) (raw %ld KB)\n",
						pset->w, pset->h, pset->bpp, 
						(sizeof(double) * pset->w * pset->h)/1024, pset->size/1024);
			}

			/* All png should be equally sized */
			if( image.width != pset->w 
				 || image.height != pset->h 
				 || image.bpp != pset->bpp ) {
				printerr("WARNING: Ignoring PNG file '%s'. It's %dx%d (%d Bpp)"\
						" instead of %ldx%ld (%ld Bpp) as it should be.",
						full_png_path, image.width, image.height,
						image.bpp, pset->w, pset->h, pset->bpp);

				png_close_file(&image);
				continue;
			}

			/* Check table sizes and realloc if necessary */
			if( allocpats == npats ) {
				allocpats += 256;   /* 256 extra patterns */

				if( patternset_expand(pset, allocpats) == FALSE ) {
					printerr("ERROR: Out of memory\n");
					exit_error = TRUE;
					png_close_file(&image);
					continue;
				} 
			}

			/* Get png raw data, convert it to double and set it as
			 * pattern input, associating it with the patternset
			 * directory code */
			if( (ret = png_get_data(&image, rawdata)) != PNG_NO_ERROR){
				printerr("WARNING: Couldn't get data from '%s': %s. Ignoring file.\n", 
						full_png_path, png_error_string(ret));
				png_close_file(&image);
				continue;
			}

			/* Convert raw data into pixel double values */
			if( (ret = pattern_create(&(pset->input[npats]), 
							rawdata, pset->size, pset->bpp)) == FALSE) {
				printerr("ERROR: Couln't alloc pattern\n");
				exit_error = TRUE;
				png_close_file(&image);
				continue;
			}

			png_close_file(&image);

			/* Set patternset name-code if unset 
			 * Associate the name with the pattern through the code */
			if( pset->names[npsets] == NULL ) {
				pset->names[npsets] = (char *) malloc (strlen(dirs[i]->d_name) + 1);
				strcpy(pset->names[npsets], dirs[i]->d_name);
			}
			pset->codes[npats] = npsets;


			++npats;     /* Next pattern id */
			++readpngs;  /* Patternset valid patterns count */

		}   /* PNG file j in dir i */

		/* Next patternset (if it wasn't ignored) */
		if( readpngs > 0 )
			++npsets;
		else 
			printerr("WARNING: No files read from '%s' patternset directory.\n", 
					full_dir_path);

		/* Free file list */
		if( files != NULL ) {
			for(k = 0; k < npngs; ++k)
				free(files[k]);
			free(files);
			files = NULL;
		}
	}

	exit_error = (npats == 0);
	printerr("Pattern loading finished. %zd patterns read from '%s'\n", npats, dir_path);

	if( !exit_error ) {
		/* Set patternset values */
		pset->npats = npats;
		pset->npsets = npsets;
		pset->ni = pset->size / pset->bpp;
		pset->no = npsets;
		*pset_ptr = pset;
	} 

	/* Free file list */
	if( dirs != NULL ) {
		for(i = 0; i < ndirs; ++i)
			if( dirs[i] != NULL )
				free(dirs[i]);
		free(dirs);
	}

	/* Free temporals */
	if( rawdata != NULL ) 
		free(rawdata);

	if( exit_error )
		patternset_free(&pset);

	return !exit_error;
}

int patternset_free(patternset * p) {
	patternset pset = NULL;
	int i = 0;

	if( p == NULL )
		return TRUE;

	/* Rename pset */
	pset = *p;

	if( pset == NULL )
		return TRUE;

	/* Free inputs */
	if( pset->input != NULL ) {
		for(i = 0; i < pset->npats; ++i)
			if( pset->input[i] != NULL ) {
				free(pset->input[i]);
				pset->input[i] = NULL;
			}
		free(pset->input);
		pset->input = NULL;
	}

	/* Free codes */
	if( pset->codes != NULL )
		free(pset->codes);
	pset->codes = NULL;

	/* Free names */
	if( pset->names != NULL ) {
		for(i = 0; i < pset->npsets; ++i)
			if( pset->names[i] != NULL )
				free(pset->names[i]);
		free(pset->names);
		pset->names = NULL;
	}

	/* Free the structure */
	free(*p);
	*p = NULL;

	return TRUE;
}

/* Directory selector */
static int dir_select(const struct dirent * dire) {
	/* Check wether if it is not a DIR.
	 * Some FS doesn't handle d_type, so we check UNKNOWN as well */
	if( dire->d_type != DT_UNKNOWN 
			&& dire->d_type != DT_DIR )
		return 0;

	/* Discard . and .. */
	if( strncmp(dire->d_name, ".", 2) == 0 
		 || strncmp(dire->d_name, "..", 3) == 0 )
		return 0;
	
	return 1;
}

static int png_select(const struct dirent * dire){
	int len = 0;

	/* Check wether if it is not a DIR.
	 * Some FS doesn't handle d_type, so we check UNKNOWN as well */
	if( dire->d_type != DT_UNKNOWN 
			&& dire->d_type != DT_REG )
		return 0;

	/* At least 5 chars: 'x.png' */
	if( (len = strlen(dire->d_name)) < 5 )
		return 0;

	/* Must end in '.png' */
	if( strncmp(dire->d_name + len - 4, ".png", 4) != 0 ) 
		return 0;

	return 1;
}

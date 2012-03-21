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
 * pattern: Initialized double pattern vector.
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
		printerr("ERROR: Unaligned raw data and bpp (%ld and %ld).\n", size, bpp);
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

	/* Alloc row pointers for each pattern */
	pset->input = (double **) malloc (sizeof(double *) * npats);
	pset->codes = (size_t *) malloc (sizeof(size_t) * npats);

	if( pset->names == NULL 
			|| pset->input == NULL
			|| pset->codes == NULL )
		return FALSE;
	}

	/* Set unused names to NULL */
	for(; i < npsets; ++i) 
		pset->names[i] = NULL;

	*pset_ptr = pset;

	return TRUE;
}

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

/* Searches for valid png files.
 * @param dir_path Full path to the root patterns directory.
 * @param npats Number of patterns, by reference. To be set by the function.
 * @param npsets Number of patternsets, by reference. To be set by the function.
 * @param w Image width, by reference. To be set by the function.
 * @param h Image height, by reference. To be set by the function.
 * @param b Image bpp, by reference. To be set by the function.
 * @param png_paths Uninitialized list of strings by reference. To be freed by the user.
 * @param png_codes Unitialized list of codes for each png by reference.
 * @param pset_names Unitialized list of names for each code by reference.
 *
 * @return png_paths length. <= 0 on error */
static size_t list_valid_pngs(const char * dir_path, size_t * npats, size_t * npsets,
		size_t * w, size_t * h, size_t * b, char *** png_paths, size_t ** png_codes,
		char *** pset_names) {
	int ndirs = 0, listlen = 0, ndirvalidpngs = 0, ndirvalid = 0, ndirpngs = 0,
		npngs = 0, d = 0, p = 0, ret = 0;
	struct dirent ** dirs = NULL,
				  ** files = NULL;
	char full_dir_path[PATH_MAX],
		 full_png_path[PATH_MAX];
	png_t image;

	/* Initialize pnglite */
	png_init(NULL, NULL);

	/* Initial values for external variables */
	*npats = *npsets = 0;
	*w = *h = *b = -1;

	/* The png images are supposed to be
	 * in a 2 layers structure like the following:
	 *
	 * dir
	 *  |- pdir
	 *  |   |- image1.png
	 *  |   `- image2.png
	 *  |- pdir
	 *  (...)
	 *
	 * 1. Open dir and list pdirs
	 * 2. Open each pdir and list the pngs
	 * 3. Open each png and check if its valid
	 *    (First opened png will provide the sizes)
	 * 4. Save valid png full path
     *
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
	 * png_paths: Full paths to a valid png to be parsed. Length: npngs
	 * codes: Patternset code for each pattern. codes[pat] = pset Length: npngs
	 * names: Patternset names. names[pset] = psetname Length: nvaliddir
	 */

	/* Read top directory and get patternsets names */
	if( (ndirs = scandir(dir_path, &dirs, dir_select, NULL)) < 0 ) {
		printerr("ERROR: Couldn't open patternset directory at '%s'\n",
				dir_path);
		return 0;
	}

	/* Alloc space for dir names list */
	(*pset_names) = (char **) malloc (ndirs * sizeof(char*));
	for(d = 0; d < ndirs; ++d)
		(*pset_names)[d] = NULL;

	/* Read all patternset dirs */
	for(d = 0; d < ndirs; ++d){
		ndirvalidpngs = 0;   /* valid pngs within this directory */

		/* Get directory full path */
		sprintf(full_dir_path, "%s/%s", dir_path, dirs[d]->d_name);

		/* Read all pngs within the patternset dirs[d] */
		if( (ndirpngs = scandir(full_dir_path, &files, png_select, NULL)) == -1 ) {
			printerr("WARNING: Couldn't open patterns dir: '%s'\n",
					full_dir_path);
			continue;
		}

		if( ndirpngs == 0 ) {
			printerr("WARNING: Emtpy patterns dir: '%s'\n", full_dir_path);
			continue;
		}

		/* Expand png paths and codes list if neccesary */
		if( npngs >= listlen ) {
			listlen = npngs + ndirpngs;

			(*png_paths) = (char **) realloc (*png_paths,
					sizeof(char *) * listlen);
			if( *png_paths == NULL ) {
				printerr("ERROR: Out of memory for png paths.\n");
				return -1;
			}

			(*png_codes) = (size_t *) realloc (*png_codes, sizeof(size_t) * listlen);
			if( *png_codes == NULL ){
				printerr("ERROR: Out of memory for png codes.\n");
				return -1;
			}
		}

		/* Read pngs in directory */
		for(p = 0; p < ndirpngs; ++p){

			/* Get png file full path */
			sprintf(full_png_path, "%s/%s", full_dir_path, files[p]->d_name);

			/* Try to open image */
			if( (ret = png_open_file(&image, full_png_path)) != PNG_NO_ERROR){
				printerr("WARNING: Couldn't open PNG image: '%s': %s\n",
						full_png_path, png_error_string(ret));
				continue;
			}

			png_close_file(&image);

			/* Get reference sizes if first time */
			if( *w == -1 ){
				*w = image.width;
				*h = image.height;
				*b = image.bpp;

				printf("INFO: First PNG loaded. "\
						"Sizes: %ldx%ld (%ld Bpp) (pattern %ld KB) (raw %ld KB)\n",
						*w, *h, *b, (sizeof(double) * *w * *h)/1024, (*w * *h)/1024);
			}


			/* Valid png image */
			if( *w == image.width && *h == image.height && *b == image.bpp ){

				/* Copy full path to the png_paths list */
				(*png_paths)[npngs] = (char *) malloc (strlen(full_png_path));
				if( (*png_paths)[npngs] == NULL ){
					printerr("ERROR: Out of memory for png pathname.\n");
					return 0;
				}

				strcpy((*png_paths)[npngs], full_png_path);

				/* Set code for dir */
				(*png_codes)[npngs] = ndirvalid;

				++ndirvalidpngs;
				++npngs;


				/* Incompatible image */
			} else {
				printerr("WARNING: Ignoring PNG file '%s'. It's %dx%d (%d Bpp)"\
						" instead of %ldx%ld (%ld Bpp) as it should be.\n",
						full_png_path, image.width, image.height,
						image.bpp, *w, *h, *b);
			}

		}

		if( ndirvalidpngs == 0 ) {
			printerr("WARNING: No valid PNG file was read from '%s' dir.\n",
					full_dir_path);
		} else {
			/* Another valid dir.  Copy its name */
			(*pset_names)[ndirvalid] = (char *) malloc (strlen(dirs[d]->d_name));
			strcpy((*pset_names)[ndirvalid], dirs[d]->d_name);
			++ndirvalid;
		}

	}

	*npats = npngs;
	*npsets = ndirvalid;

	return npngs;
}

int patternset_readpath(patternset * pset_ptr, const char * dir_path) {
	size_t npngs = 0, npats = 0, npsets = 0, i = 0, w, h, bpp;
	int ret = 0;
	unsigned char * rawdata = NULL;
	char ** png_paths = NULL;
	patternset pset = NULL;
	png_t image;

	/* Create patternset */
	pset = (patternset) malloc (sizeof(patternset_t));

	/* List all pngs to be read */
	if( (npngs = list_valid_pngs(dir_path, &npats, &npsets, &w, &h, &bpp,
					&png_paths, &pset->codes, &(pset->names))) <= 0 ) {
		free(pset);
		return TRUE;
	}

	/* Initialize patternset values */
	patternset_init(pset, npsets, npats, w*h);
	pset->w = w;
	pset->h = h;
	pset->bpp = bpp;
	pset->size = w * h * bpp;

	/* Temp buffer to store raw image data
	 * before converting it to double in the pattern */
	rawdata = (unsigned char *) malloc (sizeof(unsigned char) * pset->size);

	/* Initialize pnglite */
	png_init(NULL, NULL);

	/* Open each valid image */
	for(i = 0; i < npngs; ++i){
		if( (ret = png_open_file(&image, png_paths[i])) != PNG_NO_ERROR) {
			printerr("WARNING: Couldn't open PNG image: '%s': %s\n",
					png_paths[i], png_error_string(ret));
			continue;
		}

		/* Get png raw data, convert it to double and set it as
		 * pattern input, associating it with the patternset
		 * directory code */
		if( (ret = png_get_data(&image, rawdata)) != PNG_NO_ERROR){
			printerr("WARNING: Couldn't get data from '%s': %s. Ignoring file.\n",
					png_paths[i], png_error_string(ret));
		}

		png_close_file(&image);
	}

	for(i = 0; i < npngs; ++i)
		free(png_paths[i]);
	free(png_paths);

	printerr("Pattern loading finished. %zd patterns read from '%s'\n", npats, dir_path);

	if( npats > 0 ) {
		/* Set patternset values */
		pset->npats = npats;
		pset->npsets = npsets;
		pset->ni = pset->size / pset->bpp;
		pset->no = npsets;
		*pset_ptr = pset;
	} else {
		free(pset);
	}

	/* Free temporals */
	if( rawdata != NULL )
		free(rawdata);

	return npats;
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

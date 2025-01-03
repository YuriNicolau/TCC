#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define DT 0.002f // delta t
#define DZ 20.0f // delta z
#define DX 20.0f // delta x
#define DY 20.0f // delta y
#define V 1500.0f // wave velocity v = 1500 m/s

// SPACE_ORDER: number of neighbors to calculate. 
// Accept values are 2, 4, 6, 8, 10, 12, 14 or 16
#ifndef SPACE_ORDER
#define SPACE_ORDER 2
#endif

// STENCIL_RADIUS: half of spatial order
#define STENCIL_RADIUS SPACE_ORDER/2

/*
 * save the matrix on a file.txt
 */
void save_grid(size_t z_slice, size_t nx, size_t ny, double *grid){

	system("mkdir -p wavefield");	  

	char file_name[32];
	sprintf(file_name, "wavefield/wavefield.txt");

	// save the result
	FILE *file;
	file = fopen(file_name, "w");

	for(size_t i = STENCIL_RADIUS; i < nx - STENCIL_RADIUS; i++) {

		size_t offset = (z_slice * nx + i) * ny;

		for(size_t j = STENCIL_RADIUS; j < ny - STENCIL_RADIUS; j++) {
			fprintf(file, "%lf ", grid[offset + j]);
		}
		fprintf(file, "\n");
	}

	fclose(file);
}


int main(int argc, char* argv[]) {

	// validate the parameters
	if(argc != 5){
		printf("Usage: ./stencil N1 N2 N3 ITERATIONS\n");
		printf("N1 N2 N3: grid sizes for the stencil\n");
		printf("ITERATIONS: number of timesteps\n");		
		exit(-1);
	}	

	// number of grid points in Z
	size_t nz = atoi(argv[1]);

	// number of grid points in X
	size_t nx = atoi(argv[2]);

	// number of grid points in Y
	size_t ny = atoi(argv[3]);

	// number of timesteps
	size_t iterations = atoi(argv[4]);   

	// validate the spatial order
	if( SPACE_ORDER % 2 != 0 || SPACE_ORDER < 2 || SPACE_ORDER > 16 ){
		printf("ERROR: spatial order must be 2, 4, 6, 8, 10, 12, 14 or 16\n");
		exit(-1);
	}	

	printf("Grid Sizes: %ld x %ld x %ld\n", nz, nx, ny);
	printf("Iterations: %ld\n", iterations);
	printf("Spatial Order: %d\n", SPACE_ORDER);

	// add the spatial order (halo zone) to the grid size
	nz += SPACE_ORDER;
	nx += SPACE_ORDER;
	ny += SPACE_ORDER;	

	// ************* BEGIN INITIALIZATION *************

	printf("Initializing ... \n");

	// array of coefficients
	double *coefficient = (double*) malloc( (STENCIL_RADIUS + 1) * sizeof(double));

	// get the coefficients for the specific spatial ordem
	switch (SPACE_ORDER){
		case 2:
			coefficient[0] = -2.0;
			coefficient[1] = 1.0;
			break;

		case 4:
			coefficient[0] = -2.50000e+0;
			coefficient[1] = 1.33333e+0;
			coefficient[2] = -8.33333e-2;
			break;

		case 6:
			coefficient[0] = -2.72222e+0;
			coefficient[1] = 1.50000e+0;
			coefficient[2] = -1.50000e-1;
			coefficient[3] = 1.11111e-2;
			break;

		case 8:
			coefficient[0] = -2.84722e+0;
			coefficient[1] = 1.60000e+0;
			coefficient[2] = -2.00000e-1;
			coefficient[3] = 2.53968e-2;
			coefficient[4] = -1.78571e-3;
			break;

		case 10:
			coefficient[0] = -2.92722e+0;
			coefficient[1] = 1.66667e+0;
			coefficient[2] = -2.38095e-1;
			coefficient[3] = 3.96825e-2;
			coefficient[4] = -4.96032e-3;
			coefficient[5] = 3.17460e-4;
			break;

		case 12:
			coefficient[0] = -2.98278e+0;
			coefficient[1] = 1.71429e+0;
			coefficient[2] = -2.67857e-1;
			coefficient[3] = 5.29101e-2;
			coefficient[4] = -8.92857e-3;
			coefficient[5] = 1.03896e-3;
			coefficient[6] = -6.01251e-5;
			break;

		case 14:
			coefficient[0] = -3.02359e+0;
			coefficient[1] = 1.75000e+0;
			coefficient[2] = -2.91667e-1;
			coefficient[3] = 6.48148e-2;
			coefficient[4] = -1.32576e-2;
			coefficient[5] = 2.12121e-3;
			coefficient[6] = -2.26625e-4;
			coefficient[7] = 1.18929e-5;
			break;

		case 16:
			coefficient[0] = -3.05484e+0;
			coefficient[1] = 1.77778e+0;
			coefficient[2] = -3.11111e-1;
			coefficient[3] = 7.54209e-2;
			coefficient[4] = -1.76768e-2;
			coefficient[5] = 3.48096e-3;
			coefficient[6] = -5.18001e-4;
			coefficient[7] = 5.07429e-5;
			coefficient[8] = -2.42813e-6;
			break;
	}

	// represent the matrix of wavefield as an array
	double *prev_u = (double*) malloc(nz * nx * ny * sizeof(double));
	double *next_u = (double*) malloc(nz * nx * ny * sizeof(double));

	// represent the matrix of velocities as an array
	double *vel_model = (double*) malloc(nz * nx * ny * sizeof(double));  

	// initialize matrix
	#pragma omp parallel for
	for(size_t i = 0; i < nz; i++){
		for(size_t j = 0; j < nx; j++){
			for(size_t k = 0; k < ny; k++){
				size_t offset = (i * nx + j) * ny + k;
				prev_u[offset] = 0.0;
				next_u[offset] = 0.0;
				vel_model[offset] = V;
			}
		}
	}

	omp_set_default_device(omp_get_default_device());

	// Add a source to initial wavefield as an initial condition
	double val = 1.f;	 

	// add a source to initial wavefield as an initial condition
	for (int s = 4; s >= 0; s--) {
		for (int i = nz / 2 - s; i < nz / 2 + s; i++) {
			for (int j = nx / 2 - s; j < nx / 2 + s; j++) {

				size_t offset = (i * nx + j) * ny;

				for (int k = ny / 2 - s; k < ny / 2 + s; k++) {
					prev_u[offset + k] = val;
				}
			}
		}
		val *= 0.9;
	}

	// ************** END INITIALIZATION **************

	printf("Computing wavefield ... \n");   

	double dzSquared = DZ * DZ;
	double dxSquared = DX * DX;
	double dySquared = DY * DY;
	double dtSquared = DT * DT;

	// variable to measure execution time
	struct timeval time_start;
	struct timeval time_end;

	// get the start time
	gettimeofday(&time_start, NULL);

	#pragma omp target enter data map(to: coefficient[:(STENCIL_RADIUS + 1)])
	#pragma omp target enter data map(to: prev_u[:(nz * nx * ny)])
	#pragma omp target enter data map(to: next_u[:(nz * nx * ny)])
	#pragma omp target enter data map(to: vel_model[:(nz * nx * ny)])

	// wavefield modeling
	for(size_t n = 0; n < iterations; n++) {

	#pragma omp target teams distribute parallel for collapse(3)
		for(size_t i = STENCIL_RADIUS; i < nz - STENCIL_RADIUS; i++) {
			for(size_t j = STENCIL_RADIUS; j < nx - STENCIL_RADIUS; j++) {
				for(size_t k = STENCIL_RADIUS; k < ny - STENCIL_RADIUS; k++) {
					// index of the current point in the grid
					size_t current = (i * nx + j) * ny + k;

					// stencil code to update grid
					double value = coefficient[0] * (prev_u[current]/dzSquared + prev_u[current]/dxSquared + prev_u[current]/dySquared);

					// radius of the stencil
					for(size_t ir = 1; ir <= STENCIL_RADIUS; ir++){
						value += coefficient[ir] * (
								( (prev_u[current + ir] + prev_u[current - ir]) / dySquared ) + //neighbors in Y direction
								( (prev_u[current + (ir * ny)] + prev_u[current - (ir * ny)]) / dxSquared ) + //neighbors in X direction
								( (prev_u[current + (ir * nx * ny)] + prev_u[current - (ir * nx * ny)]) / dzSquared )); //neighbors in Z direction
					}
					value *= dtSquared * vel_model[current] * vel_model[current];
					next_u[current] = 2.0 * prev_u[current] - next_u[current] + value;
				}
			}
		}

		// swap arrays for next iteration
		double *swap = next_u;
		next_u = prev_u;
		prev_u = swap;		
	}

	/*
	 * Testar aqui a compressão
	 */
	size_t compressed_size;
	size_t max_compressed_size;
	size_t uncompressed_size;

	double *compressed_next_u;

	compressed_next_u = (double*)omp_target_alloc(max_compressed_size, omp_get_default_device());

	//*/

	#pragma omp target exit data map(from: next_u[:(nz * nx * ny)])

	// get the end time
	gettimeofday(&time_end, NULL);

	double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) + (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

	// get a slice in the Z middle
	size_t z_slice = (int) nz / 2;

	//printf("Saving uncompressed...\n");
	//save_grid(z_slice, compressed_size/sizeof(double), 1, next_u);
	//printf("Saving compressed...\n");
	//save_grid(z_slice, nx, ny, compressed_next_u);

	printf("Iterations completed in %f seconds \n", exec_time);

	free(prev_u);
	free(next_u);
	free(vel_model);

	return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#if defined(CPU_OPENMP) || defined(GPU_OPENMP)
    #include <omp.h>
#endif

// use single (float) or double precision
// according to the value passed in the compilation cmd
#if defined(FLOAT)
   typedef float f_type;
#elif defined(DOUBLE)
   typedef double f_type;
#endif

// Forward without snaphots saving
extern "C" double forward(f_type *u, f_type *velocity, f_type *damp,
               f_type *wavelet, size_t wavelet_size, size_t wavelet_count,
               f_type *coeff, 
               size_t *src_points_interval, size_t src_points_interval_size,
               f_type *src_points_values, size_t src_points_values_size,
               size_t *src_points_values_offset,
               size_t *rec_points_interval, size_t rec_points_interval_size,
               f_type *rec_points_values, size_t rec_points_values_size,
               size_t *rec_points_values_offset,
               f_type *receivers, size_t num_sources, size_t num_receivers,
               size_t nz, size_t nx, size_t ny, f_type dz, f_type dx, f_type dy,
               f_type dt,
               size_t begin_timestep, size_t end_timestep,
               size_t space_order){

    size_t stencil_radius = space_order / 2;

    size_t domain_size = nz * nx * ny;

    f_type dzSquared = dz * dz;
    f_type dxSquared = dx * dx;
    f_type dySquared = dy * dy;
    f_type dtSquared = dt * dt;

    // timestep pointers
    size_t prev_t = 0;
    size_t current_t = 1;
    size_t next_t = 2;

    // variable to measure execution time
    struct timeval time_start;
    struct timeval time_end;

    // get the start time
    gettimeofday(&time_start, NULL);

    // select the device
    #ifdef DEVICEID
    omp_set_default_device(DEVICEID);
    #endif   

    size_t shot_record_size = wavelet_size * num_receivers;
    size_t u_size = 3 * domain_size;

    #pragma omp target enter data map(to: u[:u_size])
    #pragma omp target enter data map(to: velocity[:domain_size])
    #pragma omp target enter data map(to: damp[:domain_size])
    #pragma omp target enter data map(to: coeff[:stencil_radius+1])
    #pragma omp target enter data map(to: src_points_interval[:src_points_interval_size])
    #pragma omp target enter data map(to: src_points_values[:src_points_values_size])
    #pragma omp target enter data map(to: src_points_values_offset[:num_sources])
    #pragma omp target enter data map(to: rec_points_interval[:rec_points_interval_size])
    #pragma omp target enter data map(to: rec_points_values[:rec_points_values_size])
    #pragma omp target enter data map(to: rec_points_values_offset[:num_receivers])
    #pragma omp target enter data map(to: wavelet[:wavelet_size * wavelet_count])
    #pragma omp target enter data map(to: receivers[:shot_record_size])
    
        
    // wavefield modeling
    for(size_t n = begin_timestep; n <= end_timestep; n++) {
        
        prev_t = (n - 1) % 3;
        current_t = n % 3;
        next_t = (n + 1) % 3; 

        /*
            Section 1: update the wavefield according to the acoustic wave equation
        */
        
        #pragma omp target teams distribute parallel for collapse(3)
        for(size_t i = stencil_radius; i < nz - stencil_radius; i++) {
            for(size_t j = stencil_radius; j < nx - stencil_radius; j++) {
                for(size_t k = stencil_radius; k < ny - stencil_radius; k++) {
                    // index of the current point in the grid
                    size_t domain_offset = (i * nx + j) * ny + k;

                    size_t prev_u = prev_t * domain_size + domain_offset;
                    size_t current_u = current_t * domain_size + domain_offset;
                    size_t next_u = next_t * domain_size + domain_offset;

                    // stencil code to update grid
                    f_type value = 0.0;

                    f_type sum_y = coeff[0] * u[current_u];
                    f_type sum_x = coeff[0] * u[current_u];
                    f_type sum_z = coeff[0] * u[current_u];

                    // radius of the stencil                    
                    for(size_t ir = 1; ir <= stencil_radius; ir++){
                        //neighbors in the Y direction
                        sum_y += coeff[ir] * (u[current_u + ir] + u[current_u - ir]);

                        //neighbors in the X direction
                        sum_x += coeff[ir] * (u[current_u + (ir * ny)] + u[current_u - (ir * ny)]);

                        //neighbors in the Z direction
                        sum_z += coeff[ir] * (u[current_u + (ir * nx * ny)] + u[current_u - (ir * nx * ny)]);
                    }

                    value += sum_y/dySquared + sum_x/dxSquared + sum_z/dzSquared;

                    // parameter to be used
                    f_type slowness = 1.0 / (velocity[domain_offset] * velocity[domain_offset]);

                    // denominator with damp coefficient
                    f_type denominator = (1.0 + damp[domain_offset] * dt / 2);
                    f_type numerator = (1.0 - damp[domain_offset] * dt / 2);

                    value *= (dtSquared / slowness) / denominator;

                    u[next_u] = 2.0 / denominator * u[current_u] - (numerator / denominator) * u[prev_u] + value;
                }
            }
        }

        /*
            Section 2: add the source term
        */
        
        #pragma omp target teams distribute parallel for
        // for each source
        for(size_t src = 0; src < num_sources; src++){

            size_t wavelet_offset = n - 1;

            if(wavelet_count > 1){
                wavelet_offset = (n-1) * num_sources + src;
            }

            if(wavelet[wavelet_offset] != 0.0){

                // each source has 6 (z_b, z_e, x_b, x_e, y_b, y_e) point intervals
                size_t offset_src = src * 6;

                // interval of grid points of the source in the Z axis
                size_t src_z_begin = src_points_interval[offset_src + 0];
                size_t src_z_end = src_points_interval[offset_src + 1];

                // interval of grid points of the source in the X axis
                size_t src_x_begin = src_points_interval[offset_src + 2];
                size_t src_x_end = src_points_interval[offset_src + 3];

                // interval of grid points of the source in the Y axis
                size_t src_y_begin = src_points_interval[offset_src + 4];
                size_t src_y_end = src_points_interval[offset_src + 5];

                // number of grid points of the source in each axis
                size_t src_z_num_points = src_z_end - src_z_begin + 1;
                size_t src_x_num_points = src_x_end - src_x_begin + 1;
                //size_t src_y_num_points = src_y_end - src_y_begin + 1;

                // pointer to src value offset
                size_t offset_src_kws_index_z = src_points_values_offset[src];

                // index of the Kaiser windowed sinc value of the source point
                size_t kws_index_z = offset_src_kws_index_z;

                // for each source point in the Z axis                
                for(size_t i = src_z_begin; i <= src_z_end; i++){
                    size_t kws_index_x = offset_src_kws_index_z + src_z_num_points;

                    // for each source point in the X axis                    
                    for(size_t j = src_x_begin; j <= src_x_end; j++){

                        size_t kws_index_y = offset_src_kws_index_z + src_z_num_points + src_x_num_points;

                        // for each source point in the Y axis                       
                        for(size_t k = src_y_begin; k <= src_y_end; k++){

                            f_type kws = src_points_values[kws_index_z] * src_points_values[kws_index_x] * src_points_values[kws_index_y];

                            // current source point in the grid
                            size_t domain_offset = (i * nx + j) * ny + k;
                            size_t next_u = next_t * domain_size + domain_offset;

                            // parameter to be used
                            f_type slowness = 1.0 / (velocity[domain_offset] * velocity[domain_offset]);

                            // denominator with damp coefficient
                            f_type denominator = (1.0 + damp[domain_offset] * dt / 2);

                            f_type value = dtSquared / slowness * kws * wavelet[wavelet_offset] / denominator;
                            
                            #pragma omp atomic               
                            u[next_u] += value;

                            kws_index_y++;
                        }
                        kws_index_x++;
                    }
                    kws_index_z++;
                }
            }
        }

        /*
            Section 3: compute the receivers
        */
        
        #pragma omp target teams distribute parallel for
        // for each receiver
        for(size_t rec = 0; rec < num_receivers; rec++){

            f_type sum = 0.0;

            // each receiver has 6 (z_b, z_e, x_b, x_e, y_b, y_e) point intervals
            size_t offset_rec = rec * 6;

            // interval of grid points of the receiver in the Z axis
            size_t rec_z_begin = rec_points_interval[offset_rec + 0];
            size_t rec_z_end = rec_points_interval[offset_rec + 1];

            // interval of grid points of the receiver in the X axis
            size_t rec_x_begin = rec_points_interval[offset_rec + 2];
            size_t rec_x_end = rec_points_interval[offset_rec + 3];

            // interval of grid points of the receiver in the Y axis
            size_t rec_y_begin = rec_points_interval[offset_rec + 4];
            size_t rec_y_end = rec_points_interval[offset_rec + 5];

            // number of grid points of the receiver in each axis
            size_t rec_z_num_points = rec_z_end - rec_z_begin + 1;
            size_t rec_x_num_points = rec_x_end - rec_x_begin + 1;
            //size_t rec_y_num_points = rec_y_end - rec_y_begin + 1;

            // pointer to rec value offset
            size_t offset_rec_kws_index_z = rec_points_values_offset[rec];

            // index of the Kaiser windowed sinc value of the receiver point
            size_t kws_index_z = offset_rec_kws_index_z;

            // for each receiver point in the Z axis            
            for(size_t i = rec_z_begin; i <= rec_z_end; i++){
                size_t kws_index_x = offset_rec_kws_index_z + rec_z_num_points;

                // for each receiver point in the X axis                
                for(size_t j = rec_x_begin; j <= rec_x_end; j++){

                    size_t kws_index_y = offset_rec_kws_index_z + rec_z_num_points + rec_x_num_points;

                    // for each source point in the Y axis                    
                    for(size_t k = rec_y_begin; k <= rec_y_end; k++){

                        f_type kws = rec_points_values[kws_index_z] * rec_points_values[kws_index_x] * rec_points_values[kws_index_y];

                        // current receiver point in the grid
                        size_t domain_offset = (i * nx + j) * ny + k;
                        size_t current_u = current_t * domain_size + domain_offset;                   
                        sum += u[current_u] * kws;

                        kws_index_y++;
                    }
                    kws_index_x++;
                }
                kws_index_z++;
            }

            size_t current_rec_n = (n-1) * num_receivers + rec;
            receivers[current_rec_n] = sum;
        }

    }

    
    #pragma omp target exit data map(from: receivers[:shot_record_size])
    #pragma omp target exit data map(from: u[:u_size])

    #pragma omp target exit data map(delete: velocity[:domain_size])
    #pragma omp target exit data map(delete: damp[:domain_size])
    #pragma omp target exit data map(delete: coeff[:stencil_radius+1])
    #pragma omp target exit data map(delete: src_points_interval[:src_points_interval_size])
    #pragma omp target exit data map(delete: src_points_values[:src_points_values_size])
    #pragma omp target exit data map(delete: src_points_values_offset[:num_sources])
    #pragma omp target exit data map(delete: rec_points_interval[:rec_points_interval_size])
    #pragma omp target exit data map(delete: rec_points_values[:rec_points_values_size])
    #pragma omp target exit data map(delete: rec_points_values_offset[:num_receivers])
    #pragma omp target exit data map(delete: wavelet[:wavelet_size * wavelet_count])

    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) + (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    return exec_time;
}

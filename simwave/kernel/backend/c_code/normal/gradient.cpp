#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <stack>
#include <iostream>

using namespace std;

// use single (float) or double precision
// according to the value passed in the compilation cmd
#if defined(FLOAT)
   typedef float f_type;
#elif defined(DOUBLE)
   typedef double f_type;
#endif

// Checkpoint struct
struct CheckpointStruct{             
  int index;
  int timestep;  
  f_type *prev;
  f_type *current;
};     

// Forward with snaphots saving
stack<CheckpointStruct> forward_saving(f_type *velocity, f_type *damp,
                                       f_type *wavelet, size_t wavelet_size, size_t wavelet_count,
                                       f_type *coeff, 
                                       size_t *src_points_interval, size_t src_points_interval_size,
                                       f_type *src_points_values, size_t src_points_values_size,
                                       size_t *src_points_values_offset,               
                                       size_t num_sources, size_t num_receivers,
                                       size_t nz, size_t nx, size_t ny, f_type dz, f_type dx, f_type dy,
                                       size_t saving_stride, f_type dt,
                                       size_t begin_timestep, size_t end_timestep,
                                       size_t space_order, size_t num_snapshots, int device_id){

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

    size_t snapshot_index = 0;

    size_t u_size = 3 * domain_size;

    // alocate memory for u
    f_type *u = (f_type*) malloc(u_size * sizeof(f_type));

    #pragma omp target enter data map(to: u[:u_size])

    // init u with zeros
    #pragma omp target teams distribute parallel for collapse(3)
    for(size_t i = 0; i < nz; i++){
        for(size_t j = 0; j < nx; j++){
            for(size_t k = 0; k < ny; k++){
                size_t domain_offset = (i * nx + j) * ny + k;

                size_t prev_u = prev_t * domain_size + domain_offset;
                size_t current_u = current_t * domain_size + domain_offset;
                size_t next_u = next_t * domain_size + domain_offset;

                u[prev_u] = (f_type) 0.0;
                u[current_u] = (f_type) 0.0;
                u[next_u] = (f_type) 0.0;
            }
        }
    }

    // create a stack of snapshots
    stack<CheckpointStruct> snapshots;
        
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

                    size_t prev_u= prev_t * domain_size + domain_offset;
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
            Section 4: save a snapshot
        */
       if( (n-1) % saving_stride == 0 ){

            printf("Salvando %ld [%ld]\n", n, snapshot_index); 

            // create the checkpoint
            CheckpointStruct checkpoint;
            checkpoint.index  = snapshot_index;
            checkpoint.timestep  = (int) n;
            checkpoint.prev = (f_type*) malloc(domain_size * sizeof(f_type));
            checkpoint.current = (f_type*) malloc(domain_size * sizeof(f_type));            
            
            #pragma omp target update from(u[:u_size])
            #pragma omp parallel for simd

            for(size_t i = 0; i < nz; i++) {
                for(size_t j = 0; j < nx; j++) {
                    for(size_t k = 0; k < ny; k++) {

                        // index of the current point in the grid
                        size_t domain_offset = (i * nx + j) * ny + k;                        

                        // current and prev states
                        size_t prev_u = prev_t * domain_size + domain_offset;
                        size_t current_u = current_t * domain_size + domain_offset;

                        checkpoint.prev[domain_offset] = u[prev_u];
                        checkpoint.current[domain_offset] = u[current_u]; 

                    }
                }
            }

            snapshots.push(checkpoint);
            snapshot_index++;            
       }

    }

    #pragma omp target exit data map(delete: u[:u_size])
    free(u);

    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) + (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0; 

    cout << "\nRun Forward in " << exec_time << " seconds\n"; 

    return snapshots;
}


// forward from checkpoint
f_type* forward_checkpoint(f_type *u, f_type *snapshot_d_prev, f_type *snapshot_d_current, f_type *velocity, f_type *damp,
                          f_type *wavelet, size_t wavelet_size, size_t wavelet_count,               
                          f_type *coeff, 
                          size_t *src_points_interval, size_t src_points_interval_size,
                          f_type *src_points_values, size_t src_points_values_size,
                          size_t *src_points_values_offset,
                          size_t num_sources,
                          size_t nz, size_t nx, size_t ny, f_type dz, f_type dx, f_type dy,
                          f_type dt,
                          size_t begin_timestep, size_t end_timestep,
                          size_t space_order, int device_id){    

    size_t stencil_radius = space_order / 2;

    size_t domain_size = nz * nx * ny;

    f_type dzSquared = dz * dz;
    f_type dxSquared = dx * dx;
    f_type dySquared = dy * dy;
    f_type dtSquared = dt * dt;

    // timestep pointers
    size_t prev_t = (begin_timestep - 1) % 3;
    size_t current_t = begin_timestep % 3;
    size_t next_t = (begin_timestep + 1) % 3;

    // variable to measure execution time
    struct timeval time_start;
    struct timeval time_end;

    // get the start time
    gettimeofday(&time_start, NULL);

    // set u with current snapshot    
    #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(u, snapshot_d_prev, snapshot_d_current) device(device_id)
    for(size_t i = 0; i < nz; i++){
        for(size_t j = 0; j < nx; j++){
            for(size_t k = 0; k < ny; k++){
                size_t domain_offset = (i * nx + j) * ny + k;

                size_t prev_u = prev_t * domain_size + domain_offset;
                size_t current_u = current_t * domain_size + domain_offset;
                size_t next_u = next_t * domain_size + domain_offset;

                u[prev_u] = snapshot_d_prev[domain_offset];
                u[current_u] = snapshot_d_current[domain_offset];
                u[next_u] = (f_type) 0.0;
            }
        }
    }    
        
    // wavefield modeling
    for(size_t n = begin_timestep; n <= end_timestep; n++) {
        
        prev_t = (n - 1) % 3;
        current_t = n % 3;
        next_t = (n + 1) % 3; 

        /*
            Section 1: update the wavefield according to the acoustic wave equation
        */        
        #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(u) device(device_id)
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
        
        #pragma omp target teams distribute parallel for is_device_ptr(u) device(device_id)
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

    }

    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) + (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    //return exec_time;

    size_t current_u = current_t * domain_size;    
    return &u[current_u];
}

// 3D gradient
extern "C" double gradient(f_type *v, f_type *grad, f_type *velocity, f_type *damp,
               f_type *wavelet_forward, size_t wavelet_forward_size, size_t wavelet_forward_count,
               f_type *wavelet_adjoint, size_t wavelet_adjoint_size, size_t wavelet_adjoint_count,
               f_type *coeff, 
               size_t *src_points_interval, size_t src_points_interval_size,
               f_type *src_points_values, size_t src_points_values_size,
               size_t *src_points_values_offset,               
               size_t *rec_points_interval, size_t rec_points_interval_size,
               f_type *rec_points_values, size_t rec_points_values_size,
               size_t *rec_points_values_offset,
               size_t num_sources, size_t num_receivers,
               size_t nz, size_t nx, size_t ny, f_type dz, f_type dx, f_type dy,
               size_t saving_stride, f_type dt,
               size_t begin_timestep, size_t end_timestep,
               size_t space_order, size_t num_snapshots){

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

    // select the device
    #ifdef DEVICEID
    omp_set_default_device(DEVICEID);
    #endif

    // get default device
    int device_id = omp_get_default_device();
    int host = omp_get_initial_device();
   
    size_t u_size = 3 * domain_size;
    size_t v_size = 3 * domain_size; // prev, current, next
    
    #pragma omp target enter data map(to: v[:v_size])
    #pragma omp target enter data map(to: grad[:domain_size])
    #pragma omp target enter data map(to: velocity[:domain_size])
    #pragma omp target enter data map(to: damp[:domain_size])
    #pragma omp target enter data map(to: coeff[:stencil_radius+1])
    #pragma omp target enter data map(to: src_points_interval[:src_points_interval_size])
    #pragma omp target enter data map(to: src_points_values[:src_points_values_size])
    #pragma omp target enter data map(to: src_points_values_offset[:num_sources])
    #pragma omp target enter data map(to: rec_points_interval[:rec_points_interval_size])
    #pragma omp target enter data map(to: rec_points_values[:rec_points_values_size])
    #pragma omp target enter data map(to: rec_points_values_offset[:num_receivers])
    #pragma omp target enter data map(to: wavelet_forward[:wavelet_forward_size * wavelet_forward_count])
    #pragma omp target enter data map(to: wavelet_adjoint[:wavelet_adjoint_size * wavelet_adjoint_count])

    // allocates memory for u on the device
    // used by forward    
    f_type *u = (f_type*) omp_target_alloc(3 * domain_size * sizeof(f_type), device_id);
    
    /** run forward to calculate all checkpoints **/

    printf("\nRunning Forward to save %ld checkpoints\n", num_snapshots);

    stack<CheckpointStruct> snapshots = forward_saving(velocity, damp,
                                                        wavelet_forward, wavelet_forward_size, wavelet_forward_count,
                                                        coeff, 
                                                        src_points_interval, src_points_interval_size,
                                                        src_points_values, src_points_values_size,
                                                        src_points_values_offset,              
                                                        num_sources, num_receivers,
                                                        nz, nx, ny, dz, dx, dy, saving_stride, dt,
                                                        begin_timestep, end_timestep,
                                                        space_order, num_snapshots, device_id);    

    printf("\nRunning Gradient\n");

    // get the start time
    gettimeofday(&time_start, NULL);

    // allocate memory for the snapshot on the device    
    f_type *snapshot_d_prev = (f_type*) omp_target_alloc(domain_size * sizeof(f_type), device_id);
    f_type *snapshot_d_current = (f_type*) omp_target_alloc(domain_size * sizeof(f_type), device_id);

    // copy current checkpoint
    omp_target_memcpy(snapshot_d_prev, snapshots.top().prev, domain_size * sizeof(f_type), 0, 0, device_id, host);
    omp_target_memcpy(snapshot_d_current, snapshots.top().current, domain_size * sizeof(f_type), 0, 0, device_id, host);
    printf("Copying checkpoint %d in timestep %ld\n", snapshots.top().index, end_timestep);
    
    // wavefield modeling
    for(size_t n = end_timestep; n >= begin_timestep; n--) {

        // no saving case
        // since we are moving backwards, we invert the next_t and prev_t pointer       
        next_t = (n - 1) % 3;
        current_t = n % 3;
        prev_t = (n + 1) % 3;

        /*
            Section 1: update the wavefield according to the acoustic wave equation
        */
       
        #pragma omp target teams distribute parallel for collapse(3)
        for(size_t i = stencil_radius; i < nz - stencil_radius; i++) {
            for(size_t j = stencil_radius; j < nx - stencil_radius; j++) {
                for(size_t k = stencil_radius; k < ny - stencil_radius; k++) {
                    // index of the current point in the grid
                    size_t domain_offset = (i * nx + j) * ny + k;

                    size_t prev_v = prev_t * domain_size + domain_offset;
                    size_t current_v = current_t * domain_size + domain_offset;
                    size_t next_v = next_t * domain_size + domain_offset;

                    // stencil code to update grid
                    f_type value = 0.0;

                    f_type sum_y = coeff[0] * v[current_v];
                    f_type sum_x = coeff[0] * v[current_v];
                    f_type sum_z = coeff[0] * v[current_v];

                    // radius of the stencil                    
                    for(size_t ir = 1; ir <= stencil_radius; ir++){
                        //neighbors in the Y direction
                        sum_y += coeff[ir] * (v[current_v + ir] + v[current_v - ir]);

                        //neighbors in the X direction
                        sum_x += coeff[ir] * (v[current_v + (ir * ny)] + v[current_v - (ir * ny)]);

                        //neighbors in the Z direction
                        sum_z += coeff[ir] * (v[current_v + (ir * nx * ny)] + v[current_v - (ir * nx * ny)]);
                    }

                    value += sum_y/dySquared + sum_x/dxSquared + sum_z/dzSquared;

                    // parameter to be used
                    f_type slowness = 1.0 / (velocity[domain_offset] * velocity[domain_offset]);

                    // denominator with damp coefficient
                    f_type denominator = (1.0 + damp[domain_offset] * dt / 2);
                    f_type numerator = (1.0 - damp[domain_offset] * dt / 2);

                    value *= (dtSquared / slowness) / denominator;

                    v[next_v] = 2.0 / denominator * v[current_v] - (numerator / denominator) * v[prev_v] + value;
                }
            }
        }

        /*
            Section 2: add the source term

            The receivers are the sources the in the adjoint
        */

        
        #pragma omp target teams distribute parallel for

        // for each receiver
        for(size_t rec = 0; rec < num_receivers; rec++){

            size_t wavelet_offset = n - 1;

            if(wavelet_adjoint_count > 1){
                wavelet_offset = (n-1) * num_receivers + rec;
            }

            if(wavelet_adjoint[wavelet_offset] != 0.0){

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

                    // for each source point in the X axis                    
                    for(size_t j = rec_x_begin; j <= rec_x_end; j++){

                        size_t kws_index_y = offset_rec_kws_index_z + rec_z_num_points + rec_x_num_points;

                        // for each source point in the Y axis                        
                        for(size_t k = rec_y_begin; k <= rec_y_end; k++){

                            f_type kws = rec_points_values[kws_index_z] * rec_points_values[kws_index_x] * rec_points_values[kws_index_y];

                            // current source point in the grid
                            size_t domain_offset = (i * nx + j) * ny + k;
                            size_t next_v = next_t * domain_size + domain_offset;

                            // parameter to be used
                            f_type slowness = 1.0 / (velocity[domain_offset] * velocity[domain_offset]);

                            // denominator with damp coefficient
                            f_type denominator = (1.0 + damp[domain_offset] * dt / 2);

                            f_type value = dtSquared / slowness * kws * wavelet_adjoint[wavelet_offset] / denominator;
                            
                            #pragma omp atomic                            
                            v[next_v] += value;

                            kws_index_y++;
                        }
                        kws_index_x++;
                    }
                    kws_index_z++;
                }
            }
        }

        /*
            Section 3: get current snapshot
        */
        f_type *u_snapshot;            

        if( (n-1) % saving_stride == 0 ){
            
            // here we use the checkpoint         
           
            printf("Recovering checkpoint %d in timestep %ld\n", snapshots.top().index, n);

            u_snapshot = snapshot_d_current;                        

        }else{                       
            // recompute
            u_snapshot = forward_checkpoint(u, snapshot_d_prev, snapshot_d_current, velocity, damp,
                          wavelet_forward, wavelet_forward_size, wavelet_forward_count,               
                          coeff, 
                          src_points_interval, src_points_interval_size,
                          src_points_values, src_points_values_size,
                          src_points_values_offset,
                          num_sources,
                          nz, nx, ny, dz, dx, dy,
                          dt,
                          (snapshots.top().index * saving_stride + 1) , n,
                          space_order, device_id);

            //printf("Recompute %ld using chekpoint %d : [%ld - %ld]\n", n, checkpoint_id, (checkpoint_id * saving_stride + 1), n);

        }


        /*
            Section 4: gradient calculation
        */        
        #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(u_snapshot) device(device_id)      
        for(size_t i = stencil_radius; i < nz - stencil_radius; i++) {
            for(size_t j = stencil_radius; j < nx - stencil_radius; j++) {
                for(size_t k = stencil_radius; k < ny - stencil_radius; k++) {
                    // index of the current point in the grid
                    size_t domain_offset = (i * nx + j) * ny + k;

                    size_t prev_v = prev_t * domain_size + domain_offset;
                    size_t current_v = current_t * domain_size + domain_offset;
                    size_t next_v = next_t * domain_size + domain_offset;

                    f_type v_second_time_derivative = (v[prev_v] - 2.0 * v[current_v] + v[next_v]) / dtSquared;

                    // update gradient
                    grad[domain_offset] -= v_second_time_derivative * u_snapshot[domain_offset];
                }
            }
        }

        // remove the recent used snapshot
        if( (n-1) % saving_stride == 0 ){
            // remove the the current snapshot from the stack
            free(snapshots.top().prev);
            free(snapshots.top().current);
            snapshots.pop();
            cout << "---- RAM ----" << endl;
            system("free -g -h");
            cout << "-------------" << endl;

            if (!snapshots.empty()){
                // copy the next checkpoint
                // dst  src
                omp_target_memcpy(snapshot_d_prev, snapshots.top().prev, domain_size * sizeof(f_type), 0, 0, device_id, host);
                omp_target_memcpy(snapshot_d_current, snapshots.top().current, domain_size * sizeof(f_type), 0, 0, device_id, host);
                printf("Copying checkpoint %d in timestep %ld\n", snapshots.top().index, n);
            }
        }

    }
    
    #pragma omp target exit data map(from: grad[:domain_size])    
    #pragma omp target exit data map(delete: v[:v_size])
    #pragma omp target exit data map(delete: velocity[:domain_size])
    #pragma omp target exit data map(delete: damp[:domain_size])
    #pragma omp target exit data map(delete: coeff[:stencil_radius+1])
    #pragma omp target exit data map(delete: src_points_interval[:src_points_interval_size])
    #pragma omp target exit data map(delete: src_points_values[:src_points_values_size])
    #pragma omp target exit data map(delete: src_points_values_offset[:num_sources])
    #pragma omp target exit data map(delete: rec_points_interval[:rec_points_interval_size])
    #pragma omp target exit data map(delete: rec_points_values[:rec_points_values_size])
    #pragma omp target exit data map(delete: rec_points_values_offset[:num_receivers])  
    #pragma omp target exit data map(delete: wavelet_forward[:wavelet_forward_size * wavelet_forward_count])
    #pragma omp target exit data map(delete: wavelet_adjoint[:wavelet_adjoint_size * wavelet_adjoint_count])

    omp_target_free(snapshot_d_prev, device_id);
    omp_target_free(snapshot_d_current, device_id);
    omp_target_free(u, device_id);
    
    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) + (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    return exec_time;
}

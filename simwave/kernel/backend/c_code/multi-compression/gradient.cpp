#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <stack>
#include <iostream>

using namespace std;

typedef float f_type;


#include "nvcomp/nvcomp.h"
#include "nvcomp/nvcompManagerFactory.hpp"

using namespace nvcomp;
using namespace std;

#define CUDA_CHECK(cond)                       \
    do                                         \
    {                                          \
        cudaError_t error = cond;              \
        if (error != cudaSuccess)              \
        {                                      \
            std::cerr << "Falha" << std::endl; \
        }                                      \
    } while (false)

// use single (float) or double precision
// according to the value passed in the compilation cmd
#if defined(FLOAT)
   typedef float f_type;
#elif defined(DOUBLE)
   typedef double f_type;
#endif

#define NUM_DEVICES omp_get_num_devices()

size_t individual_nz            = 0;
size_t individual_domain_size   = 0;

// Checkpoint struct
struct CheckpointStruct{             
  size_t index;
  size_t timestep;  
  f_type *prev;
  f_type *current;
  size_t size_compressed_prev;
  size_t size_compressed_current;
  
  CompressionConfig& configure_compression_prev;
  CompressionConfig& configure_compression_cur;
};     

// Forward with snaphots saving
stack<CheckpointStruct> forward_saving(f_type *d_velocity, f_type *d_damp,
                                       f_type *d_wavelet, size_t wavelet_size, size_t wavelet_count,
                                       f_type *d_coeff, 
                                       size_t *d_src_points_interval, size_t src_points_interval_size,
                                       f_type *d_src_points_values, size_t src_points_values_size,
                                       size_t *d_src_points_values_offset,               
                                       size_t num_sources, size_t num_receivers,
                                       size_t nz, size_t nx, size_t ny, f_type dz, f_type dx, f_type dy,
                                       size_t saving_stride, f_type dt,
                                       size_t begin_timestep, size_t end_timestep,
                                       size_t space_order, size_t num_snapshots, int device_id, f_type *top_u, f_type *bottom_u, f_type *u,
                                       uint8_t *snapshot_d_prev_compressed,  uint8_t *snapshot_d_cur_compressed,
                                       BitcompManager& nvcomp_manager){


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

    printf("Init u with zeros!\n");
    #pragma omp target teams distribute parallel for collapse(3) device(device_id) is_device_ptr(u, d_velocity, d_damp, d_coeff, d_src_points_values, d_src_points_values_offset)
    for(size_t i = 0; i < individual_nz + 2*stencil_radius; i++){
        for(size_t j = 0; j < nx; j++){
            for(size_t k = 0; k < ny; k++){
                size_t domain_offset = (i * nx + j) * ny + k;

                size_t prev_u = prev_t * individual_domain_size + domain_offset;
                size_t current_u = current_t * individual_domain_size + domain_offset;
                size_t next_u = next_t * individual_domain_size + domain_offset;

                u[prev_u] = (f_type) 0.0;
                u[current_u] = (f_type) 0.0;
                u[next_u] = (f_type) 0.0;
            }
        }
    }

    // create a stack of snapshots
    stack<CheckpointStruct> snapshots;

    int i_ini = stencil_radius;
    int i_fim = individual_nz+stencil_radius;

    if (device_id == 0)
        i_ini = stencil_radius + stencil_radius;
    if(device_id == NUM_DEVICES-1)
        i_fim = individual_nz;
        
    // wavefield modeling
    printf("Wavefield modeling!\n");
    for(size_t n = begin_timestep; n <= end_timestep; n++) {
        
        prev_t = (n - 1) % 3;
        current_t = n % 3;
        next_t = (n + 1) % 3; 

        /*
            Section 1: update the wavefield according to the acoustic wave equation
        */
       
        #pragma omp target teams distribute parallel for collapse(3) device(device_id) is_device_ptr(u, d_velocity, d_damp, d_coeff, d_src_points_interval, d_src_points_values, d_src_points_values_offset)
        for(size_t i = i_ini; i < i_fim; i++) {
            for(size_t j = stencil_radius; j < nx - stencil_radius; j++) {
                for(size_t k = stencil_radius; k < ny - stencil_radius; k++) {
                    // index of the current point in the grid
                    size_t domain_offset = (i * nx + j) * ny + k;

                    size_t prev_u= prev_t * individual_domain_size + domain_offset;
                    size_t current_u = current_t * individual_domain_size + domain_offset;
                    size_t next_u = next_t * individual_domain_size + domain_offset;

                    // stencil code to update grid
                    f_type value = 0.0;

                    f_type sum_y = d_coeff[0] * u[current_u];
                    f_type sum_x = d_coeff[0] * u[current_u];
                    f_type sum_z = d_coeff[0] * u[current_u];

                    // radius of the stencil                    
                    for(size_t ir = 1; ir <= stencil_radius; ir++){
                        //neighbors in the Y direction
                        sum_y += d_coeff[ir] * (u[current_u + ir] + u[current_u - ir]);

                        //neighbors in the X direction
                        sum_x += d_coeff[ir] * (u[current_u + (ir * ny)] + u[current_u - (ir * ny)]);

                        //neighbors in the Z direction
                        sum_z += d_coeff[ir] * (u[current_u + (ir * nx * ny)] + u[current_u - (ir * nx * ny)]);
                    }

                    value += sum_y/dySquared + sum_x/dxSquared + sum_z/dzSquared;

                    // parameter to be used
                    f_type slowness = 1.0 / (d_velocity[domain_offset] * d_velocity[domain_offset]);

                    // denominator with damp coefficient
                    f_type denominator = (1.0 + d_damp[domain_offset] * dt / 2);
                    f_type numerator = (1.0 - d_damp[domain_offset] * dt / 2);

                    value *= (dtSquared / slowness) / denominator;

                    u[next_u] = 2.0 / denominator * u[current_u] - (numerator / denominator) * u[prev_u] + value;
                }
            }
        }

        #pragma omp barrier
        if (device_id > 0) {
            cudaMemcpyPeer(top_u + (individual_nz + stencil_radius) * nx * ny * sizeof(f_type), //Dst
                    device_id-1,																//Dst device
                    u + stencil_radius * nx * ny * sizeof(f_type),							  	//Src
                    device_id,																  	//Src device
                    stencil_radius * nx * ny * sizeof(f_type));								 	//Size
        }

        if (device_id < NUM_DEVICES - 1) {
            cudaMemcpyPeer(bottom_u, 							//Dst
                    device_id+1,									//Dst device
                    u + individual_nz * nx * ny * sizeof(f_type), 	//Src
                    device_id,									 	//Src device
                    stencil_radius * nx * ny * sizeof(f_type));	//Size
        }

        #pragma omp barrier


        /*
            Section 2: add the source term
        */
       
        //printf("Adding the source term!! (device = %d)\n", device_id);
        #pragma omp target teams distribute parallel for device(device_id) is_device_ptr(u, d_velocity, d_damp, d_coeff, d_src_points_interval, d_src_points_values, d_src_points_values_offset, d_wavelet)
        for(size_t src = 0; src < num_sources; src++){

            size_t wavelet_offset = n - 1;

            if(wavelet_count > 1){
                wavelet_offset = (n-1) * num_sources + src;
                //wavelet_offset += device_id*individual_domain_size;
            }
            
            //printf("src = %u, wavelet_offset = %u\n", src, wavelet_offset);
            if(d_wavelet[wavelet_offset] != 0.0){

                // each source has 6 (z_b, z_e, x_b, x_e, y_b, y_e) point intervals
                size_t offset_src = src * 6;

                // interval of grid points of the source in the Z axis
                size_t src_z_begin = d_src_points_interval[offset_src + 0];
                size_t src_z_end = d_src_points_interval[offset_src + 1];

                // interval of grid points of the source in the X axis
                size_t src_x_begin = d_src_points_interval[offset_src + 2];
                size_t src_x_end = d_src_points_interval[offset_src + 3];

                // interval of grid points of the source in the Y axis
                size_t src_y_begin = d_src_points_interval[offset_src + 4];
                size_t src_y_end = d_src_points_interval[offset_src + 5];

                // number of grid points of the source in each axis
                size_t src_z_num_points = src_z_end - src_z_begin + 1;
                size_t src_x_num_points = src_x_end - src_x_begin + 1;
                //size_t src_y_num_points = src_y_end - src_y_begin + 1;

                // pointer to src value offset
                size_t offset_src_kws_index_z = d_src_points_values_offset[src];

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

                            f_type kws = d_src_points_values[kws_index_z] * d_src_points_values[kws_index_x] * d_src_points_values[kws_index_y];

                            // current source point in the grid
                            size_t domain_offset = (i * nx + j) * ny + k;
                            size_t next_u = next_t * individual_domain_size + (domain_offset % individual_domain_size);

                            // parameter to be used
                            f_type slowness = 1.0 / (d_velocity[domain_offset % individual_domain_size] * d_velocity[domain_offset % individual_domain_size]);

                            // denominator with damp coefficient
                            f_type denominator = (1.0 + d_damp[domain_offset % individual_domain_size] * dt / 2);

                            f_type value = dtSquared / slowness * kws * d_wavelet[wavelet_offset] / denominator;
                            
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
        f_type *h_u = NULL;
       if( (n-1) % saving_stride == 0 ){

            printf("Salvando %ld [%ld]\n", n, snapshot_index); 

            CompressionConfig compression_configure_prev = nvcomp_manager.configure_compression(individual_domain_size *sizeof(f_type));
            CompressionConfig compression_configure_cur = nvcomp_manager.configure_compression(individual_domain_size *sizeof(f_type));

            nvcomp_manager.compress((const uint8_t *) &u[prev_t * individual_domain_size], snapshot_d_prev_compressed, compression_configure_prev);
            nvcomp_manager.compress((const uint8_t *) &u[current_t * individual_domain_size], snapshot_d_cur_compressed, compression_configure_cur);                
            
            size_t size_compressed_prev = nvcomp_manager.get_compressed_output_size(snapshot_d_prev_compressed);
            size_t size_compressed_cur = nvcomp_manager.get_compressed_output_size(snapshot_d_cur_compressed);
         
            //TODO 2: Colocar o tamanho da compactação na struct para descompressão
            CheckpointStruct checkpoint = {snapshot_index, 
                                           n,
                                           (f_type*) malloc(size_compressed_prev), 
                                           (f_type*) malloc(size_compressed_cur),
                                            size_compressed_prev,
                                            size_compressed_cur,
                                            compression_configure_prev,
                                            compression_configure_cur
                                           };


            printf("Salva %3ld[%ld] ", n, snapshot_index); 
            printf("size(prev_comp)= %8zu, ", checkpoint.size_compressed_prev);
            printf("size(cur_comp)= %8zu, ", checkpoint.size_compressed_current);
            printf("comp_ratio= %lf\n", domain_size *sizeof(f_type) / (double) checkpoint.size_compressed_current);


            // TODO 3: Tentar transferir para CPU o tamanho da compactação
            // obtido no nvcomp_manager.get_compressed_output_size(gpu_comp_buffer)

            //TODO 5: Podemos utilizar cópia assincrona
            CUDA_CHECK( cudaMemcpy(checkpoint.prev, snapshot_d_prev_compressed, checkpoint.size_compressed_prev, cudaMemcpyDeviceToHost)   );  
            CUDA_CHECK( cudaMemcpy(checkpoint.current, snapshot_d_cur_compressed, checkpoint.size_compressed_current, cudaMemcpyDeviceToHost) );
            
            snapshots.push(checkpoint);
            snapshot_index++;            
       }
        free(h_u);
    }

    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) + (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0; 

    cout << "\nRun Forward in " << exec_time << " seconds\n"; 

    return snapshots;
}


// forward from checkpoint
f_type* forward_checkpoint(f_type *u, f_type *snapshot_d_prev, f_type *snapshot_d_current, f_type *d_velocity, f_type *d_damp,
                          f_type *d_wavelet, size_t wavelet_size, size_t wavelet_count,               
                          f_type *d_coeff, 
                          size_t *src_points_interval, size_t src_points_interval_size,
                          f_type *d_src_points_values, size_t src_points_values_size,
                          size_t *src_points_values_offset,
                          size_t num_sources,
                          size_t nz, size_t nx, size_t ny, f_type dz, f_type dx, f_type dy,
                          f_type dt,
                          size_t begin_timestep, size_t end_timestep,
                          size_t space_order, int device_id, f_type *top_u, f_type *bottom_u){

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
    #pragma omp target teams distribute parallel for collapse(3) device(device_id) is_device_ptr(u, snapshot_d_prev, snapshot_d_current, d_velocity, d_damp, d_wavelet, d_coeff, src_points_interval, d_src_points_values, src_points_values_offset, top_u, bottom_u)
    for(size_t i = 0; i < individual_nz; i++){
        for(size_t j = 0; j < nx; j++){
            for(size_t k = 0; k < ny; k++){
                size_t domain_offset = (i * nx + j) * ny + k;

                size_t prev_u = prev_t * individual_domain_size + domain_offset;
                size_t current_u = current_t * individual_domain_size+ domain_offset;
                size_t next_u = next_t * individual_domain_size + domain_offset;

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
        int i_ini = stencil_radius;
        int i_fim = individual_nz+stencil_radius;

        if (device_id == 0)
            i_ini = stencil_radius + stencil_radius;
        if(device_id == NUM_DEVICES-1)
            i_fim = individual_nz;

        #pragma omp target teams distribute parallel for collapse(3) device(device_id) is_device_ptr(u, snapshot_d_prev, snapshot_d_current, d_velocity, d_damp, d_wavelet, d_coeff, src_points_interval, d_src_points_values, src_points_values_offset, top_u, bottom_u)
        for(size_t i = i_ini; i < i_fim; i++) {
            for(size_t j = stencil_radius; j < nx - stencil_radius; j++) {
                for(size_t k = stencil_radius; k < ny - stencil_radius; k++) {
                    // index of the current point in the grid
                    size_t domain_offset = (i * nx + j) * ny + k;

                    size_t prev_u = prev_t * individual_domain_size + domain_offset;
                    size_t current_u = current_t * individual_domain_size + domain_offset;
                    size_t next_u = next_t * individual_domain_size + domain_offset;

                    // stencil code to update grid
                    f_type value = 0.0;

                    f_type sum_y = d_coeff[0] * u[current_u];
                    f_type sum_x = d_coeff[0] * u[current_u];
                    f_type sum_z = d_coeff[0] * u[current_u];

                    // radius of the stencil                    
                    for(size_t ir = 1; ir <= stencil_radius; ir++){
                        //neighbors in the Y direction
                        sum_y += d_coeff[ir] * (u[current_u + ir] + u[current_u - ir]);

                        //neighbors in the X direction
                        sum_x += d_coeff[ir] * (u[current_u + (ir * ny)] + u[current_u - (ir * ny)]);

                        //neighbors in the Z direction
                        sum_z += d_coeff[ir] * (u[current_u + (ir * nx * ny)] + u[current_u - (ir * nx * ny)]);
                    }

                    value += sum_y/dySquared + sum_x/dxSquared + sum_z/dzSquared;

                    // parameter to be used
                    f_type slowness = 1.0 / (d_velocity[domain_offset] * d_velocity[domain_offset]);

                    // denominator with damp coefficient
                    f_type denominator = (1.0 + d_damp[domain_offset] * dt / 2);
                    f_type numerator = (1.0 - d_damp[domain_offset] * dt / 2);

                    value *= (dtSquared / slowness) / denominator;

                    u[next_u] = 2.0 / denominator * u[current_u] - (numerator / denominator) * u[prev_u] + value;
                }
            }
        }

        #pragma omp barrier
        if (device_id > 0) {
            cudaMemcpyPeer(top_u + (individual_nz + stencil_radius) * nx * ny * sizeof(f_type), //Dst
                    device_id-1,																//Dst device
                    u + stencil_radius * nx * ny * sizeof(f_type),							  	//Src
                    device_id,																  	//Src device
                    stencil_radius * nx * ny * sizeof(f_type));								 	//Size
        }

        if (device_id < NUM_DEVICES - 1) {
            cudaMemcpyPeer(bottom_u, 							//Dst
                    device_id+1,									//Dst device
                    u + individual_nz * nx * ny * sizeof(f_type), 	//Src
                    device_id,									 	//Src device
                    stencil_radius * nx * ny * sizeof(f_type));	//Size
        }

        /*
            Section 2: add the source term
        */
        
        #pragma omp target teams distribute parallel for device(device_id) is_device_ptr(u, snapshot_d_prev, snapshot_d_current, d_velocity, d_damp, d_wavelet, d_coeff, src_points_interval, d_src_points_values, src_points_values_offset, top_u, bottom_u)
        // for each source
        for(size_t src = 0; src < num_sources; src++){

            size_t wavelet_offset = n - 1;

            if(wavelet_count > 1){
                wavelet_offset = (n-1) * num_sources + src;
            }

            if(d_wavelet[wavelet_offset] != 0.0){

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

                            f_type kws = d_src_points_values[kws_index_z] * d_src_points_values[kws_index_x] * d_src_points_values[kws_index_y];

                            // current source point in the grid
                            size_t domain_offset = (i * nx + j) * ny + k;
                            size_t next_u = next_t * individual_domain_size + (domain_offset % individual_domain_size);

                            // parameter to be used
                            f_type slowness = 1.0 / (d_velocity[domain_offset % individual_domain_size] * d_velocity[domain_offset % individual_domain_size]);

                            // denominator with damp coefficient
                            f_type denominator = (1.0 + d_damp[domain_offset % individual_domain_size] * dt / 2);

                            f_type value = dtSquared / slowness * kws * d_wavelet[wavelet_offset] / denominator;

                            u[0] += value;

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

    size_t current_u = current_t * individual_domain_size ;    
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

    f_type *us[NUM_DEVICES];

    individual_nz = nz/NUM_DEVICES;
    individual_domain_size = (individual_nz + 2*stencil_radius) * nx * ny;
    size_t domain_size_total = domain_size + 2*stencil_radius * nx * ny;

    int host = omp_get_initial_device();
    #pragma omp parallel num_threads(NUM_DEVICES) shared(us)
    {
        // get default device
        int device_id = omp_get_thread_num();
        cudaSetDevice(device_id);
       
        size_t u_size = 3 * individual_domain_size;
        size_t v_size = 3 * individual_domain_size;

        nvcompStatus_t final_status;
        // TODO 1: Fix - 
        // Prepare compression environment 
        CUDA_CHECK( cudaSetDevice(device_id) );
        cudaStream_t stream;
        CUDA_CHECK(  cudaStreamCreate(&stream)  );
        nvcompType_t data_type = NVCOMP_TYPE_INT;
        BitcompManager nvcomp_manager{data_type, 0, stream, device_id};

        size_t freeMem;
        size_t totalMem;
        CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
        if (freeMem <  individual_domain_size * sizeof(f_type)) {
            std::cout << "Insufficient GPU memory to perform compression." << std::endl;
            exit(1);
        }

        size_t comp_scratch_bytes = nvcomp_manager.get_required_scratch_buffer_size();
        uint8_t* d_comp_scratch;
        CUDA_CHECK(cudaMalloc(&d_comp_scratch, comp_scratch_bytes));
        nvcomp_manager.set_scratch_buffer(d_comp_scratch);

        // const int chunk_size = 1 << 16;
        // nvcompType_t data_type = NVCOMP_TYPE_CHAR;

        // LZ4Manager nvcomp_manager{chunk_size, data_type, stream};    

        printf("domain_size = %zu\n", individual_domain_size *sizeof(f_type));
        // printf("max_compressed_buffer_size = %ld\n", compression_configure.max_compressed_buffer_size);

        uint8_t *snapshot_d_prev_compressed;
        uint8_t *snapshot_d_cur_compressed;

        //Alocate memory on GPU
        CompressionConfig compression_configure_malloc = nvcomp_manager.configure_compression(individual_domain_size *sizeof(f_type));

        CUDA_CHECK( cudaMalloc(&snapshot_d_prev_compressed, compression_configure_malloc.max_compressed_buffer_size) );
        CUDA_CHECK( cudaMalloc(&snapshot_d_cur_compressed, compression_configure_malloc.max_compressed_buffer_size) );
        printf("max size = %zu\n", compression_configure_malloc.max_compressed_buffer_size);


        // allocates memory on the device
		f_type *d_v;
		f_type *d_grad;
		f_type *d_velocity;
		f_type *d_damp;
		f_type *d_coeff;
		size_t *d_src_points_interval;
		f_type *d_src_points_values;
		size_t *d_src_points_values_offset;
		size_t *d_rec_points_interval;
		f_type *d_rec_points_values;
		size_t *d_rec_points_values_offset;
		f_type *d_wavelet_forward;
		f_type *d_wavelet_adjoint;

		cudaMalloc(&d_v,							v_size * sizeof(f_type));
		cudaMalloc(&d_grad,							(domain_size + 2*stencil_radius*nx*ny) * sizeof(f_type));
		cudaMalloc(&d_velocity,						(domain_size + 2*stencil_radius*nx*ny) * sizeof(f_type));
		cudaMalloc(&d_damp,							(domain_size + 2*stencil_radius*nx*ny) * sizeof(f_type));
		cudaMalloc(&d_coeff,						(stencil_radius+1) * sizeof(f_type));
		cudaMalloc(&d_src_points_interval,			src_points_interval_size * sizeof(size_t));
		cudaMalloc(&d_src_points_values,			src_points_values_size * sizeof(f_type));
		cudaMalloc(&d_src_points_values_offset,		num_sources * sizeof(size_t));
		cudaMalloc(&d_rec_points_interval,			rec_points_interval_size * sizeof(size_t));
		cudaMalloc(&d_rec_points_values,			rec_points_values_size * sizeof(f_type));
		cudaMalloc(&d_rec_points_values_offset,		num_receivers * sizeof(size_t));
		cudaMalloc(&d_wavelet_forward,				wavelet_forward_size * wavelet_forward_count * sizeof(f_type));
		cudaMalloc(&d_wavelet_adjoint,				wavelet_adjoint_size * wavelet_adjoint_count * sizeof(f_type));

		// copy data to device
		cudaMemcpy(d_v + stencil_radius*nx*ny*sizeof(f_type),			v,							individual_domain_size * sizeof(f_type),						cudaMemcpyHostToDevice);
		cudaMemcpy(d_grad + stencil_radius*nx*ny*sizeof(f_type),		grad,						individual_domain_size * sizeof(f_type),						cudaMemcpyHostToDevice);
		cudaMemcpy(d_velocity + stencil_radius*nx*ny*sizeof(f_type),	velocity,					individual_domain_size * sizeof(f_type),						cudaMemcpyHostToDevice);
		cudaMemcpy(d_damp + stencil_radius*nx*ny*sizeof(f_type),		damp,						individual_domain_size * sizeof(f_type),						cudaMemcpyHostToDevice);
		cudaMemcpy(d_coeff,												coeff,						(stencil_radius+1) * sizeof(f_type),							cudaMemcpyHostToDevice);
		cudaMemcpy(d_src_points_interval,								src_points_interval,		src_points_interval_size * sizeof(size_t),						cudaMemcpyHostToDevice);
		cudaMemcpy(d_src_points_values,									src_points_values,			src_points_values_size * sizeof(f_type),						cudaMemcpyHostToDevice);
		cudaMemcpy(d_src_points_values_offset,							src_points_values_offset,	num_sources * sizeof(size_t),									cudaMemcpyHostToDevice);
		cudaMemcpy(d_rec_points_interval,								rec_points_interval,		rec_points_interval_size * sizeof(size_t),						cudaMemcpyHostToDevice);
		cudaMemcpy(d_rec_points_values,									rec_points_values,			rec_points_values_size * sizeof(f_type),						cudaMemcpyHostToDevice);
		cudaMemcpy(d_rec_points_values_offset,							rec_points_values_offset,	num_receivers * sizeof(size_t),									cudaMemcpyHostToDevice);
		cudaMemcpy(d_wavelet_forward,									wavelet_forward,			wavelet_forward_size * wavelet_forward_count * sizeof(f_type),	cudaMemcpyHostToDevice);
		cudaMemcpy(d_wavelet_adjoint,									wavelet_adjoint,			wavelet_adjoint_size * wavelet_adjoint_count * sizeof(f_type),	cudaMemcpyHostToDevice);
        // allocates memory for u on the device
        // used by forward

		cudaMalloc(&us[device_id], u_size * sizeof(f_type));

        f_type *u = us[device_id];
        /** run forward to calculate all checkpoints **/

        printf("\nRunning Forward to save %ld checkpoints\n", num_snapshots);

        #pragma omp barrier
        stack<CheckpointStruct> snapshots = forward_saving(d_velocity, d_damp,
                                                            d_wavelet_forward, wavelet_forward_size, wavelet_forward_count,
                                                            d_coeff, 
                                                            d_src_points_interval, src_points_interval_size,
                                                            d_src_points_values, src_points_values_size,
                                                            d_src_points_values_offset,              
                                                            num_sources, num_receivers,
                                                            nz, nx, ny, dz, dx, dy, saving_stride, dt,
                                                            begin_timestep, end_timestep,
                                                            space_order, num_snapshots, device_id, us[device_id-1], us[device_id+1], u,
                                                            snapshot_d_prev_compressed, snapshot_d_cur_compressed, 
                                                            nvcomp_manager);    
        

        printf("\nRunning Gradient\n");

        // get the start time
        gettimeofday(&time_start, NULL);

        #pragma omp barrier
        // allocate memory for the snapshot on the device    
        f_type *snapshot_d_prev;
        f_type *snapshot_d_current;

        cudaMalloc(&snapshot_d_prev_compressed, compression_configure_malloc.max_compressed_buffer_size);
        cudaMalloc(&snapshot_d_cur_compressed, compression_configure_malloc.max_compressed_buffer_size);

        // copy current checkpoint
        DecompressionConfig decompression_configure_prev = nvcomp_manager.configure_decompression(snapshot_d_prev_compressed);
        DecompressionConfig decompression_configure_cur = nvcomp_manager.configure_decompression(snapshot_d_cur_compressed);
        //TODO 4: Descomprimir os dados e colocar os dados reais nos snapshot_d_prev e snapshot_d_current

        //#pragma omp target data use_device_ptr(snapshot_d_prev, snapshot_d_current)
        {
            CUDA_CHECK( cudaMemcpy(snapshot_d_prev_compressed, snapshots.top().prev, snapshots.top().size_compressed_prev, cudaMemcpyHostToDevice) );  
            CUDA_CHECK( cudaMemcpy(snapshot_d_cur_compressed, snapshots.top().current, snapshots.top().size_compressed_current, cudaMemcpyHostToDevice) );

            nvcomp_manager.decompress( (uint8_t *) snapshot_d_prev,  (const uint8_t *) snapshot_d_prev_compressed, decompression_configure_prev);

            final_status = *decompression_configure_prev.get_status();
            if(final_status != nvcompSuccess) {
                throw std::runtime_error("Error na Descompressao 1.\n");
            }

            nvcomp_manager.decompress( (uint8_t *) snapshot_d_current,  (const uint8_t *) snapshot_d_cur_compressed, decompression_configure_cur);

            final_status = *decompression_configure_cur.get_status();
            if(final_status != nvcompSuccess) {
                throw std::runtime_error("Error na Descompressao 1.\n");
            }
        }


        printf("Copying checkpoint %d in timestep %ld\n", snapshots.top().index, end_timestep);

        // wavefield modeling
        for(size_t n = end_timestep; n >= begin_timestep; n--) {

            // no saving case
            // since we are moving backwards, we invert the next_t and prev_t pointer
            next_t =    (n - 1) % 3;
            current_t = n % 3;
            prev_t =    (n + 1) % 3;

            /*
                Section 1: update the wavefield according to the acoustic wave equation
            */
            int i_ini = stencil_radius;
            int i_fim = individual_nz+stencil_radius;

            if (device_id == 0)
                i_ini = stencil_radius + stencil_radius;

            if(device_id == NUM_DEVICES-1)
                i_fim = individual_nz;


            #pragma omp barrier
            #pragma omp target teams distribute parallel for collapse(3) device(device_id) is_device_ptr(u, d_v, d_grad, d_velocity, d_damp, d_coeff, d_src_points_interval, d_src_points_values, d_src_points_values_offset, d_rec_points_interval, d_rec_points_values, d_rec_points_values_offset, d_wavelet_forward, d_wavelet_adjoint)
            for(size_t i = i_ini; i < i_fim; i++) {
                for(size_t j = stencil_radius; j < nx - stencil_radius; j++) {
                    for(size_t k = stencil_radius; k < ny - stencil_radius; k++) {
                        // index of the current point in the grid
                        size_t domain_offset = (i * nx + j) * ny + k;

                        size_t prev_v =         prev_t      * individual_domain_size + domain_offset;
                        size_t current_v =      current_t   * individual_domain_size + domain_offset;
                        size_t next_v =         next_t      * individual_domain_size + domain_offset;

                        // stencil code to update grid
                        f_type value = 0.0;

                        f_type sum_y = d_coeff[0] * d_v[current_v];
                        f_type sum_x = d_coeff[0] * d_v[current_v];
                        f_type sum_z = d_coeff[0] * d_v[current_v];

                        // radius of the stencil                    
                        for(size_t ir = 1; ir <= stencil_radius; ir++){
                            //neighbors in the Y direction
                            sum_y += d_coeff[ir] * (d_v[current_v + ir] + d_v[current_v - ir]);

                            //neighbors in the X direction
                            sum_x += d_coeff[ir] * (d_v[current_v + (ir * ny)] + d_v[current_v - (ir * ny)]);

                            //neighbors in the Z direction
                            sum_z += d_coeff[ir] * (d_v[current_v + (ir * nx * ny)] + d_v[current_v - (ir * nx * ny)]);
                        }

                        value += sum_y/dySquared + sum_x/dxSquared + sum_z/dzSquared;

                        // parameter to be used
                        f_type slowness = 1.0 / (d_velocity[domain_offset] * d_velocity[domain_offset]);

                        // denominator with damp coefficient
                        f_type denominator = (1.0 + d_damp[domain_offset] * dt / 2);
                        f_type numerator = (1.0 - d_damp[domain_offset] * dt / 2);

                        value *= (dtSquared / slowness) / denominator;

                        d_v[next_v] = 2.0 / denominator * d_v[current_v] - (numerator / denominator) * d_v[prev_v] + value;
                    }
                }
            }

            #pragma omp barrier
			if (device_id > 0) {
				cudaMemcpyPeer(us[device_id-1] + (individual_nz + stencil_radius) * nx * ny * sizeof(f_type), //Dst
						device_id-1,																//Dst device
						u + stencil_radius * nx * ny * sizeof(f_type),							  	//Src
						device_id,																  	//Src device
						stencil_radius * nx * ny * sizeof(f_type));								 	//Size
			}

			if (device_id < NUM_DEVICES - 1) {
				cudaMemcpyPeer(us[device_id+1], 							//Dst
						device_id+1,									//Dst device
						u + individual_nz * nx * ny * sizeof(f_type), 	//Src
						device_id,									 	//Src device
						stencil_radius * nx * ny * sizeof(f_type));	//Size
			}
			#pragma omp barrier

            /*
                Section 2: add the source term

                The receivers are the sources the in the adjoint
                */

            #pragma omp target teams distribute parallel for device(device_id) is_device_ptr(u, d_v, d_grad, d_velocity, d_damp, d_coeff, d_src_points_interval, d_src_points_values, d_src_points_values_offset, d_rec_points_interval, d_rec_points_values, d_rec_points_values_offset, d_wavelet_forward, d_wavelet_adjoint)
            for(size_t rec = 0; rec < num_receivers; rec++){

                size_t wavelet_offset = n - 1;

                if(wavelet_adjoint_count > 1){
                    wavelet_offset = (n-1) * num_receivers + rec;
                }

                if(d_wavelet_adjoint[wavelet_offset] != 0.0){
                    // each receiver has 6 (z_b, z_e, x_b, x_e, y_b, y_e) point intervals
                    size_t offset_rec = rec * 6;

                    // interval of grid points of the receiver in the Z axis
                    size_t rec_z_begin = d_rec_points_interval[offset_rec + 0];
                    size_t rec_z_end = d_rec_points_interval[offset_rec + 1];
                    
                    size_t rec_x_begin = d_rec_points_interval[offset_rec + 2];
                    size_t rec_x_end = d_rec_points_interval[offset_rec + 3];

                    // interval of grid points of the receiver in the Y axis
                    size_t rec_y_begin = d_rec_points_interval[offset_rec + 4];
                    size_t rec_y_end = d_rec_points_interval[offset_rec + 5];


                    // number of grid points of the receiver in each axis
                    size_t rec_z_num_points = rec_z_end - rec_z_begin + 1;
                    size_t rec_x_num_points = rec_x_end - rec_x_begin + 1;
                    //size_t rec_y_num_points = rec_y_end - rec_y_begin + 1;

                    // pointer to rec value offset
                    size_t offset_rec_kws_index_z = d_rec_points_values_offset[rec];

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

                                f_type kws = d_rec_points_values[kws_index_z] * d_rec_points_values[kws_index_x] * d_rec_points_values[kws_index_y];

                                // current source point in the grid
                                size_t domain_offset = (i * nx + j) * ny + k;
                                size_t next_v = next_t * individual_domain_size + (domain_offset % individual_domain_size);

                                // parameter to be used
                                f_type slowness = 1.0 / (d_velocity[domain_offset % individual_domain_size] * d_velocity[domain_offset % individual_domain_size]);

                                // denominator with damp coefficient
                                f_type denominator = (1.0 + d_damp[domain_offset % individual_domain_size] * dt / 2);
                                d_v[next_v] = dtSquared / slowness * kws * d_wavelet_adjoint[wavelet_offset] / denominator + d_v[next_v];

                                kws_index_y++;
                            }
                            kws_index_x++;
                        }
                        kws_index_z++;
                    }
                }
            }

            #pragma omp barrier
            /*
                Section 3: get current snapshot
            */
            f_type *u_snapshot;            

            if( (n-1) % saving_stride == 0 ){
                
                // here we use the checkpoint         
               
                //printf("Recovering checkpoint %d in timestep %ld\n", snapshots.top().index, n);

                u_snapshot = snapshot_d_current;                        

            }else{                       
                // recompute
                // call forward_checkpoint with device pointers:
                u_snapshot = forward_checkpoint(u, snapshot_d_prev, snapshot_d_current, d_velocity, d_damp,
                              d_wavelet_forward, wavelet_forward_size, wavelet_forward_count,               
                              d_coeff, 
                              d_src_points_interval, src_points_interval_size,
                              d_src_points_values, src_points_values_size,
                              d_src_points_values_offset,
                              num_sources,
                              nz, nx, ny, dz, dx, dy,
                              dt,
                              (snapshots.top().index * saving_stride + 1) , n,
                              space_order, device_id, us[device_id-1], us[device_id+1]);

            }
            /*
                Section 4: gradient calculation
            */        
            #pragma omp target teams distribute parallel for collapse(3) device(device_id) is_device_ptr(u, d_v, d_grad, d_velocity, d_damp, d_coeff, d_src_points_interval, d_src_points_values, d_src_points_values_offset, d_rec_points_interval, d_rec_points_values, d_rec_points_values_offset, d_wavelet_forward, d_wavelet_adjoint, u_snapshot)
            for(size_t i = i_ini; i < i_fim; i++) {
                for(size_t j = stencil_radius; j < nx - stencil_radius; j++) {
                    for(size_t k = stencil_radius; k < ny - stencil_radius; k++) {
                        // index of the current point in the grid
                        size_t domain_offset = (i * nx + j) * ny + k;

                        size_t prev_v = prev_t * individual_domain_size + domain_offset;
                        size_t current_v = current_t * individual_domain_size + domain_offset;
                        size_t next_v = next_t * individual_domain_size + domain_offset;

                        f_type v_second_time_derivative = (d_v[prev_v] - 2.0 * d_v[current_v] + d_v[next_v]) / dtSquared;

                        // update gradient
                        d_grad[domain_offset] -= v_second_time_derivative * u_snapshot[domain_offset];
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
                    DecompressionConfig decompression_configure_prev = nvcomp_manager.configure_decompression(snapshot_d_prev_compressed);
                    DecompressionConfig decompression_configure_cur = nvcomp_manager.configure_decompression(snapshot_d_cur_compressed);

                    //#pragma omp target data use_device_ptr(snapshot_d_prev, snapshot_d_current)
                    {
                        CUDA_CHECK( cudaMemcpy(snapshot_d_prev_compressed, snapshots.top().prev, snapshots.top().size_compressed_prev, cudaMemcpyHostToDevice) );  
                        CUDA_CHECK( cudaMemcpy(snapshot_d_cur_compressed, snapshots.top().current, snapshots.top().size_compressed_current, cudaMemcpyHostToDevice) );
                        nvcomp_manager.decompress( (uint8_t *) snapshot_d_prev,  (const uint8_t *) snapshot_d_prev_compressed, decompression_configure_prev);
                        final_status = *decompression_configure_prev.get_status();
                        if(final_status != nvcompSuccess) {
                            throw std::runtime_error("Error na Descompressao 1.\n");
                            exit(1);
                        }

                        nvcomp_manager.decompress( (uint8_t *) snapshot_d_current,  (const uint8_t *) snapshot_d_cur_compressed, decompression_configure_cur);
                        final_status = *decompression_configure_cur.get_status();
                        if(final_status != nvcompSuccess) {
                            throw std::runtime_error("Error na Descompressao 1.\n");
                            exit(1);
                        }
                    }


                    printf("Extract chckpt %zu in timestep %zu size(comp)=%ld  size(decomp)=%zu  comp.ratio=%f \n", snapshots.top().index, n, snapshots.top().size_compressed_current, domain_size * sizeof(f_type), domain_size /(float) snapshots.top().size_compressed_current );


            }            
                }

        }
		cudaMemcpy(grad, d_grad + stencil_radius*nx*ny*sizeof(f_type), individual_domain_size * sizeof(f_type), cudaMemcpyDeviceToHost);
		cudaFree(d_v);
		cudaFree(d_grad);
		cudaFree(d_velocity);
		cudaFree(d_damp);
        CUDA_CHECK( cudaStreamDestroy(stream) ); 
        CUDA_CHECK( cudaFree(snapshot_d_prev_compressed) );
        CUDA_CHECK( cudaFree(snapshot_d_cur_compressed) );    
    } 
    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) + (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    return exec_time;
}

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "png.h"
#include <vector>
#include <assert.h>
#include <iostream>
#include <memory>
#include "mpi.h"
#include "utils/image.h"
#include "utils/dct.h"
#include <string>
#include <chrono>
#include <cstdio>  // Para remove()

Image<float> get_srm_3x3() {
    Image<float> kernel(3,3,1);
    kernel.set(0,0,0,-1); kernel.set(0,1,0,2); kernel.set(0,2,0,-1);
    kernel.set(1,0,0,2);  kernel.set(1,1,0,-4); kernel.set(1,2,0,2);
    kernel.set(2,0,0,-1); kernel.set(2,1,0,2); kernel.set(2,2,0,-1);
    return kernel;
}

Image<float> get_srm_5x5() {
    Image<float> kernel(5,5,1);
    int vals[5][5] = {
        {-1,2,-2,2,-1},
        { 2,-6,8,-6,2},
        {-2,8,-12,8,-2},
        { 2,-6,8,-6,2},
        {-1,2,-2,2,-1}
    };
    for(int i=0;i<5;i++)
        for(int j=0;j<5;j++)
            kernel.set(i,j,0,vals[i][j]);
    return kernel;
}

Image<float> get_srm_kernel(int size) {
    return size == 3 ? get_srm_3x3() : get_srm_5x5();
}


Image<float> simple_convolution(const Image<float>& img, const Image<float>& kernel) {
    int kernel_size = kernel.width;
    int halo = kernel_size / 2;
    int W = img.width;
    int H = img.height;
    int C = img.channels;
    
    Image<float> result(W, H, C);
    
    for (int j = 0; j < H; j++) {
        for (int i = 0; i < W; i++) {
            for (int c = 0; c < C; c++) {
                float sum = 0.0;
                for (int u = 0; u < kernel_size; u++) {
                    for (int v = 0; v < kernel_size; v++) {
                        int s = j + u - halo;
                        int t = i + v - halo;
                        
                        if (s >= 0 && s < H && t >= 0 && t < W) {
                            sum += img.get(s, t, c) * kernel.get(u, v, 0);
                        }
                    }
                }
                result.set(j, i, c, sum);
            }
        }
    }
    
    return result;
}

Image<float> get_complete_grayscale(const Image<unsigned char>& image) {
    int W = image.width;
    int H = image.height;
    
    if (image.channels == 1) {
        // Ya es escala de grises
        Image<float> gray_float(W, H, 1);
        
        for (int j = 0; j < H; j++) {
            for (int i = 0; i < W; i++) {
                gray_float.set(j, i, 0, (float)image.get(j, i, 0));
            }
        }
        
        return gray_float;
    } else {
        // Convertir RGB > escala de grises
        Image<float> gray(W, H, 1);
        
        for (int j = 0; j < H; j++) {
            for (int i = 0; i < W; i++) {
                float value = 0.299f * image.get(j, i, 0) + 
                              0.587f * image.get(j, i, 1) + 
                              0.114f * image.get(j, i, 2);
                gray.set(j, i, 0, value);
            }
        }
        
        return gray;
    }
}


Image<unsigned char> compute_srm_parallel(const Image<unsigned char>& image, int kernel_size, int rank, int procs) {
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Computing SRM " << kernel_size << "x" << kernel_size << "..." << std::endl;
    }
    
    if (image.empty()) {
        std::cerr << "Rank " << rank << ": ERROR - Input image is empty!" << std::endl;
        return Image<unsigned char>();
    }
    
    
    Image<float> grayscale = get_complete_grayscale(image);
    
    if (grayscale.empty()) {
        std::cerr << "Rank " << rank << ": ERROR - Grayscale conversion failed!" << std::endl;
        return Image<unsigned char>();
    }
    
    int H = grayscale.height;
    int W = grayscale.width;
    
    
    int rows_per_proc = H / procs;
    int extra = H % procs;
    
    int start_row, local_rows;
    
    if (rank < extra) {
        start_row = rank * (rows_per_proc + 1);
        local_rows = rows_per_proc + 1;
    } else {
        start_row = rank * rows_per_proc + extra;
        local_rows = rows_per_proc;
    }
    
    
    if (start_row >= H) {
        local_rows = 0;
    } else if (start_row + local_rows > H) {
        local_rows = H - start_row;
    }
    
    Image<unsigned char> final_result;
    
    if (local_rows > 0) {
        
        Image<float> local_part(W, local_rows, 1);
        
        for (int j = 0; j < local_rows; j++) {
            int global_row = start_row + j;
            for (int i = 0; i < W; i++) {
                local_part.set(j, i, 0, grayscale.get(global_row, i, 0));
            }
        }
        
        
        Image<float> kernel = get_srm_kernel(kernel_size);
        Image<float> convolved = simple_convolution(local_part, kernel);
        
        
        Image<float> abs_result(W, local_rows, 1);
        for (int j = 0; j < local_rows; j++) {
            for (int i = 0; i < W; i++) {
                float val = convolved.get(j, i, 0);
                abs_result.set(j, i, 0, val < 0 ? -val : val);
            }
        }
        
        float local_min = abs_result.get(0, 0, 0);
        float local_max = abs_result.get(0, 0, 0);
        
        for (int j = 0; j < local_rows; j++) {
            for (int i = 0; i < W; i++) {
                float val = abs_result.get(j, i, 0);
                if (val < local_min) local_min = val;
                if (val > local_max) local_max = val;
            }
        }
        
        
        float global_min, global_max;
        MPI_Allreduce(&local_min, &global_min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        
        
        float range = global_max - global_min;
        if (range == 0.0f) range = 1.0f;
        
        Image<unsigned char> local_result(W, local_rows, 1);
        
        for (int j = 0; j < local_rows; j++) {
            for (int i = 0; i < W; i++) {
                float val = abs_result.get(j, i, 0);
                float normalized = (val - global_min) / range;
                unsigned char byte_val = (unsigned char)(normalized * 255.0f);
                local_result.set(j, i, 0, byte_val);
            }
        }
        
      
        if (rank == 0) {
            // Crear imagen final
            final_result = Image<unsigned char>(W, H, 1);
            
            // Copiar parte del proc. 0
            for (int j = 0; j < local_rows; j++) {
                for (int i = 0; i < W; i++) {
                    final_result.set(start_row + j, i, 0, local_result.get(j, i, 0));
                }
            }
            
            // Recibir de otros procesos
            for (int p = 1; p < procs; p++) {
                
                int p_start_row, p_local_rows;
                if (p < extra) {
                    p_start_row = p * (rows_per_proc + 1);
                    p_local_rows = rows_per_proc + 1;
                } else {
                    p_start_row = p * rows_per_proc + extra;
                    p_local_rows = rows_per_proc;
                }
                
                if (p_start_row >= H) p_local_rows = 0;
                else if (p_start_row + p_local_rows > H) p_local_rows = H - p_start_row;
                
                if (p_local_rows > 0) {
                    std::vector<unsigned char> buffer(p_local_rows * W);
                    MPI_Recv(buffer.data(), p_local_rows * W, MPI_UNSIGNED_CHAR,
                            p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    // Copiar al res final
                    for (int j = 0; j < p_local_rows; j++) {
                        for (int i = 0; i < W; i++) {
                            final_result.set(p_start_row + j, i, 0, buffer[j * W + i]);
                        }
                    }
                }
            }
        } else {
            // Otros procesos envían su resultado
            std::vector<unsigned char> buffer(local_rows * W);
            
            for (int j = 0; j < local_rows; j++) {
                for (int i = 0; i < W; i++) {
                    buffer[j * W + i] = local_result.get(j, i, 0);
                }
            }
            
            MPI_Send(buffer.data(), local_rows * W, MPI_UNSIGNED_CHAR,
                    0, 0, MPI_COMM_WORLD);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    return final_result;
}


Image<unsigned char> compute_dct_parallel(const Image<unsigned char>& image,
                                          int block_size, bool invert,
                                          int rank, int procs) {
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Computing DCT (block size: " << block_size << ")..." << std::endl;
    }
    
    // Solo el proceso 0 ejecuta DCT (es más simple)
    if (rank != 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        return Image<unsigned char>();
    }
    
    // Proceso 0 hace todo el trabajo DCT
    Image<float> grayscale = get_complete_grayscale(image);
    
    if (grayscale.empty()) {
        std::cerr << "ERROR: Grayscale conversion failed for DCT!" << std::endl;
        return Image<unsigned char>();
    }
    
    int H = grayscale.height;
    int W = grayscale.width;
    
    
    if (W % block_size != 0 || H % block_size != 0) {
        std::cerr << "ERROR: Image dimensions must be multiples of block size!" << std::endl;
        return Image<unsigned char>();
    }
    
    
    std::vector<Block<float>> blocks = grayscale.get_blocks(block_size);
    int total_blocks = blocks.size();
    
    std::cout << "Processing " << total_blocks << " blocks..." << std::endl;
    
    
    for(auto& block : blocks) {
        float** dctBlock = dct::create_matrix(block_size, block_size);
        dct::direct(dctBlock, block, 0);

        if (invert) {
           
            for(int k = 0; k < block_size/2; k++) {
                for(int l = 0; l < block_size/2; l++) {
                    dctBlock[k][l] = 0.0f;
                }
            }
            dct::inverse(block, dctBlock, 0, 0.0, 255.);
        } else {
            dct::assign(dctBlock, block, 0);
        }
        dct::delete_matrix(dctBlock);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    // Convertir resultado a unsigned char MANUAL PORQUE EL MPI SE QUEJAA
    Image<unsigned char> result = Image<unsigned char>(W, H, 1);
    for (int j = 0; j < H; j++) {
        for (int i = 0; i < W; i++) {
            float val = grayscale.get(j, i, 0);
            if (val < 0.0f) val = 0.0f;
            if (val > 255.0f) val = 255.0f;
            result.set(j, i, 0, (unsigned char)val);
        }
    }
    
    return result;
}


Image<unsigned char> compute_ela(const Image<unsigned char>& image, int quality, int rank) {
    // Solo el proceso 0 ejecuta ELA (usa archivos temporales)
    if (rank != 0) {
        return Image<unsigned char>();
    }
    
    std::cout << "Computing ELA (quality: " << quality << ")..." << std::endl;
    
    // 1. Convertir a escala de grises si es necesario
    Image<unsigned char> grayscale;
    if (image.channels == 1) {
        grayscale = image;
    } else {
        grayscale = Image<unsigned char>(image.width, image.height, 1);
        for (int j = 0; j < image.height; j++) {
            for (int i = 0; i < image.width; i++) {
                float value = 0.299f * image.get(j, i, 0) + 
                              0.587f * image.get(j, i, 1) + 
                              0.114f * image.get(j, i, 2);
                grayscale.set(j, i, 0, (unsigned char)value);
            }
        }
    }
    
    // 2. Guardar con compresión JPEG
    std::string temp_filename = "_temp_ela.jpg";
    save_to_file(temp_filename, grayscale, quality);
    
    // 3. Cargar la imagen comprimida
    Image<unsigned char> compressed = load_from_file(temp_filename);
    
    if (compressed.empty()) {
        std::cerr << "ERROR: Failed to load compressed image!" << std::endl;
        std::remove(temp_filename.c_str());
        return Image<unsigned char>();
    }
    
    // 4. Calcular diferencia (Error Level)
    Image<unsigned char> ela_result(image.width, image.height, 1);
    
    // Asegurar que las dimensiones coincidan
    int min_height = std::min(grayscale.height, compressed.height);
    int min_width = std::min(grayscale.width, compressed.width);
    
    float max_diff = 0.0f;
    
    for (int j = 0; j < min_height; j++) {
        for (int i = 0; i < min_width; i++) {
            int diff = std::abs((int)grayscale.get(j, i, 0) - (int)compressed.get(j, i, 0));
            if (diff > max_diff) max_diff = diff;
            ela_result.set(j, i, 0, (unsigned char)diff);
        }
    }
    
    // 5. Normalizar para mejor visualización
    if (max_diff > 0) {
        for (int j = 0; j < min_height; j++) {
            for (int i = 0; i < min_width; i++) {
                float diff = (float)ela_result.get(j, i, 0);
                unsigned char normalized = (unsigned char)((diff / max_diff) * 255.0f);
                ela_result.set(j, i, 0, normalized);
            }
        }
    }
    
    // 6. Limpiar archivo temporal
    std::remove(temp_filename.c_str());
    
    return ela_result;
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    if(argc < 2) {
        if(rank == 0)
            std::cerr << "Image filename missing. El comando es: ./main <file>" << std::endl;
        MPI_Finalize();
        return 1;
    }


    Image<unsigned char> image;
    int dims[3] = {0, 0, 0};
    
    if(rank == 0) {
        std::cout << "========================================" << std::endl;
        std::cout << "MPI Parallel Analisys by Group 4 of IC" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Loading image: " << argv[1] << std::endl;
        
        image = load_from_file(argv[1]);
        if (image.empty()) {
            std::cerr << "ERROR: Failed to load image!" << std::endl;
            dims[0] = dims[1] = dims[2] = -1;
        } else {
            dims[0] = image.width;
            dims[1] = image.height;
            dims[2] = image.channels;
            std::cout << "Image loaded: " << dims[0] << "x" << dims[1] 
                      << ", channels: " << dims[2] << std::endl;
            std::cout << "Processes: " << procs << std::endl;
        }
    }
    
    // Broadcast dim
    MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (dims[0] <= 0 || dims[1] <= 0) {
        if (rank == 0) std::cerr << "Invalid image dimensions! (mira image.h)" << std::endl;
        MPI_Finalize();
        return 1;
    }
    
    // Otros procesos crean imag
    if (rank != 0) {
        image = Image<unsigned char>(dims[0], dims[1], dims[2]);
    }
    
    // Broadcast imag
    int total_pixels = dims[0] * dims[1] * dims[2];
    MPI_Bcast(image.data(), total_pixels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
   
    
    // ELA (solo proceso 0
    if (rank == 0) {
        std::cout << "1. Error Level Analysis (ELA)" << std::endl;
        std::cout << "-------------------------------" << std::endl;
    }
    
    auto ela_start = std::chrono::high_resolution_clock::now();
    Image<unsigned char> ela_result = compute_ela(image, 90, rank);
    auto ela_end = std::chrono::high_resolution_clock::now();
    
    if (rank == 0 && !ela_result.empty()) {
        save_to_file("ela.png", ela_result);
        auto ela_duration = std::chrono::duration_cast<std::chrono::milliseconds>(ela_end - ela_start);
        std::cout << "ELA completed in " << ela_duration.count() << " ms" << std::endl;
        std::cout << "Saved: ela.png" << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // SRM 3x3 (paral)
    if (rank == 0) {
        std::cout << "2. SRM Filter 3x3" << std::endl;
        std::cout << "-------------------" << std::endl;
    }
    
    auto srm3_start = std::chrono::high_resolution_clock::now();
    Image<unsigned char> srm3_result = compute_srm_parallel(image, 3, rank, procs);
    auto srm3_end = std::chrono::high_resolution_clock::now();
    
    if (rank == 0 && !srm3_result.empty()) {
        save_to_file("srm_kernel_3x3.png", srm3_result);
        auto srm3_duration = std::chrono::duration_cast<std::chrono::milliseconds>(srm3_end - srm3_start);
        std::cout << "SRM 3x3 completed in " << srm3_duration.count() << " ms" << std::endl;
        std::cout << "Saved: srm_kernel_3x3.png" << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // SRM 5x5 (paral)
    if (rank == 0) {
        std::cout << "3. SRM Filter 5x5" << std::endl;
        std::cout << "-------------------" << std::endl;
    }
    
    auto srm5_start = std::chrono::high_resolution_clock::now();
    Image<unsigned char> srm5_result = compute_srm_parallel(image, 5, rank, procs);
    auto srm5_end = std::chrono::high_resolution_clock::now();
    
    if (rank == 0 && !srm5_result.empty()) {
        save_to_file("srm_kernel_5x5.png", srm5_result);
        auto srm5_duration = std::chrono::duration_cast<std::chrono::milliseconds>(srm5_end - srm5_start);
        std::cout << "SRM 5x5 completed in " << srm5_duration.count() << " ms" << std::endl;
        std::cout << "Saved: srm_kernel_5x5.png" << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // DCT (solo proc0)
    if (rank == 0) {
        std::cout << "4. DCT Analysis" << std::endl;
        std::cout << "----------------" << std::endl;
    }
    
    auto dct_start = std::chrono::high_resolution_clock::now();
    Image<unsigned char> dct_result;
    
    try {
        dct_result = compute_dct_parallel(image, 8, true, rank, procs);
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "   DCT FALLO: " << e.what() << std::endl;
        }
    }
    
    auto dct_end = std::chrono::high_resolution_clock::now();
    
    if (rank == 0 && !dct_result.empty()) {
        save_to_file("dct_invert.png", dct_result);
        auto dct_duration = std::chrono::duration_cast<std::chrono::milliseconds>(dct_end - dct_start);
        std::cout << "DCT completed in " << dct_duration.count() << " ms" << std::endl;
        std::cout << "Saved: dct_invert.png" << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    
    if (rank == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Processing Complete!" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Generated files:" << std::endl;
        std::cout << "  • ela.png - Error Level Analysis" << std::endl;
        std::cout << "  • srm_kernel_3x3.png - SRM 3x3 filter" << std::endl;
        std::cout << "  • srm_kernel_5x5.png - SRM 5x5 filter" << std::endl;
        std::cout << "  • dct_invert.png - DCT analysis" << std::endl;
        std::cout << std::endl;
        std::cout << "Note: All processes worked together for SRM filters." << std::endl;
        std::cout << "========================================" << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
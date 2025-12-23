#ifndef __IMAGE__H__
#define __IMAGE__H__
#include "mpi.h"
#include <vector>
#include <memory>
#include <iostream>
#include <cassert>
#include <string>
#include <limits>
#include <cmath>
#include <algorithm>

template <typename T>
class Block;

template <typename T>
class Image
{
public:
    int width, height, channels;
    std::shared_ptr<T[]> matrix;
    
    void release();
    Image();
    Image(int width, int height, int channels);
    Image(const Image<T> &a);
    ~Image();
    Image<T>& operator=(const Image<T> &other);
    Image<T> operator*(const Image<T> &other) const;
    Image<T> operator*(float scalar) const;
    Image<T> operator+(const Image<T> &other) const;
    Image<T> operator+(float scalar) const;
    T get(int row, int col, int channel) const;
    void set(int row, int col, int channel, T value);
    template <typename S>
    Image<S> convert() const;
    Image<T> to_grayscale() const;
    Image<T> abs() const;
    Image<float> normalized() const;
    Image<T> convolution(const Image<float> &kernel) const;
    std::vector<Block<T>> get_blocks(int block_size = 8);
    T* get_matrix() { return matrix.get(); }
    const T* get_matrix() const { return matrix.get(); }
    T* data() { return matrix.get(); }
    const T* data() const { return matrix.get(); }
    bool empty() const { return !matrix || width <= 0 || height <= 0 || channels <= 0; }
    size_t size() const { return width * height * channels; }
};

Image<unsigned char> load_from_file(const std::string &filename);
void save_to_file(const std::string &filename, const Image<unsigned char> &image, int quality = 100);

template <typename T>
class Block
{
public:
    int i, j, size, depth, rowsize;
    Image<T> *matrix;
    T get_pixel(int row, int col, int channel) const;
    void set_pixel(int row, int col, int channel, T value);
};

template <class T> T Block<T>::get_pixel(int row, int col, int channel) const
{
    assert(row >= 0 && row < size && col >= 0 && col < size);
    return matrix->get(row + j, col + i, channel);
}

template <class T> void Block<T>::set_pixel(int row, int col, int channel, T value)
{
    assert(row >= 0 && row < size && col >= 0 && col < size);
    matrix->set(row + j, col + i, channel, value);
}

// img vacia sin memoria
template <class T> Image<T>::Image()
{
    width = 0;
    height = 0;
    channels = 0;
    matrix = nullptr;
}

// reserva memoria para img nueva
template <class T> Image<T>::Image(int width, int height, int channels)
{
    this->width = width;
    this->height = height;
    this->channels = channels;
    
    if (width > 0 && height > 0 && channels > 0) {
        matrix = std::shared_ptr<T[]>(new T[height * width * channels]);
    } else {
        matrix = nullptr;
    }
}

// copia comparte los datos SIN DUPLICAR
template <class T> Image<T>::Image(const Image<T> &a)
{
    width = a.width;
    height = a.height;
    channels = a.channels;
    matrix = a.matrix;
}

// destructor img
template <class T> Image<T>::~Image()
{
    release();
}

// asigna compartiendo datos
template <class T> Image<T>& Image<T>::operator=(const Image<T> &a)
{
    if (this == &a){
        return *this;
    }
    
    release();
    
    width = a.width;
    height = a.height;
    channels = a.channels;
    matrix = a.matrix;
    
    return *this;
}

// libera memo poniendola a null
template <class T> void Image<T>::release()
{
    matrix = nullptr;
}

template <class T> T Image<T>::get(int row, int col, int channel) const
{
    assert(matrix != nullptr);
    assert(row >= 0 && row < height);
    assert(col >= 0 && col < width);
    assert(channel >= 0 && channel < channels);
    
    return matrix[row * width * channels + col * channels + channel];
}

template <class T> void Image<T>::set(int row, int col, int channel, T value)
{
    assert(matrix != nullptr);
    assert(row >= 0 && row < height);
    assert(col >= 0 && col < width);
    assert(channel >= 0 && channel < channels);
    
    matrix[row * width * channels + col * channels + channel] = value;
}

// Helper function para determinar tipo MPI
template<typename T>
MPI_Datatype get_mpi_type() {
    if (std::is_same<T, float>::value)
        return MPI_FLOAT;
    else if (std::is_same<T, double>::value)
        return MPI_DOUBLE;
    else if (std::is_same<T, int>::value)
        return MPI_INT;
    else if (std::is_same<T, unsigned char>::value)
        return MPI_UNSIGNED_CHAR;
    else
        return MPI_BYTE;
}

// Helper function para calcular distribución de filas
inline void calculate_row_distribution(int height, int size, int rank, 
                                      int &start_row, int &local_rows) {
    int rows_per_proc = height / size;
    int remainder = height % size;
    
    start_row = rank * rows_per_proc + std::min(rank, remainder);
    local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
}

// Helper function para calcular recvcounts y displs
inline void calculate_gather_params(int height, int width, int channels, int size,
                                   std::vector<int> &recvcounts, std::vector<int> &displs) {
    int rows_per_proc = height / size;
    int remainder = height % size;
    int elements_per_row = width * channels;
    
    recvcounts.resize(size);
    displs.resize(size);
    
    int offset = 0;
    for (int i = 0; i < size; i++) {
        int proc_rows = rows_per_proc + (i < remainder ? 1 : 0);
        recvcounts[i] = proc_rows * elements_per_row;
        displs[i] = offset;
        offset += recvcounts[i];
    }
}

// multiplica pixel a pixel con mpi
template <class T> Image<T> Image<T>::operator*(const Image<T> &other) const
{
    assert(width == other.width && height == other.height && channels == other.channels);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int start_row, local_rows;
    calculate_row_distribution(height, size, rank, start_row, local_rows);

    // Si no hay filas locales, retornar imagen vacía
    if (local_rows <= 0) {
        return Image<T>();
    }

    Image<T> local_result(width, local_rows, channels);
    
    for (int j = 0; j < local_rows; j++) {
        for (int i = 0; i < width; i++) {
            for (int c = 0; c < channels; c++) {
                int global_row = start_row + j;
                T value = this->get(global_row, i, c) * other.get(global_row, i, c);
                local_result.set(j, i, c, value);
            }
        }
    }

    std::vector<int> recvcounts, displs;
    calculate_gather_params(height, width, channels, size, recvcounts, displs);

    Image<T> new_image;
    T* result_ptr = nullptr;
    
    if (rank == 0) {
        new_image = Image<T>(width, height, channels);
        result_ptr = new_image.get_matrix();
    }

    MPI_Datatype mpi_type = get_mpi_type<T>();
    MPI_Gatherv(local_result.get_matrix(), local_rows * width * channels, mpi_type,
                result_ptr, recvcounts.data(), displs.data(), mpi_type, 0, MPI_COMM_WORLD);

    return new_image;
}

// multiplica todos los pixeles x un numero
template <class T> Image<T> Image<T>::operator*(float scalar) const
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int start_row, local_rows;
    calculate_row_distribution(height, size, rank, start_row, local_rows);

    // Si no hay filas locales, retornar imagen vacía
    if (local_rows <= 0) {
        return Image<T>();
    }

    Image<T> local_result(width, local_rows, channels);
    
    for (int j = 0; j < local_rows; j++) {
        for (int i = 0; i < width; i++) {
            for (int c = 0; c < channels; c++) {
                int global_row = start_row + j;
                T value = (T)(this->get(global_row, i, c) * scalar);
                local_result.set(j, i, c, value);
            }
        }
    }

    std::vector<int> recvcounts, displs;
    calculate_gather_params(height, width, channels, size, recvcounts, displs);

    Image<T> new_image;
    T* result_ptr = nullptr;
    
    if (rank == 0) {
        new_image = Image<T>(width, height, channels);
        result_ptr = new_image.get_matrix();
    }

    MPI_Datatype mpi_type = get_mpi_type<T>();
    MPI_Gatherv(local_result.get_matrix(), local_rows * width * channels, mpi_type,
                result_ptr, recvcounts.data(), displs.data(), mpi_type, 0, MPI_COMM_WORLD);

    return new_image;
}

// suma pxl a pxl usando mpi
template <class T> Image<T> Image<T>::operator+(const Image<T> &other) const
{
    assert(width == other.width && height == other.height && channels == other.channels);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int start_row, local_rows;
    calculate_row_distribution(height, size, rank, start_row, local_rows);

    // Si no hay filas locales, crear array temporal vacío
    int local_size = local_rows * width * channels;
    std::vector<T> local_data;
    
    if (local_rows > 0) {
        local_data.resize(local_size);
        int idx = 0;
        for (int j = 0; j < local_rows; j++) {
            for (int i = 0; i < width; i++) {
                for (int c = 0; c < channels; c++) {
                    int global_row = start_row + j;
                    local_data[idx++] = this->get(global_row, i, c) + other.get(global_row, i, c);
                }
            }
        }
    }

    std::vector<int> recvcounts, displs;
    calculate_gather_params(height, width, channels, size, recvcounts, displs);

    Image<T> new_image;
    T* result_ptr = nullptr;
    
    if (rank == 0) {
        new_image = Image<T>(width, height, channels);
        result_ptr = new_image.get_matrix();
    }

    MPI_Datatype mpi_type = get_mpi_type<T>();
    
    // Usar Gatherv con puntero condicional
    const T* send_ptr = local_rows > 0 ? local_data.data() : nullptr;
    MPI_Gatherv(send_ptr, local_size, mpi_type,
                result_ptr, recvcounts.data(), displs.data(), mpi_type, 0, MPI_COMM_WORLD);

    return new_image;
}

// suma num a todos los pxl con mpi
template <class T>
Image<T> Image<T>::operator+(float scalar) const
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int start_row, local_rows;
    calculate_row_distribution(height, size, rank, start_row, local_rows);

    // Si no hay filas locales, crear array temporal vacío
    int local_size = local_rows * width * channels;
    std::vector<T> local_data;
    
    if (local_rows > 0) {
        local_data.resize(local_size);
        int idx = 0;
        for (int j = 0; j < local_rows; j++) {
            for (int i = 0; i < width; i++) {
                for (int c = 0; c < channels; c++) {
                    int global_row = start_row + j;
                    local_data[idx++] = (T)(this->get(global_row, i, c) + scalar);
                }
            }
        }
    }

    std::vector<int> recvcounts, displs;
    calculate_gather_params(height, width, channels, size, recvcounts, displs);

    Image<T> new_image;
    T* result_ptr = nullptr;
    
    if (rank == 0) {
        new_image = Image<T>(width, height, channels);
        result_ptr = new_image.get_matrix();
    }

    MPI_Datatype mpi_type = get_mpi_type<T>();
    
    const T* send_ptr = local_rows > 0 ? local_data.data() : nullptr;
    MPI_Gatherv(send_ptr, local_size, mpi_type,
                result_ptr, recvcounts.data(), displs.data(), mpi_type, 0, MPI_COMM_WORLD);

    return new_image;
}

// valor abs de todos pixeles con mpi
template <class T> Image<T> Image<T>::abs() const
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int start_row, local_rows;
    calculate_row_distribution(height, size, rank, start_row, local_rows);

    // Si no hay filas locales, crear array temporal vacío
    int local_size = local_rows * width * channels;
    std::vector<T> local_data;
    
    if (local_rows > 0) {
        local_data.resize(local_size);
        int idx = 0;
        for (int j = 0; j < local_rows; j++) {
            for (int i = 0; i < width; i++) {
                for (int c = 0; c < channels; c++) {
                    int global_row = start_row + j;
                    T value = this->get(global_row, i, c);
                    // Usar std::abs para tipos numéricos
                    if constexpr (std::is_floating_point<T>::value || std::is_integral<T>::value) {
                        local_data[idx++] = std::abs(value);
                    } else {
                        local_data[idx++] = value; // Para unsigned char, no hacer nada
                    }
                }
            }
        }
    }

    std::vector<int> recvcounts, displs;
    calculate_gather_params(height, width, channels, size, recvcounts, displs);

    Image<T> new_image;
    T* result_ptr = nullptr;
    
    if (rank == 0) {
        new_image = Image<T>(width, height, channels);
        result_ptr = new_image.get_matrix();
    }

    MPI_Datatype mpi_type = get_mpi_type<T>();
    
    const T* send_ptr = local_rows > 0 ? local_data.data() : nullptr;
    MPI_Gatherv(send_ptr, local_size, mpi_type,
                result_ptr, recvcounts.data(), displs.data(), mpi_type, 0, MPI_COMM_WORLD);

    return new_image;
}

// aplica filtro de convolucion (VERSIÓN CORREGIDA)
template <class T> Image<T> Image<T>::convolution(const Image<float> &kernel) const
{
    assert(kernel.width % 2 != 0 && kernel.height % 2 != 0 &&
           kernel.width == kernel.height && kernel.channels == 1);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int kernel_size = kernel.width;
    int halo = kernel_size / 2; 

    int start_row, local_rows;
    calculate_row_distribution(height, size, rank, start_row, local_rows);

    // Si no hay filas locales, retornar imagen vacía
    if (local_rows <= 0) {
        return Image<T>();
    }

    // Broadcast del kernel
    std::vector<float> kernel_data(kernel_size * kernel_size);
    if (rank == 0) {
        for (int u = 0; u < kernel_size; u++) {
            for (int v = 0; v < kernel_size; v++) {
                kernel_data[u * kernel_size + v] = kernel.get(u, v, 0);
            }
        }
    }
    MPI_Bcast(kernel_data.data(), kernel_size * kernel_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Calcular filas extendidas para el halo
    int extended_start = std::max(0, start_row - halo);
    int extended_end = std::min(height, start_row + local_rows + halo);

    // Procesar localmente
    std::vector<T> local_data(local_rows * width * channels);
    int idx = 0;
    
    for (int j = 0; j < local_rows; j++) {
        for (int i = 0; i < width; i++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0;
                int global_row = start_row + j;

                for (int u = 0; u < kernel_size; u++) {
                    for (int v = 0; v < kernel_size; v++) {
                        int s = global_row + u - halo;
                        int t = i + v - halo;

                        // Verificar límites
                        if (s >= 0 && s < height && t >= 0 && t < width) {
                            sum += this->get(s, t, c) * kernel_data[u * kernel_size + v];
                        }
                    }
                }
                local_data[idx++] = (T)(sum);
            }
        }
    }

    std::vector<int> recvcounts, displs;
    calculate_gather_params(height, width, channels, size, recvcounts, displs);

    Image<T> convolved;
    T* result_ptr = nullptr;
    
    if (rank == 0) {
        convolved = Image<T>(width, height, channels);
        result_ptr = convolved.get_matrix();
    }

    MPI_Datatype mpi_type = get_mpi_type<T>();
    MPI_Gatherv(local_data.data(), local_rows * width * channels, mpi_type,
                result_ptr, recvcounts.data(), displs.data(), mpi_type, 0, MPI_COMM_WORLD);

    return convolved;
}

// cambia tipo de datos de los pixeles
template <class T>
template <typename S>
Image<S> Image<T>::convert() const
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int start_row, local_rows;
    calculate_row_distribution(height, size, rank, start_row, local_rows);

    // Si no hay filas locales, crear array temporal vacío
    int local_size = local_rows * width * channels;
    std::vector<S> local_data;
    
    if (local_rows > 0) {
        local_data.resize(local_size);
        int idx = 0;
        for (int j = 0; j < local_rows; j++) {
            for (int i = 0; i < width; i++) {
                for (int c = 0; c < channels; c++) {
                    int global_row = start_row + j;
                    local_data[idx++] = (S)this->get(global_row, i, c);
                }
            }
        }
    }

    std::vector<int> recvcounts, displs;
    calculate_gather_params(height, width, channels, size, recvcounts, displs);

    Image<S> new_image;
    S* result_ptr = nullptr;
    
    if (rank == 0) {
        new_image = Image<S>(width, height, channels);
        result_ptr = new_image.get_matrix();
    }

    MPI_Datatype mpi_type = get_mpi_type<S>();
    
    const S* send_ptr = local_rows > 0 ? local_data.data() : nullptr;
    MPI_Gatherv(send_ptr, local_size, mpi_type,
                result_ptr, recvcounts.data(), displs.data(), mpi_type, 0, MPI_COMM_WORLD);

    return new_image;
}

// convierte de RGB -> escala de grises
template <class T>
Image<T> Image<T>::to_grayscale() const
{
    if (channels == 1){
        return *this;
    }
    
    // Nota: Esta función no está paralelizada
    Image<T> gray(width, height, 1);
    
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            float value = 0.299f * this->get(j, i, 0) + 
                          0.587f * this->get(j, i, 1) + 
                          0.114f * this->get(j, i, 2);
            gray.set(j, i, 0, (T)value);
        }
    }
    return gray;
}

// divide img en bloques cuadrados 
template <class T>
std::vector<Block<T>> Image<T>::get_blocks(int block_size)
{
    assert(width % block_size == 0 && height % block_size == 0);
    
    std::vector<Block<T>> blocks;
    
    int n_blocks_row = height / block_size;
    int n_blocks_col = width / block_size;
    
    blocks.reserve(n_blocks_row * n_blocks_col);
    
    for (int row = 0; row < height; row += block_size) {
        for (int col = 0; col < width; col += block_size) {
            Block<T> b;
            b.i = col;
            b.j = row;
            b.size = block_size;
            b.depth = channels;
            b.rowsize = width * channels;
            b.matrix = const_cast<Image<T>*>(this);
            blocks.push_back(b);
        }
    }
    
    return blocks;
}

template <class T>
Image<float> Image<T>::normalized() const
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int start_row, local_rows;
    calculate_row_distribution(height, size, rank, start_row, local_rows);
    
    // Encontrar mínimo y máximo local
    T local_min = std::numeric_limits<T>::max();
    T local_max = std::numeric_limits<T>::lowest();
    
    if (local_rows > 0) {
        for (int j = 0; j < local_rows; j++) {
            for (int i = 0; i < width; i++) {
                for (int c = 0; c < channels; c++) {
                    int global_row = start_row + j;
                    T value = this->get(global_row, i, c);
                    if (value < local_min) local_min = value;
                    if (value > local_max) local_max = value;
                }
            }
        }
    }
    
    // Reducir para encontrar mínimo y máximo global
    T global_min, global_max;
    MPI_Datatype mpi_type = get_mpi_type<T>();
    MPI_Allreduce(&local_min, &global_min, 1, mpi_type, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, mpi_type, MPI_MAX, MPI_COMM_WORLD);
    
    // Evitar división por cero
    float range = (float)(global_max - global_min);
    if (range == 0.0f) range = 1.0f;
    
    // Normalizar datos locales
    std::vector<float> local_data;
    if (local_rows > 0) {
        local_data.resize(local_rows * width * channels);
        int idx = 0;
        for (int j = 0; j < local_rows; j++) {
            for (int i = 0; i < width; i++) {
                for (int c = 0; c < channels; c++) {
                    int global_row = start_row + j;
                    T value = this->get(global_row, i, c);
                    local_data[idx++] = (float)(value - global_min) / range;
                }
            }
        }
    }

    std::vector<int> recvcounts, displs;
    calculate_gather_params(height, width, channels, size, recvcounts, displs);

    Image<float> new_image;
    float* result_ptr = nullptr;
    
    if (rank == 0) {
        new_image = Image<float>(width, height, channels);
        result_ptr = new_image.get_matrix();
    }

    const float* send_ptr = local_rows > 0 ? local_data.data() : nullptr;
    MPI_Gatherv(send_ptr, local_rows * width * channels, MPI_FLOAT,
                result_ptr, recvcounts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    return new_image;
}

#endif
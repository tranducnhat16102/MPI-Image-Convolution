#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h> // Để đo thời gian cơ bản

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" // Cần file này

#ifdef _WIN32
#define FSEEKO _fseeki64
#else
#define FSEEKO fseeko
#endif

typedef enum { RGB, GREY } color_t;

// --- Khai báo Hàm ---
void usage_seq(int argc, char** argv, char** image_filename, int* width, int* height, int* loops, color_t* imageType);
uint8_t* offset_seq(uint8_t* array, int i, int j_byte_offset, int padded_width_bytes);
void convolute_pixel_grey(uint8_t* src, uint8_t* dst, int i, int j, int padded_width_bytes, float** h);
void convolute_pixel_rgb(uint8_t* src, uint8_t* dst, int i, int j_byte_start, int padded_width_bytes, float** h);

// --- Hàm main Tuần tự ---
int main(int argc, char** argv) {
    // --- Khởi tạo biến ---
    char* image_filename = NULL;
    int width = 0, height = 0, loops = 0;
    color_t imageType = GREY;
    int image_channels = 1;
    int i, j, t; // Biến lặp

    clock_t start_time_total, end_time_total;
    clock_t start_time_conv, end_time_conv;
    double cpu_time_total, cpu_time_conv;

    start_time_total = clock(); // Bắt đầu đo tổng thời gian

    // === Giai đoạn 1: Initialization ===
    usage_seq(argc, argv, &image_filename, &width, &height, &loops, &imageType);
    image_channels = (imageType == GREY) ? 1 : 3;
    printf("[SEQ] Executing: Image=%s, Size=%dx%d, Loops=%d, Type=%s\n",
        image_filename, width, height, loops, (imageType == GREY ? "GREY" : "RGB"));

    // Khởi tạo Bộ lọc (Kernel)
    float** h = malloc(3 * sizeof(float*));
    if (!h) { fprintf(stderr, "[SEQ] Error allocating filter rows\n"); return EXIT_FAILURE; }
    for (i = 0; i < 3; i++) {
        h[i] = malloc(3 * sizeof(float));
        if (!h[i]) { fprintf(stderr, "[SEQ] Error allocating filter cols\n"); /* cleanup */ return EXIT_FAILURE; }
    }
    float gaussian_blur[3][3] = { {1.f / 16, 2.f / 16, 1.f / 16}, {2.f / 16, 4.f / 16, 2.f / 16}, {1.f / 16, 2.f / 16, 1.f / 16} };
    for (i = 0; i < 3; i++) { for (j = 0; j < 3; j++) { h[i][j] = gaussian_blur[i][j]; } }

    // Tính toán Kích thước có Padding
    int padded_rows = height + 2;
    int padded_cols = width + 2;
    int padded_width_bytes = padded_cols * image_channels;
    size_t padded_buffer_size = (size_t)padded_rows * padded_width_bytes;

    // Cấp phát Bộ đệm Ảnh (Có Padding)
    uint8_t* src_padded = NULL, * dst_padded = NULL, * tmp_padded = NULL;
    src_padded = calloc(1, padded_buffer_size);
    dst_padded = calloc(1, padded_buffer_size);
    if (!src_padded || !dst_padded) {
        fprintf(stderr, "[SEQ] Error allocating image buffers (size %zu)\n", padded_buffer_size);
        // cleanup h
        return EXIT_FAILURE;
    }
    printf("[SEQ] Allocated padded buffers (%dx%d bytes each)\n", padded_rows, padded_width_bytes);

    // === Giai đoạn 2: Input ===
    FILE* f_in = fopen(image_filename, "rb");
    if (!f_in) {
        fprintf(stderr, "[SEQ] Error opening input file: %s\n", image_filename);
        // cleanup buffers, h
        return EXIT_FAILURE;
    }
    printf("[SEQ] Reading raw image data...\n");
    size_t row_bytes_to_read = (size_t)width * image_channels;
    for (i = 0; i < height; ++i) { // Đọc từng hàng của ảnh gốc
        uint8_t* row_ptr_in_padded = offset_seq(src_padded, i + 1, image_channels, padded_width_bytes);
        size_t bytes_read = fread(row_ptr_in_padded, 1, row_bytes_to_read, f_in);
        if (bytes_read != row_bytes_to_read) {
            fprintf(stderr, "[SEQ] Error reading row %d. Read %zu, expected %zu\n", i, bytes_read, row_bytes_to_read);
            fclose(f_in); // cleanup
            return EXIT_FAILURE;
        }
    }
    fclose(f_in);
    printf("[SEQ] Finished reading.\n");

    // === Giai đoạn 3: Computation ===
    printf("[SEQ] Starting convolution loops (%d)...\n", loops);
    start_time_conv = clock(); // Bắt đầu đo thời gian tích chập

    for (t = 0; t < loops; ++t) {
        // Duyệt qua từng pixel cần tính trong ảnh đích (bỏ qua padding)
        for (i = 1; i <= height; ++i) { // Hàng logic 1 đến height
            for (j = 1; j <= width; ++j) { // Cột logic 1 đến width
                if (imageType == GREY) {
                    convolute_pixel_grey(src_padded, dst_padded, i, j, padded_width_bytes, h);
                }
                else { // RGB
                    // Offset byte của kênh R của pixel (i,j) trong buffer đệm
                    int current_pixel_r_byte_offset = image_channels + j * image_channels; // j cột logic, +1 cột padding trái
                    convolute_pixel_rgb(src_padded, dst_padded, i, current_pixel_r_byte_offset, padded_width_bytes, h);
                }
            } // Kết thúc vòng lặp j (cột)
        } // Kết thúc vòng lặp i (hàng)

        // Hoán đổi buffer nguồn và đích cho vòng lặp tiếp theo
        tmp_padded = src_padded;
        src_padded = dst_padded;
        dst_padded = tmp_padded; // dst_padded giờ chứa dữ liệu cũ/rác
        // Kết quả của vòng lặp t nằm trong src_padded
        // printf("[SEQ] Loop %d completed.\n", t + 1); // Có thể bỏ comment để xem tiến trình

    } // Kết thúc vòng lặp t

    end_time_conv = clock(); // Kết thúc đo thời gian tích chập
    cpu_time_conv = ((double)(end_time_conv - start_time_conv)) / CLOCKS_PER_SEC;
    printf("[SEQ] Convolution loops finished. Time elapsed: %f seconds\n", cpu_time_conv);

    // === Giai đoạn 4: Output Preparation ===
    printf("[SEQ] Preparing final image buffer (removing padding)...\n");
    size_t final_image_bytes = (size_t)width * height * image_channels;
    uint8_t* final_image_buffer = malloc(final_image_bytes);
    if (!final_image_buffer) {
        fprintf(stderr, "[SEQ] Error allocating final image buffer\n");
        // cleanup
        return EXIT_FAILURE;
    }

    // Copy dữ liệu từ src_padded (kết quả cuối) sang final_image_buffer
    for (i = 0; i < height; ++i) {
        uint8_t* src_row_start = offset_seq(src_padded, i + 1, image_channels, padded_width_bytes);
        uint8_t* dst_row_start = final_image_buffer + (size_t)i * width * image_channels;
        memcpy(dst_row_start, src_row_start, row_bytes_to_read); // row_bytes_to_read = width*channels
    }
    printf("[SEQ] Final buffer prepared.\n");

    // === Giai đoạn 5: Output ===
    char png_filename[512];
    char* image_basename = strrchr(image_filename, '\\');
    if (!image_basename) image_basename = strrchr(image_filename, '/');
    if (!image_basename) image_basename = image_filename; else image_basename++;
    snprintf(png_filename, sizeof(png_filename), "blur_SEQ_%s.png", image_basename);

    printf("[SEQ] Writing PNG file: %s\n", png_filename);
    int stride_in_bytes = width * image_channels;
    int success = stbi_write_png(png_filename, width, height, image_channels, final_image_buffer, stride_in_bytes);
    if (!success) { fprintf(stderr, "[SEQ] Error writing PNG file.\n"); }
    else { printf("[SEQ] Successfully wrote PNG file.\n"); }

    // === Giai đoạn 6: Cleanup ===
    printf("[SEQ] Cleaning up memory...\n");
    free(src_padded);
    free(dst_padded);
    free(final_image_buffer);
    for (i = 0; i < 3; ++i) { if (h[i]) free(h[i]); }
    free(h);
    free(image_filename);

    end_time_total = clock(); // Kết thúc đo tổng thời gian
    cpu_time_total = ((double)(end_time_total - start_time_total)) / CLOCKS_PER_SEC;
    printf("[SEQ] Total execution time: %f seconds\n", cpu_time_total);

    printf("[SEQ] Sequential execution finished.\n");
    return EXIT_SUCCESS;
}

// --- Định nghĩa Hàm ---

void usage_seq(int argc, char** argv, char** image_filename, int* width, int* height, int* loops, color_t* imageType) {
    if (argc == 6 && (!strcmp(argv[5], "grey") || !strcmp(argv[5], "rgb"))) {
        *image_filename = malloc((strlen(argv[1]) + 1) * sizeof(char));
        if (!*image_filename) { fprintf(stderr, "Usage Error: Failed malloc image name.\n"); exit(EXIT_FAILURE); }
        strcpy(*image_filename, argv[1]);
        *width = atoi(argv[2]); *height = atoi(argv[3]); *loops = atoi(argv[4]);
        *imageType = (!strcmp(argv[5], "grey")) ? GREY : RGB;

        if (*width <= 0 || *height <= 0 || *loops < 0) { // Cho phép loops = 0
            fprintf(stderr, "\nUsage Error: width/height must be positive, loops non-negative.\n");
            free(*image_filename); exit(EXIT_FAILURE);
        }
    }
    else {
        fprintf(stderr, "\nInput Error!\nUsage: %s image_name width height loops [rgb|grey]\n\n", argv[0]);
        exit(EXIT_FAILURE);
    }
}

uint8_t* offset_seq(uint8_t* array, int i, int j_byte_offset, int padded_width_bytes) {
    return &array[(size_t)padded_width_bytes * i + j_byte_offset];
}

void convolute_pixel_grey(uint8_t* src, uint8_t* dst, int i, int j, int padded_width_bytes, float** h) {
    float val = 0.0f; int k, l, ki, lj;
    for (k = -1, ki = 0; k <= 1; ++k, ++ki) {
        for (l = -1, lj = 0; l <= 1; ++l, ++lj) {
            int neighbor_row = i + k; int neighbor_col_byte = j + l;
            size_t index = (size_t)padded_width_bytes * neighbor_row + neighbor_col_byte;
            val += src[index] * h[ki][lj];
        }
    }
    size_t dst_index = (size_t)padded_width_bytes * i + j;
    if (val < 0.0f) dst[dst_index] = 0; else if (val > 255.0f) dst[dst_index] = 255; else dst[dst_index] = (uint8_t)(val + 0.5f);
}

void convolute_pixel_rgb(uint8_t* src, uint8_t* dst, int i, int j_byte_start_dst, int padded_width_bytes, float** h) {
    float r = 0.f, g = 0.f, b = 0.f; int k, l, ki, lj;
    for (k = -1, ki = 0; k <= 1; ++k, ++ki) {
        int neighbor_row = i + k;
        for (l = -1, lj = 0; l <= 1; ++l, ++lj) {
            int j_byte_offset_neighbor = j_byte_start_dst + (l * 3);
            size_t index_r = (size_t)padded_width_bytes * neighbor_row + j_byte_offset_neighbor;
            r += src[index_r] * h[ki][lj];
            g += src[index_r + 1] * h[ki][lj];
            b += src[index_r + 2] * h[ki][lj];
        }
    }
    if (r < 0.f) r = 0.f; else if (r > 255.f) r = 255.f; dst[j_byte_start_dst] = (uint8_t)(r + 0.5f);
    if (g < 0.f) g = 0.f; else if (g > 255.f) g = 255.f; dst[j_byte_start_dst + 1] = (uint8_t)(g + 0.5f);
    if (b < 0.f) b = 0.f; else if (b > 255.f) b = 255.f; dst[j_byte_start_dst + 2] = (uint8_t)(b + 0.5f);
}
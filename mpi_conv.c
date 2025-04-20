#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "mpi.h"
// #include <time.h> // Không cần thiết khi dùng MPI_Wtime

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" // Đảm bảo file này tồn tại

#ifdef _WIN32
#define FSEEKO _fseeki64 // Hoặc fseeko nếu trình biên dịch hỗ trợ
#else
#define FSEEKO fseeko
#endif


typedef enum { RGB, GREY } color_t;

// --- Function Prototypes ---
void convolute(uint8_t* src, uint8_t* dst, int row_from, int row_to, int col_from, int col_to, int padded_width_bytes, int padded_rows, float** h, color_t imageType);
void convolute_grey(uint8_t* src, uint8_t* dst, int x, int y, int padded_width_bytes, int padded_rows, float** h);
void convolute_rgb(uint8_t* src, uint8_t* dst, int x, int y_byte_start, int padded_width_bytes, int padded_rows, float** h);
void Usage(int argc, char** argv, char** image, int* width, int* height, int* loops, color_t* imageType);
uint8_t* offset(uint8_t* array, int i, int j_byte_offset, int padded_width_bytes);
int divide_rows(int rows, int cols, int workers);

// --- Main MPI Program ---
int main(int argc, char** argv) {
    int i, j, width = 0, height = 0, loops = 0, t;
    int row_div = 0, col_div = 0, rows = 0, cols = 0;
    int start_row = 0, start_col = 0;
    double timer = 0.0, start_wtime = 0.0;
    char* image = NULL;
    char* image_arg = NULL; // To store the original argv[1] for non-root processes
    color_t imageType = GREY; // Initialize default value
    int image_channels = 1; // Default, will be updated
    int padded_width_bytes = 0; // Bytes per row IN PADDED BUFFER
    size_t local_array_size_bytes = 0; // Size of padded local buffer

    // --- MPI Initialization ---
    int process_id, num_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Status status; // General status object if needed, often ignored

    // --- KHAI BÁO SRC, DST, TMP ---
    uint8_t* src = NULL;
    uint8_t* dst = NULL;
    uint8_t* tmp = NULL;

    // --- Parameter Handling ---
    if (process_id == 0) {
        Usage(argc, argv, &image, &width, &height, &loops, &imageType); // Root parses arguments
        row_div = divide_rows(height, width, num_processes);

        // Check if a valid division was found and dimensions are divisible
        if (row_div <= 0) {
            fprintf(stderr, "Process 0: Cannot find a suitable process grid decomposition for %d processes and image size %dx%d.\n", num_processes, width, height);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (height % row_div != 0) {
            fprintf(stderr, "Process 0: Cannot divide height (%d) by number of process rows (%d).\n", height, row_div);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        col_div = num_processes / row_div;
        if (width % col_div != 0) {
            fprintf(stderr, "Process 0: Cannot divide width (%d) by number of process columns (%d).\n", width, col_div);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        printf("Process 0: Decomposition: %d rows of processes, %d columns of processes.\n", row_div, col_div);
    }
    else {
        // Store potential pointer to argument for later use (safer approach is Bcast)
        if (argc > 1 && argv[1] != NULL) {
            image_arg = argv[1];
        }
    }

    // --- Broadcast Parameters ---
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&loops, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imageType, 1, MPI_INT, 0, MPI_COMM_WORLD); // Bcast enum value as int
    MPI_Bcast(&row_div, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&col_div, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // --- All Processes Allocate Image Name Buffer and Receive Name ---
    int image_name_len = 0;
    if (process_id == 0) {
        image_name_len = strlen(image) + 1;
    }
    MPI_Bcast(&image_name_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (process_id != 0) {
        image = malloc(image_name_len * sizeof(char));
        if (image == NULL) {
            fprintf(stderr, "Process %d: Failed to allocate memory for image name (%d bytes).\n", process_id, image_name_len);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    // Broadcast the actual image name string from root to all others
    MPI_Bcast(image, image_name_len, MPI_CHAR, 0, MPI_COMM_WORLD);


    // --- Local Calculation ---
    rows = height / row_div; // Rows per process
    cols = width / col_div;  // Cols per process
    start_row = (process_id / col_div) * rows;
    start_col = (process_id % col_div) * cols;
    image_channels = (imageType == GREY) ? 1 : 3; // SET THE CORRECT NUMBER OF CHANNELS

    // Calculate padded dimensions and strides (bytes)
    int padded_cols = cols + 2; // +1 left, +1 right padding
    int padded_rows = rows + 2; // +1 top, +1 bottom padding
    padded_width_bytes = padded_cols * image_channels; // Bytes per row IN PADDED BUFFER
    local_array_size_bytes = (size_t)padded_rows * padded_width_bytes; // Size of padded local buffer

    // --- MPI Datatypes for Halo Exchange (using correct image_channels) ---
    MPI_Datatype row_type; // For sending/receiving full rows of data (cols * channels bytes)
    MPI_Datatype col_type; // For sending/receiving full columns of data (rows * channels bytes, non-contiguous)

    MPI_Type_contiguous(cols * image_channels, MPI_BYTE, &row_type);
    MPI_Type_vector(rows,                  // number of blocks (rows)
        image_channels,        // block length (1 pixel = 1 or 3 bytes)
        padded_width_bytes,    // stride between start of blocks (bytes) in padded buffer
        MPI_BYTE, &col_type);

    MPI_Type_commit(&row_type);
    MPI_Type_commit(&col_type);

    // --- Filter Initialization (Same for all processes) ---
    float** h = malloc(3 * sizeof(float*));
    if (h == NULL) { fprintf(stderr, "Proc %d: Failed malloc h (rows).\n", process_id); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); }
    for (i = 0; i < 3; i++) {
        h[i] = malloc(3 * sizeof(float));
        if (h[i] == NULL) {
            fprintf(stderr, "Proc %d: Failed malloc h (cols).\n", process_id);
            // Partial cleanup needed before abort
            for (int k = 0; k < i; ++k) free(h[k]); free(h); if (image) free(image);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    float gaussian_blur[3][3] = { {1.0f / 16, 2.0f / 16, 1.0f / 16}, {2.0f / 16, 4.0f / 16, 2.0f / 16}, {1.0f / 16, 2.0f / 16, 1.0f / 16} };
    for (i = 0; i < 3; i++) { for (j = 0; j < 3; j++) { h[i][j] = gaussian_blur[i][j]; } }

    // --- Allocate Local Image Buffers (WITH PADDING using correct size) ---
    src = calloc(1, local_array_size_bytes); // Use calloc for zero-initialization (padding)
    dst = calloc(1, local_array_size_bytes);
    if (src == NULL || dst == NULL) {
        fprintf(stderr, "Process %d: Not enough memory for image buffers (size %zu bytes).\n", process_id, local_array_size_bytes);
        if (src) free(src); if (dst) free(dst);
        // Cleanup filter, image name, MPI types...
        for (i = 0; i < 3; ++i) if (h && h[i]) free(h[i]); if (h) free(h);
        if (image) free(image);
        MPI_Type_free(&row_type); MPI_Type_free(&col_type);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // --- Parallel Read (Using MPI I/O into PADDED buffer) ---
    MPI_File fh;
    int ferr = MPI_File_open(MPI_COMM_WORLD, image, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (ferr != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING]; int len;
        MPI_Error_string(ferr, error_string, &len);
        fprintf(stderr, "Process %d: Could not open input file %s. MPI Error: %s\n", process_id, image, error_string);
        // Cleanup...
        free(src); free(dst); for (i = 0; i < 3; ++i) if (h && h[i]) free(h[i]); if (h) free(h); if (image) free(image); MPI_Type_free(&row_type); MPI_Type_free(&col_type);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    size_t row_bytes_to_read = (size_t)cols * image_channels; // Use correct channel count
    printf("Process %d: Reading %d rows, %zu bytes per row (chans=%d).\n", process_id, rows, row_bytes_to_read, image_channels);
    for (i = 1; i <= rows; i++) { // Read into rows 1 to 'rows' of padded buffer
        // Offset in file (uses correct channel count)
        MPI_Offset disp = (MPI_Offset)(start_row + i - 1) * width * image_channels + (MPI_Offset)start_col * image_channels;
        // Pointer in memory (skipping left padding)
        uint8_t* row_ptr = offset(src, i, image_channels, padded_width_bytes); // Uses correct channel count for padding offset

        // Use read_at_all (generally preferred over set_view + read_all)
        ferr = MPI_File_read_at_all(fh, disp, row_ptr, row_bytes_to_read, MPI_BYTE, &status);

        if (ferr != MPI_SUCCESS) {
            char error_string[MPI_MAX_ERROR_STRING]; int len;
            MPI_Error_string(ferr, error_string, &len);
            fprintf(stderr, "P%d Error MPI_File_read_at_all row %d. MPI Error: %s\n", process_id, i, error_string);
            MPI_File_close(&fh); /* Cleanup... */ MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        int count; MPI_Get_count(&status, MPI_BYTE, &count);
        if (count != row_bytes_to_read) {
            fprintf(stderr, "P%d Read Error row %d. Read %d, expected %zu.\n", process_id, i, count, row_bytes_to_read);
            MPI_File_close(&fh); /* Cleanup... */ MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    MPI_File_close(&fh);
    printf("Process %d: Finished read.\n", process_id);


    // --- Compute Neighbours ---
    int north = (start_row > 0) ? process_id - col_div : MPI_PROC_NULL;
    int south = (start_row + rows < height) ? process_id + col_div : MPI_PROC_NULL;
    int west = (start_col > 0) ? process_id - 1 : MPI_PROC_NULL;
    int east = (start_col + cols < width) ? process_id + 1 : MPI_PROC_NULL;

    MPI_Barrier(MPI_COMM_WORLD); // Sync before starting computation timer
    start_wtime = MPI_Wtime();

    // --- Main Convolution Loop ---
    // Chỉ chạy nếu loops > 0
    if (loops > 0) {
        MPI_Request send_reqs[4], recv_reqs[4]; // Separate request arrays
        int n_send_reqs, n_recv_reqs;           // Separate counters

        for (t = 0; t < loops; t++) {
            n_send_reqs = 0; n_recv_reqs = 0; // Reset counters

            // --- Non-blocking Halo Exchange (Corrected Logic) ---
            // Send North / Recv From South
            if (north != MPI_PROC_NULL) MPI_Isend(offset(src, 1, image_channels, padded_width_bytes), 1, row_type, north, t * 10 + 1, MPI_COMM_WORLD, &send_reqs[n_send_reqs++]);
            if (south != MPI_PROC_NULL) MPI_Irecv(offset(src, rows + 1, image_channels, padded_width_bytes), 1, row_type, south, t * 10 + 1, MPI_COMM_WORLD, &recv_reqs[n_recv_reqs++]);
            // Send South / Recv From North
            if (south != MPI_PROC_NULL) MPI_Isend(offset(src, rows, image_channels, padded_width_bytes), 1, row_type, south, t * 10 + 2, MPI_COMM_WORLD, &send_reqs[n_send_reqs++]);
            if (north != MPI_PROC_NULL) MPI_Irecv(offset(src, 0, image_channels, padded_width_bytes), 1, row_type, north, t * 10 + 2, MPI_COMM_WORLD, &recv_reqs[n_recv_reqs++]);
            // Send West / Recv From East
            if (west != MPI_PROC_NULL) MPI_Isend(offset(src, 1, image_channels, padded_width_bytes), 1, col_type, west, t * 10 + 3, MPI_COMM_WORLD, &send_reqs[n_send_reqs++]);
            if (east != MPI_PROC_NULL) MPI_Irecv(offset(src, 1, (cols + 1) * image_channels, padded_width_bytes), 1, col_type, east, t * 10 + 3, MPI_COMM_WORLD, &recv_reqs[n_recv_reqs++]);
            // Send East / Recv From West
            if (east != MPI_PROC_NULL) MPI_Isend(offset(src, 1, cols * image_channels, padded_width_bytes), 1, col_type, east, t * 10 + 4, MPI_COMM_WORLD, &send_reqs[n_send_reqs++]);
            if (west != MPI_PROC_NULL) MPI_Irecv(offset(src, 1, 0 * image_channels, padded_width_bytes), 1, col_type, west, t * 10 + 4, MPI_COMM_WORLD, &recv_reqs[n_recv_reqs++]);

            // --- Compute Inner Part (Overlap Communication) ---
            // Tính toán phần không cần dữ liệu halo trước
            if (rows > 2 && cols > 2) {
                convolute(src, dst, 2, rows - 1, 2, cols - 1, padded_width_bytes, padded_rows, h, imageType);
            }

            // --- Wait for Halo Data (Receives ONLY) ---
            // Đợi nhận đủ dữ liệu halo từ các neighbor
            if (n_recv_reqs > 0) {
                MPI_Waitall(n_recv_reqs, recv_reqs, MPI_STATUSES_IGNORE);
            }

            // --- Compute Boundaries (sau khi đã nhận halo) ---
            // Tính toán các hàng/cột biên cần dữ liệu halo
            if (rows > 0 && cols > 0) convolute(src, dst, 1, 1, 1, cols, padded_width_bytes, padded_rows, h, imageType); // Top row + corners
            if (rows > 1 && cols > 0) convolute(src, dst, rows, rows, 1, cols, padded_width_bytes, padded_rows, h, imageType); // Bottom row + corners
            if (cols > 0 && rows > 2) convolute(src, dst, 2, rows - 1, 1, 1, padded_width_bytes, padded_rows, h, imageType); // Left edge (excl corners)
            if (cols > 1 && rows > 2) convolute(src, dst, 2, rows - 1, cols, cols, padded_width_bytes, padded_rows, h, imageType); // Right edge (excl corners)

            // --- Wait for Sends to Complete ---
            // Đảm bảo dữ liệu đã được gửi đi hết trước khi sang vòng lặp sau
            if (n_send_reqs > 0) {
                MPI_Waitall(n_send_reqs, send_reqs, MPI_STATUSES_IGNORE);
            }

            // --- Swap Buffers ---
            // Chuẩn bị cho vòng lặp tiếp theo, kết quả hiện tại nằm trong 'src'
            tmp = src;
            src = dst;
            dst = tmp;
        } // end loop t
    } // end if loops > 0

    // --- Stop Timer ---
    // Nếu loops=0, timer sẽ gần như bằng 0. Nếu loops > 0, nó sẽ đo thời gian thực thi vòng lặp.
    timer = MPI_Wtime() - start_wtime;

    // --- Gather Results to Root and Write PNG ---
    uint8_t* final_image_buffer = NULL;
    int* recvcounts = NULL;
    int* displs = NULL;

    if (process_id == 0) {
        size_t final_image_size_bytes = (size_t)width * height * image_channels; // Correct size
        final_image_buffer = (uint8_t*)malloc(final_image_size_bytes);
        recvcounts = (int*)malloc(num_processes * sizeof(int));
        displs = (int*)malloc(num_processes * sizeof(int));
        if (!final_image_buffer || !recvcounts || !displs) { /* Error handling */ MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); }

        // Calculate recvcounts and displacements for Gatherv (using correct channels)
        for (int p = 0; p < num_processes; ++p) {
            int p_rows = height / row_div; int p_cols = width / col_div;
            int p_start_row = (p / col_div) * p_rows; int p_start_col = (p % col_div) * p_cols;
            recvcounts[p] = p_rows * p_cols * image_channels; // Use correct channel count
            displs[p] = ((size_t)p_start_row * width + p_start_col) * image_channels; // Use correct channel count
        }
        size_t total_recv_bytes = 0; for (int p = 0; p < num_processes; ++p) total_recv_bytes += recvcounts[p];
        if (total_recv_bytes != final_image_size_bytes) { /* Error checking */ MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); }
    }

    // Create MPI Datatype for the local data block (excluding padding)
    MPI_Datatype local_block_type;
    MPI_Type_vector(rows,                     // Number of rows
        cols * image_channels,    // Bytes per row (block length)
        padded_width_bytes,       // Stride between rows in src buffer (bytes)
        MPI_BYTE, &local_block_type);
    MPI_Type_commit(&local_block_type);

    // Pointer to the start of actual data in the source buffer
    // Kết quả cuối cùng nằm trong 'src' sau vòng lặp cuối (hoặc sau khi đọc nếu loops=0)
    uint8_t* local_data_start = offset(src, 1, image_channels, padded_width_bytes);

    // === DEBUG CODE Section (Optional: Can be enabled with #if 1) ===
#if 0 // Đặt là 1 để bật debug ghi file cục bộ trước Gather
    {
        char debug_filename[100];
        // Đặt tên file theo ý muốn, ví dụ bao gồm số vòng lặp cuối
        sprintf(debug_filename, "debug_final_proc_%d_L%d.data", process_id, loops);
        FILE* f_debug = fopen(debug_filename, "wb");
        if (f_debug) {
            size_t local_block_bytes = (size_t)rows * cols * image_channels;
            uint8_t* temp_buffer = malloc(local_block_bytes);
            if (temp_buffer) {
                uint8_t* current_dest = temp_buffer;
                for (int r = 1; r <= rows; ++r) { // Copy từ buffer có padding
                    uint8_t* current_src_row_start = offset(src, r, image_channels, padded_width_bytes);
                    memcpy(current_dest, current_src_row_start, cols * image_channels);
                    current_dest += cols * image_channels;
                }
                fwrite(temp_buffer, 1, local_block_bytes, f_debug);
                free(temp_buffer);
                printf("Process %d wrote FINAL debug file %s (%zu bytes)\n", process_id, debug_filename, local_block_bytes);
            }
            else { fprintf(stderr, "P%d Failed alloc debug temp buffer\n", process_id); }
            fclose(f_debug);
        }
        else { fprintf(stderr, "P%d Failed open debug file %s\n", process_id, debug_filename); }
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
    // === END DEBUG CODE Section ===

    // Gather the data
    MPI_Gatherv(local_data_start, 1, local_block_type, // Send 1 block described by the type
        final_image_buffer, recvcounts, displs, MPI_BYTE, // Receive contiguous bytes on root
        0, MPI_COMM_WORLD);

    MPI_Type_free(&local_block_type); // Free the derived datatype

    // --- Root Process Writes PNG ---
    if (process_id == 0) {
        char png_filename[512];
        char* image_basename = strrchr(image, '\\');
        if (!image_basename) image_basename = strrchr(image, '/');
        if (!image_basename) image_basename = image; else image_basename++;
        snprintf(png_filename, sizeof(png_filename), "blur_%s.png", image_basename);
        int stride_in_bytes = width * image_channels; // Use correct channel count

        printf("Process 0: Writing PNG file: %s (%dx%d, channels: %d)\n", png_filename, width, height, image_channels);
        int success = stbi_write_png(png_filename, width, height, image_channels, final_image_buffer, stride_in_bytes);

        if (!success) { fprintf(stderr, "Process 0: Failed to write PNG file %s\n", png_filename); }
        else { printf("Process 0: Successfully wrote PNG file %s\n", png_filename); }

        free(final_image_buffer); free(recvcounts); free(displs);
    }

    // --- Aggregate Timings ---
    double max_timer;
    MPI_Reduce(&timer, &max_timer, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (process_id == 0) { printf("Max execution time across all processes: %.6f seconds (%d loops)\n", max_timer, loops); }

    // --- Cleanup (All Processes) ---
    free(src); free(dst); // Free cả src và dst
    for (i = 0; i < 3; ++i) if (h && h[i]) free(h[i]); if (h) free(h); // Kiểm tra trước khi free h
    if (image) free(image);
    MPI_Type_free(&row_type); MPI_Type_free(&col_type);

    // --- Finalize MPI ---
    MPI_Finalize();
    return EXIT_SUCCESS;
}


// --- Function Definitions ---

// Usage function - Sửa để chấp nhận loops = 0
void Usage(int argc, char** argv, char** image, int* width, int* height, int* loops, color_t* imageType) {
    if (argc == 6 && (!strcmp(argv[5], "grey") || !strcmp(argv[5], "rgb"))) {
        *image = malloc((strlen(argv[1]) + 1) * sizeof(char));
        if (*image == NULL) { fprintf(stderr, "Usage Error (P0): Failed malloc image name.\n"); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); }
        strcpy(*image, argv[1]);
        *width = atoi(argv[2]); *height = atoi(argv[3]); *loops = atoi(argv[4]);
        *imageType = (!strcmp(argv[5], "grey")) ? GREY : RGB;

        if (*width <= 0 || *height <= 0 || *loops < 0) { // Chấp nhận loops=0
            fprintf(stderr, "\nError (P0): width (%d), height (%d) must be positive, loops (%d) must be non-negative.\n\n", *width, *height, *loops);
            free(*image); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        printf("Process 0: Input Image: %s, Size: %dx%d, Loops: %d, Type: %s\n", *image, *width, *height, *loops, (*imageType == GREY) ? "GREY" : "RGB");
    }
    else {
        fprintf(stderr, "\nInput Error (P0)!\nUsage: %s image_name width height loops [rgb|grey]\n\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

// Convolute dispatch function (Đã sửa offset RGB)
void convolute(uint8_t* src, uint8_t* dst, int row_from, int row_to, int col_from, int col_to, int padded_width_bytes, int padded_rows, float** h, color_t imageType) {
    int i, j;
    int image_channels = (imageType == GREY) ? 1 : 3;
    for (i = row_from; i <= row_to; i++) {
        for (j = col_from; j <= col_to; j++) {
            if (imageType == GREY) {
                convolute_grey(src, dst, i, j, padded_width_bytes, padded_rows, h);
            }
            else { // RGB
                // Offset byte của kênh R của pixel (i,j) tính từ ĐẦU HÀNG ĐỆM
                int current_pixel_r_byte_offset = image_channels + j * image_channels; // Đúng cho RGB
                convolute_rgb(src, dst, i, current_pixel_r_byte_offset, padded_width_bytes, padded_rows, h);
            }
        }
    }
}

// Convolute Grey (Giả định buffer có padding)
void convolute_grey(uint8_t* src, uint8_t* dst, int x, int y, int padded_width_bytes, int padded_rows, float** h) {
    int i, j_idx, k, l; float val = 0.0f;
    for (i = x - 1, k = 0; i <= x + 1; i++, k++) { // i, j_idx là chỉ số vật lý
        for (j_idx = y - 1, l = 0; j_idx <= y + 1; j_idx++, l++) {
            size_t index = (size_t)padded_width_bytes * i + j_idx;
            val += src[index] * h[k][l];
        }
    }
    size_t dst_index = (size_t)padded_width_bytes * x + y;
    if (val < 0.0f) dst[dst_index] = 0; else if (val > 255.0f) dst[dst_index] = 255; else dst[dst_index] = (uint8_t)(val + 0.5f);
}

// Convolute RGB (Giả định buffer có padding)
void convolute_rgb(uint8_t* src, uint8_t* dst, int x, int y_byte_start, int padded_width_bytes, int padded_rows, float** h) {
    int i, j_byte_offset, k, l; float redval = 0.0f, greenval = 0.0f, blueval = 0.0f;
    for (i = x - 1, k = 0; i <= x + 1; i++, k++) { // i là chỉ số hàng vật lý
        // y_byte_start là offset byte của R kênh trung tâm TỪ ĐẦU HÀNG ĐỆM
        for (j_byte_offset = y_byte_start - 3, l = 0; j_byte_offset <= y_byte_start + 3; j_byte_offset += 3, l++) {
            size_t index = (size_t)padded_width_bytes * i + j_byte_offset; // index của R
            redval += src[index] * h[k][l];
            greenval += src[index + 1] * h[k][l];
            blueval += src[index + 2] * h[k][l];
        }
    }
    size_t dst_index = (size_t)padded_width_bytes * x + y_byte_start; // index của R đích
    if (redval < 0.0f) dst[dst_index] = 0; else if (redval > 255.0f) dst[dst_index] = 255; else dst[dst_index] = (uint8_t)(redval + 0.5f);
    if (greenval < 0.0f) dst[dst_index + 1] = 0; else if (greenval > 255.0f) dst[dst_index + 1] = 255; else dst[dst_index + 1] = (uint8_t)(greenval + 0.5f);
    if (blueval < 0.0f) dst[dst_index + 2] = 0; else if (blueval > 255.0f) dst[dst_index + 2] = 255; else dst[dst_index + 2] = (uint8_t)(blueval + 0.5f);
}

// Offset function
uint8_t* offset(uint8_t* array, int i, int j_byte_offset, int padded_width_bytes) {
    return &array[(size_t)padded_width_bytes * i + j_byte_offset];
}

// divide_rows function
int divide_rows(int rows, int cols, int workers) {
    int per, rows_to, cols_to, best = 0; int per_min = rows + cols + 1;
    for (rows_to = 1; rows_to <= workers; ++rows_to) {
        if (workers % rows_to != 0 || rows % rows_to != 0) continue;
        cols_to = workers / rows_to; if (cols % cols_to != 0) continue;
        per = rows / rows_to + cols / cols_to;
        if (per < per_min) { per_min = per; best = rows_to; }
    }
    return (best == 0) ? -1 : best;
}
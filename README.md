# Parallel Image Convolution using MPI (C Implementation)

## Giới thiệu

Dự án này thực hiện phép toán tích chập ảnh (image convolution), một kỹ thuật xử lý ảnh cơ bản, bằng cả phương pháp tuần tự truyền thống và phương pháp song song sử dụng MPI (Message Passing Interface). Mục tiêu chính là khám phá và trình diễn sự tăng tốc hiệu năng đạt được thông qua việc song song hóa một tác vụ tính toán nặng. Phép tích chập cụ thể được triển khai là Gaussian blur sử dụng kernel 3x3.

Dự án bao gồm:

1.  **Phiên bản Tuần tự (Sequential):** Một chương trình C đơn luồng thực hiện tích chập làm cơ sở so sánh và kiểm tra tính đúng đắn.
2.  **Phiên bản Song song (Parallel):** Một chương trình C sử dụng MPI để phân tán công việc tính toán tích chập trên nhiều tiến trình (processes), nhằm giảm thời gian thực thi tổng thể.

## Bài toán

Tích chập ảnh yêu cầu áp dụng một kernel (ma trận nhỏ) lên từng pixel của ảnh đầu vào. Để tính giá trị của một pixel đầu ra, cần truy cập giá trị của pixel đó và các pixel lân cận trong ảnh đầu vào. Với ảnh lớn hoặc khi áp dụng bộ lọc nhiều lần (ví dụ: tăng độ mờ), khối lượng tính toán và truy cập bộ nhớ trở nên rất lớn, khiến phiên bản tuần tự chạy chậm và trở thành nút cổ chai (bottleneck).

## Chiến lược Song song hóa

Để giải quyết vấn đề hiệu năng, phiên bản song song sử dụng các kỹ thuật sau:

1.  **Mô hình Lập trình:** **SPMD (Single Program, Multiple Data)** - Tất cả các tiến trình MPI chạy cùng một mã nguồn nhưng hoạt động trên các phần dữ liệu khác nhau.
2.  **Loại hình Song song:** **Song song Dữ liệu (Data Parallelism)**.
3.  **Phương pháp:** **Phân rã Miền dữ liệu (Domain Decomposition)**. Lưới pixel của ảnh được chia thành các khối con (hình chữ nhật). Mỗi tiến trình MPI được gán trách nhiệm tính toán cho một khối con.
4.  **Xử lý Biên và Phụ thuộc Dữ liệu:**
    *   **Padding & Vùng Halo (Ghost Zones):** Mỗi tiến trình cấp phát bộ đệm cục bộ lớn hơn một chút so với khối dữ liệu thực tế nó quản lý, tạo ra một viền padding (vùng halo) rộng 1 pixel.
    *   **Trao đổi Halo (Halo Exchange):** Trước mỗi vòng lặp tính toán tích chập, các tiến trình trao đổi dữ liệu biên với các tiến trình lân cận (Bắc, Nam, Đông, Tây) để điền vào vùng halo của nhau. Việc này đảm bảo mỗi tiến trình có đủ dữ liệu lân cận để tính toán các pixel ở biên khối của mình. Giao tiếp được thực hiện bằng các hàm non-blocking `MPI_Isend`/`MPI_Irecv` và đồng bộ hóa bằng `MPI_Wait`/`MPI_Waitall`.
5.  **Input/Output Song song:**
    *   **Input:** Sử dụng **MPI I/O** (`MPI_File_open`, `MPI_File_seek`, `MPI_File_read`). Mỗi tiến trình tính toán vị trí (offset) và đọc trực tiếp phần dữ liệu ảnh tương ứng với khối của mình từ file raw đầu vào vào bộ đệm cục bộ.
    *   **Output:** Sử dụng **MPI I/O** (`MPI_File_open`, `MPI_File_seek`, `MPI_File_write`). Sau khi hoàn thành tính toán, mỗi tiến trình ghi phần kết quả của mình (bỏ qua padding/halo) vào đúng vị trí trong file ảnh raw đầu ra chung. *(Lưu ý: Cách tiếp cận khác có thể là dùng MPI_Gatherv để thu thập về tiến trình 0 rồi P0 ghi file, nhưng code hiện tại dùng parallel write).*

## Cấu trúc Code

*   `sequential_convolution.c` (hoặc tên tương tự): Mã nguồn phiên bản tuần tự.
*   `mpi_convolution.c` (hoặc tên tương tự bạn đặt, ví dụ code bạn cung cấp gần nhất): Mã nguồn phiên bản song song sử dụng MPI.
*   `stb_image_write.h`: Thư viện header-only của Sean Barrett để ghi file ảnh PNG (cần tải về và đặt cùng thư mục mã nguồn).


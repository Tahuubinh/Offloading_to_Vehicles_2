# Luutam
Hướng dẫn sử dụng:
  Thao tác chủ yếu trong code, data_task:
  - data_task lưu các dataset tạo bởi random_task.py trong code
  - code làm việc chủ yếu với main.py và config.py:
    + Trong config chỉ cần để ý DATA_TASK để chọn dataset <br />
      eg: DATA_TASK = os.path.join(LINK_PROJECT, "data_task/200 normal task 900 - 1000")
    + Trong main.py lựa chọn các thuật toán để chạy. Muốn thay đổi các siêu tham số phải thay đổi trực tiếp trong các hàm Run_...

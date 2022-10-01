[Value-based reinforcement learning approaches for task offloading in Delay Constrained Vehicular Edge Computing](https://www.sciencedirect.com/science/article/abs/pii/S0952197622001336), Engineering Applications of Artificial Intelligence 2022.

## Citation
```
@article{binh2022value,
  title={Value-based reinforcement learning approaches for task offloading in Delay Constrained Vehicular Edge Computing},
  author={Son, Do Bao and Binh, Ta Huu and Vo, Hiep Khac and Nguyen, Binh Minh and Binh, Huynh Thi Thanh and Yu, Shui and others},
  journal={Engineering Applications of Artificial Intelligence},
  volume={113},
  pages={104898},
  year={2022},
  publisher={Elsevier}
}
```

# Luutam
Hướng dẫn sử dụng:
  Thao tác chủ yếu trong code, data_task:
  - data_task lưu các dataset tạo bởi random_task.py trong code
  - code làm việc chủ yếu với main.py và config.py:
    + Trong config chỉ cần để ý DATA_TASK để chọn dataset <br />
      eg: DATA_TASK = os.path.join(LINK_PROJECT, "data_task/200 normal task 900 - 1000")
    + Trong main.py lựa chọn các thuật toán để chạy. Muốn thay đổi các siêu tham số phải thay đổi trực tiếp trong các hàm Run_...

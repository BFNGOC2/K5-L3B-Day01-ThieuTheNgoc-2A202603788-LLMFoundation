# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603788
- **Họ và tên:** Thiều Thế Ngọc
- **SEED** (in ở ô đầu notebook): 7050

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính
| Mô hình tiếp cận | F1 test (Full) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.911 | 0.974 | 0.077 ms | 0.0 |
| 2. BoW + MLP (deep learning) | 0.941 | 0.974 | 0.885 ms | 0.0 |
| 3a. Encoder (MiniLM) + kNN | 0.887 | 0.976 | 67.215 ms | 0.0 |
| 3b. Encoder (MiniLM) + LogReg | 0.963 | 0.974 | 41.887 ms | 0.0 |
| 3c. Encoder (MiniLM) + MLP head | 0.969 | 0.974 | 38.872 ms | 0.0 |
| 4a. LLM nhỏ zero-shot | N/A | 0.952 | 2636.325 ms | 0.0 (Groq) |
| 4b. LLM lớn zero-shot | N/A | 0.952 | 2744.057 ms | 0.0 (Groq) |
| 4c. LLM nhỏ few-shot | N/A | 0.947 | 3711.163 ms | 0.0 (Groq) |

## 3. Dự đoán và thực tế
Dự đoán: Mạng 2 lớp tuyến tính xếp liên tiếp sẽ có hiệu năng F1 nằm ở mức trung gian giữa Phần 1 và Phần 2:
• Dự đoán: Mô hình sẽ bị phân kỳ hoàn toàn, Loss nhảy vọt lên vô hạn hoặc dao động không thể giảm.
• Thực tế: Sau một cú gai nhọn vọt lên ở vài epoch đầu tiên, mô hình tự hội tụ về mức loss thấp kỷ lục (0.007).
• Vì sao lệch: Do ma trận đầu vào Bag-of-Words đã được nhị phân hóa (chỉ chứa 0 và 1) và hàm mất mát BCE Loss có vùng đạo hàm biên rất phẳng, giúp mô hình vượt qua biến động ban đầu một cách an toàn.
Dự đoán mlp_norelu (Mạng MLP bỏ tầng ReLU):
• Dự đoán: Mạng 2 lớp tuyến tính xếp liên tiếp sẽ có hiệu năng F1 nằm ở mức trung gian giữa Phần 1 và Phần 2.
• Thực tế: Hiệu năng F1 bị kéo tụt xuống bằng hoặc xấp xỉ y hệt mô hình Logistic Regression tuyến tính ở Phần 1.
• Vì sao lệch: Về mặt toán học, hai hàm tuyến tính xếp liên tiếp mà không có hàm phi tuyến ngăn cách sẽ tự động nhân gộp ma trận lại thành một hàm tuyến tính phẳng duy nhất, làm mô hình mất hoàn toàn khả năng học phi tuyến.
Dự đoán knn_semantic (Truy vấn né từ khóa rác):
• Dự đoán: Mô hình Embedding sẽ bị nhầm lẫn và đoán sai thành tin thường vì không thấy chữ "free", "win".
• Thực tế: Mô hình chấm điểm Cosine chính xác 100% cả 6 kết quả gần nhất đều là spam.
• Vì sao lệch: Vì mô hình Sentence Transformers mã hóa ngữ nghĩa toàn câu. Cụm từ ngụy trang như "complimentary holiday" được xếp nằm cực kỳ sát vách với từ "free prize" trong không gian véc-tơ ngữ nghĩa ẩn.

## 4. Phân tích 50 case khó

• Mô hình tụt giảm nhiều nhất: Các mô hình dựa trên đếm từ BoW + LogReg và BoW + MLP bị sụt giảm thê thảm nhất (điểm F1 từ mức trên 0.95 ở tập thường rơi tự do xuống chỉ còn 0.54 và 0.56 trên tập ca khó).
• Nhóm case gây lỗi nặng nhất: Nhóm Thao túng / lừa đảo (Ngụy trang chữ).
2 Case cụ thể và cơ chế gây sai:
Case 1 (Thật=Spam | Đoán=Ham): "Money i have won wining number 946 wot do i do next"
Case 2 (Thật=Ham | Đoán=Spam): "Dear, will call Tmorrow.pls accomodate."
## 5. Thử thách cá nhân

Đề 1: Đổi ngưỡng quyết định 0.5 → 0.2 và → 0.8 cho mô hình Phần 1. Báo cáo chỉ số và tư vấn giải pháp cho ngân hàng bảo vệ tin nhắn OTP.
Giải thích, Tư vấn Ngân hàng: Ngân hàng sợ chặn nhầm OTP bắt buộc phải chọn ngưỡng cao 0.8. Vì tin nhắn OTP quyết định giao dịch của khách, lỗi bắt oan (False Positive) là không được phép xảy ra. Ngưỡng cao giúp mô hình cực kỳ thận trọng, triệt tiêu lỗi chặn nhầm, bảo vệ trải nghiệm khách hàng tối đa.
## 6. Phương án cho sản phẩm thật

Phương án lựa chọn: Encoder (MiniLM) + Logistic Regression.
## 7. AI log và điều đã học

Hỏi nguyên nhân lỗi RateLimitError 429 và NotFoundError 404 của Gemini API → Nhận được giải thích do hạn mức Free Tier quá tải và sai cấu trúc URL → Thực hiện viết code chèn hàm nghỉ giãn cách time.sleep(4.0) để giữ nhịp băng thông an toàn.

Điều hiểu ra:
- Hiểu sâu sắc toán học ngầm của thuật toán Logistic Regression: Quy tắc "Trừ của trừ hóa thành cộng" trong dòng cập nhật trọng số chính là cơ chế tự sửa sai tăng điểm độc hại cho từ khóa ở vòng sau.
- Nhận ra điểm yếu chí mạng của phương pháp túi từ Bag-of-Words: Việc chỉ đếm từ riêng lẻ mà bẻ gãy cấu trúc ngữ cảnh khiến mô hình hoàn toàn bất lực và dễ dàng bị đánh lừa bởi các kỹ thuật ngụy trang chữ hoặc nhồi từ khóa an toàn.

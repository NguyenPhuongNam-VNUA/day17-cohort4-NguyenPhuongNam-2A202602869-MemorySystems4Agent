# Triển khai Hoàn thiện: Memory Systems for AI Agent (Phase 2, Track 3, Day 17)

Dự án này triển khai và so sánh chuyên sâu hệ thống bộ nhớ đa tầng (Multi-tier Memory Architecture) dành cho AI Agent, bao gồm:
1. **Baseline Agent**: Bộ nhớ ngắn hạn nội phiên (Within-session Short-term Memory theo `thread_id`).
2. **Advanced Agent**: Kết hợp 3 tầng bộ nhớ:
   - **Short-term Memory**: Theo dõi ngữ cảnh lượt hội thoại gần nhất.
   - **Persistent Memory (`User.md`)**: Lưu trữ hồ sơ người dùng bền vững xuyên suốt mọi phiên làm việc (Cross-session).
   - **Compact Memory**: Tự động kích hoạt nén lịch sử hội thoại dài khi vượt ngưỡng token, giúp tối ưu chi phí ngữ cảnh prompt.

---

## 1. Cấu trúc thư mục mã nguồn

```
src/
├── model_provider.py    # Factory hỗ trợ 6 providers: openai, custom, gemini, anthropic, ollama, openrouter
├── config.py            # Cấu hình LabConfig (đường dẫn, threshold compact, model params)
├── memory_store.py      # Lõi UserProfileStore, CompactMemoryManager, ước lượng token, trích xuất fact & conflict resolution
├── agent_baseline.py    # Triển khai BaselineAgent (ngây thơ, quên fact ở thread mới, tích lũy prompt thô)
├── agent_advanced.py    # Triển khai AdvancedAgent (3 tầng memory, offline & live routing)
├── benchmark.py         # Suite benchmark so sánh Standard (10 sessions) & Long-context Stress (16 long turns)
└── test_agents.py       # Bộ kiểm thử pytest 6 test cases kiểm chứng toàn diện hành vi memory
```

---

## 2. Hướng dẫn chạy thử nghiệm

### 2.1. Chạy bài kiểm thử (Unit Tests)

```bash
pytest src/test_agents.py -v
```

Kiểm tra:
- `test_user_markdown_read_write_edit`: Kiểm tra CRUD trên `User.md` (read, write, search/replace, structured upsert).
- `test_compact_trigger`: Kiểm tra cơ chế tự động nén lịch sử cũ khi vượt ngưỡng token.
- `test_cross_session_recall`: Kiểm chứng Advanced nhớ thông tin ở phiên mới còn Baseline quên hoàn toàn.
- `test_compact_reduces_prompt_load_on_long_thread`: Kiểm chứng compact memory giảm tải prompt load trên hội thoại dài.
- `test_conflict_resolution_and_noise_filtering` (Bonus): Kiểm tra cập nhật đính chính (Đà Nẵng -> Huế -> Đà Nẵng, backend -> MLOps) và lọc bỏ thông tin gây nhiễu (Hà Nội, đùa product manager).
- `test_confidence_threshold_ignores_inquiries` (Bonus): Đảm bảo các câu hỏi dò nhớ không gây ô nhiễm hồ sơ `User.md`.

### 2.2. Chạy Benchmark

```bash
python src/benchmark.py
```

---

## 3. Bảng kết quả Benchmark thực nghiệm

### 3.1. Standard Benchmark (`data/conversations.json` - 10 hội thoại, user `dungct`)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| **Baseline Agent** | 3,663 | 27,929 | 0.0% | 15.0% | 0 B | 0 |
| **Advanced Agent** | 3,127 | 38,078 | **100.0%** | **90.0%** | 454 B | 0 |

### 3.2. Long-Context Stress Benchmark (`data/advanced_long_context.json` - 16 lượt dài, user `dungct_stress`)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| **Baseline Agent** | 628 | 26,293 | 0.0% | 15.0% | 0 B | 0 |
| **Advanced Agent** | 897 | **12,321** | **100.0%** | **100.0%** | 371 B | **23** |

---

## 4. Phân tích Chuyên sâu Trade-off & Rủi ro Hệ thống

### 4.1. Vì sao Advanced Agent có khả năng nhớ chéo phiên (Recall) vượt trội?
- **Baseline Agent** gắn chặt state với `thread_id`. Khi một thread mới bắt đầu, phiên làm việc hoàn toàn trống rỗng, khiến khả năng trả lời các câu hỏi về tên, sở thích, nơi ở trước đó đạt 0.0%.
- **Advanced Agent** sử dụng `UserProfileStore` tách biệt persistent memory ra file vật lý `state/profiles/<user_id>/User.md`. Mọi câu hỏi ở thread mới đều được nạp thông tin cốt lõi từ `User.md`, đem lại độ chính xác recall tuyệt đối (100.0%).

### 4.2. Vì sao ở hội thoại ngắn, Advanced Agent có chi phí ngữ cảnh (Prompt Tokens) cao hơn?
- Trong hội thoại ngắn (khoảng 10 lượt ngắn như `conversations.json`), tổng lượng token của cuộc hội thoại chưa vượt quá ngưỡng compact (`compact_threshold_tokens = 650`).
- Advanced Agent phải chịu **chi phí cố định (overhead)** từ việc nạp `User.md` (khoảng 60-100 tokens) và system prompt chỉ dẫn memory trong mỗi lượt gọi, làm tổng prompt tokens đạt 38,078 so với 27,929 của Baseline. Đây là đánh đổi cần thiết để duy trì tính nhất quán lâu dài.

### 4.3. Vì sao Compact Memory đem lại lợi thế quyết định ở hội thoại dài?
- Trong Long-Context Stress Benchmark (các lượt hội thoại dài, dày đặc chi tiết tin tức Artemis III, X-59, WMO, BC Energy):
  - **Baseline** giữ nguyên toàn bộ lịch sử thô, khiến chi phí prompt token tăng theo cấp số cộng tích lũy qua từng lượt, đạt **26,293 prompt tokens**.
  - **Advanced Agent** tự động kích hoạt nén lịch sử (23 lần compaction). Các đoạn hội thoại cũ được cô đọng thành mục tóm tắt súc tích, chỉ giữ lại các lượt gần nhất (`compact_keep_messages`).
  - Kết quả: Prompt tokens của Advanced Agent chỉ còn **12,321 tokens**, giúp **tiết kiệm tới 53.1% chi phí prompt context** mà vẫn đạt 100.0% recall.

### 4.4. Đánh giá tốc độ phình to bộ nhớ (Memory Growth) và Rủi ro
- File `User.md` chỉ tăng từ 0 B lên khoảng 370 - 454 Bytes qua hàng chục lượt tương tác nhờ cấu trúc Markdown Key-Value gọn gàng.
- **Rủi ro đi kèm:**
  1. *Ảo tưởng bộ nhớ (Memory Hallucination / Fact Corruption):* Nếu không có bộ lọc tự động, các câu đùa hoặc thông tin tạm thời (như "Hà Nội chỉ đi họp 2 ngày", "đùa chuyển sang làm product manager") sẽ bị ghi nhận thành sự thật vĩnh viễn.
  2. *Xung đột thông tin (Information Contradiction):* Khi người dùng đổi nơi ở (Đà Nẵng -> Huế -> Đà Nẵng) hoặc chuyển nghề (backend -> MLOps), nếu lưu song song cả hai sẽ khiến LLM bối rối.
  3. *Mất chi tiết khi tóm tắt (Information Loss in Compaction):* Tóm tắt nén quá mức có thể làm rơi rụng các số liệu kỹ thuật hoặc điều kiện biên quan trọng.

---

## 5. Các tính năng Mở rộng (Bonus Features - 90-100 Điểm)

1. **Conflict Resolution (Giải quyết xung đột thông tin):**
   - Khi người dùng đưa ra bản đính chính mới (ví dụ: chuyển từ Đà Nẵng sang Huế rồi lại sang Đà Nẵng, chuyển từ backend sang MLOps), hệ thống tự động ghi đè và cập nhật fact mới nhất, loại bỏ hoàn toàn fact cũ sai lệch.
2. **Noise Filtering (Lọc nhiễu thông tin):**
   - Nhận diện các ngữ cảnh tạm thời hoặc câu nói đùa để không đưa vào `User.md` (ví dụ: các chuyến công tác ngắn ngày, câu đùa về nghề nghiệp).
3. **Confidence Threshold & Inquiry Filtering (Ngưỡng tin cậy & Bỏ qua câu hỏi):**
   - Phân biệt giữa câu tự giới thiệu của người dùng và câu hỏi kiểm tra recall ("Mình tên gì?", "Bạn biết DũngCT không?"). Không trích xuất câu hỏi làm dữ liệu hồ sơ.
4. **Structured Entity Extraction (Trích xuất thực thể có cấu trúc):**
   - Định dạng dữ liệu nhất quán theo các trường `Tên`, `Nơi ở`, `Nghề nghiệp`, `Đồ uống yêu thích`, `Món ăn yêu thích`, `Thú cưng`, `Phong cách trả lời`, `Mối quan tâm`.

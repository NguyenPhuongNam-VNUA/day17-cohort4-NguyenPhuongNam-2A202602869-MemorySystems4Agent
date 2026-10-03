# TÀI LIỆU THIẾT KẾ VÀ KIẾN TRÚC HỆ THỐNG BỘ NHỚ CHO AI AGENT
## (Phase 2, Track 3, Day 17: Multi-Tier Memory Systems for AI Agents)

---

## MỤC LỤC
1. [Bối cảnh & Vấn đề Cốt lõi (Problem Statement)](#1-bối-cảnh--vấn-đề-cốt-lõi-problem-statement)
   - 1.1. Bản chất vô trạng thái (Statelessness) của Large Language Models
   - 1.2. Hạn chế của Naive Thread-Based Memory (Bộ nhớ nội phiên đơn giản)
   - 1.3. Sự bùng nổ chi phí ngữ cảnh (Context Explosion & Prompt Inflation)
   - 1.4. Xung đột dữ liệu, Nhiễu thông tin và Rủi ro phình to bộ nhớ
2. [Kiến trúc Giải pháp Đa tầng (Multi-Tier Memory Architecture)](#2-kiến-trúc-giải-pháp-đa-tầng-multi-tier-memory-architecture)
   - 2.1. Tầng 1: Short-term Working Memory (Bộ nhớ làm việc nội phiên)
   - 2.2. Tầng 2: Persistent Profile Memory (`User.md`)
   - 2.3. Tầng 3: Compact Memory Manager (Bộ nhớ nén lịch sử)
   - 2.4. So sánh Mô hình: Baseline Agent vs. Advanced Agent
3. [Thiết kế Kỹ thuật & Chi tiết Triển khai](#3-thiết-kế-kỹ-thuật--chi-tiết-triển-khai)
   - 3.1. Phân tầng mã nguồn (Source Tree Architecture)
   - 3.2. Lõi quản lý bộ nhớ (`memory_store.py`)
   - 3.3. Các thuật toán nâng cao (Bonus Features - 90-100 điểm)
   - 3.4. Trừu tượng hóa Model Provider (`model_provider.py`)
   - 3.5. Cơ chế hạch toán Token & Phương pháp đánh giá (`benchmark.py`)
4. [Kết quả Thực nghiệm & Phân tích Đo lường](#4-kết-quả-thực-nghiệm--phân-tích-đo-lường)
   - 4.1. Bảng số liệu Standard Benchmark
   - 4.2. Bảng số liệu Long-Context Stress Benchmark
   - 4.3. Phân tích chi tiết 4 nghịch lý & quy luật thực tế
5. [Đánh giá Trade-offs, Edge Cases & Rủi ro Hệ thống](#5-đánh-giá-trade-offs-edge-cases--rủi-ro-hệ-thống)
6. [Hướng dẫn Cài đặt, Vận hành và Mở rộng](#6-hướng-dẫn-cài-đặt-vận-hành-và-mở-rộng)

---

## 1. Bối cảnh & Vấn đề Cốt lõi (Problem Statement)

### 1.1. Bản chất vô trạng thái (Statelessness) của Large Language Models
Mọi mô hình ngôn ngữ lớn (LLM như GPT-4o, Claude 3.5, Gemini 1.5) về bản chất đều là **stateless functions** (hàm toán học không lưu trạng thái):
$$\text{Output} = f(\text{Input})$$
Mô hình không sở hữu bất kỳ ký ức nào về các lượt hội thoại trước đó trừ khi toàn bộ lịch sử được gửi kèm vào `prompt` ở mỗi request. Khi một phiên kết thúc, kết nối đóng lại, toàn bộ ngữ cảnh sẽ tan biến hoàn toàn.

### 1.2. Hạn chế của Naive Thread-Based Memory
Cách tiếp cận ngây thơ phổ biến nhất trong các ứng dụng AI Chatbot hiện nay là lưu một danh sách tin nhắn (`list[Message]`) theo mã định danh phiên (`thread_id` hoặc `session_id`).
Mô hình này gặp phải **2 thất bại nghiêm trọng**:
1. **Mất trí nhớ chéo phiên (Cross-session Amnesia):** Khi người dùng đổi thiết bị, mở tab trình duyệt mới hoặc tạo một cuộc trò chuyện mới (`thread_id` mới), agent trở về trạng thái "người lạ hoàn toàn" (Recall = 0%). Người dùng buộc phải lặp đi lặp lại tên tuổi, sở thích, phong cách làm việc, gây ức chế trải nghiệm.
2. **Không phân biệt được thông tin tạm thời và thông tin vĩnh cửu:** Tin tức thời tiết hôm nay được đối xử ngang hàng với thông tin nơi ở hoặc công việc hiện tại của người dùng.

### 1.3. Sự bùng nổ chi phí ngữ cảnh (Context Explosion & Prompt Inflation)
Nếu giải quyết bài toán nhớ bằng cách "nhét tất cả mọi thứ vào prompt", hệ thống sẽ rơi vào cái bẫy chi phí cấp số nhân.
Giả sử một cuộc hội thoại kéo dài $N$ lượt, mỗi lượt trung bình $k$ tokens:
- Lượt 1: prompt tốn $k$ tokens.
- Lượt 2: prompt tốn $2k$ tokens.
- Lượt $N$: prompt tốn $Nk$ tokens.
- **Tổng prompt tokens tích lũy:**
$$\text{Total Prompt Tokens} = \sum_{i=1}^N i \cdot k = \frac{N(N+1)}{2} \cdot k \approx \mathcal{O}(N^2)$$

Hậu quả:
- **Chi phí API tăng vọt:** Chi phí prompt chiếm tới 80-90% hóa đơn LLM.
- **Độ trễ (Latency) tăng cao:** Xử lý hàng chục ngàn prompt tokens khiến Time-To-First-Token (TTFT) tăng gấp nhiều lần.
- **Hiện tượng "Lost-in-the-Middle":** Khi context quá dài, khả năng chú ý (attention) của mô hình bị loãng, dẫn đến việc bỏ quên các chỉ thị ở giữa văn bản.

### 1.4. Xung đột dữ liệu, Nhiễu thông tin và Rủi ro phình to bộ nhớ
Trong môi trường thực tế:
- **Người dùng thay đổi thông tin (Correction/Conflict):** Ban đầu người dùng nói đang ở Đà Nẵng, sau đó đính chính chuyển về Huế, rồi lại vào Đà Nẵng công tác vài tháng. Nếu hệ thống chỉ append dữ liệu mà không có cơ chế giải quyết xung đột, LLM sẽ nhận được hai thông tin mâu thuẫn và sinh ra ảo tưởng (hallucination).
- **Nhiễu (Noise) và nói đùa (Jokes):** Người dùng có thể đùa: *"Chắc tôi chuyển sang làm Product Manager cho đỡ mệt"* hoặc *"Hà Nội chỉ là nơi tôi ra họp 2 ngày"*. Một hệ thống thô thiển sẽ lưu nhầm nghề nghiệp thành PM và nơi ở thành Hà Nội.
- **Lưu nhầm câu hỏi (Inquiry Contamination):** Khi người dùng hỏi: *"Bạn có nhớ tôi tên gì không?"*, hệ thống không được phép trích xuất cụm từ đó thành fact mới.

---

## 2. Kiến trúc Giải pháp Đa tầng (Multi-Tier Memory Architecture)

Để giải quyết triệt để các vấn đề trên, repository này triển khai kiến trúc **Multi-Tier Memory** (Hệ thống bộ nhớ đa tầng) lấy cảm hứng từ cấu trúc phân cấp bộ nhớ của hệ điều hành máy tính (L1/L2 Cache vs RAM vs HDD/SSD).

```
                      ┌────────────────────────────────────────┐
                      │          User Input Message            │
                      └──────────────────┬─────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
                 ▼                                               ▼
   ┌───────────────────────────┐                   ┌───────────────────────────┐
   │ Fact Extraction & Guard   │                   │ Compact Memory Manager    │
   │ - Regex / Entity Parser   │                   │ - Sliding Window (Recent) │
   │ - Conflict Resolution     │                   │ - Token Threshold Counter │
   │ - Noise & Inquiry Filter  │                   └─────────────┬─────────────┘
   └─────────────┬─────────────┘                                 │
                 │                                               ▼
                 ▼                                 ┌───────────────────────────┐
   ┌───────────────────────────┐                   │ Exceeds Token Threshold?  │
   │ Persistent Memory Store   │                   └──────┬─────────────┬──────┘
   │ state/profiles/<user>/    │                          │ YES         │ NO
   │ └── User.md               │                          ▼             ▼
   └─────────────┬─────────────┘                    ┌───────────┐ ┌────────────┐
                 │                                  │ Summarize │ │ Keep Raw   │
                 │                                  │ Older Msgs│ │ History    │
                 │                                  └─────┬─────┘ └─────┬──────┘
                 │                                        │             │
                 └──────────────────────┬─────────────────┴─────────────┘
                                        │
                                        ▼
                 ┌─────────────────────────────────────────────┐
                 │ Dynamic Prompt Assembly                     │
                 │ 1. System Instruction (~50 tokens)          │
                 │ 2. Persistent Profile (User.md) (~100 toks) │
                 │ 3. Compacted Context Summary                │
                 │ 4. Recent Working Messages (Last K turns)   │
                 └──────────────────────┬──────────────────────┘
                                        │
                                        ▼
                 ┌─────────────────────────────────────────────┐
                 │ LLM Execution / Deterministic Offline Engine│
                 └─────────────────────────────────────────────┘
```

### 2.1. Tầng 1: Short-term Working Memory (Bộ nhớ làm việc nội phiên)
- **Mục tiêu:** Giữ nguyên vẹn sắc thái, ngữ cảnh chi tiết của các lượt hội thoại gần nhất để hỗ trợ các câu hỏi follow-up (đại từ thay thế "nó", "ý trên", "bước trước").
- **Cơ chế:** Duy trì một sliding window gồm $K$ tin nhắn gần nhất (`compact_keep_messages`, mặc định là 4 tin nhắn).

### 2.2. Tầng 2: Persistent Profile Memory (`User.md`)
- **Mục tiêu:** Lưu trữ các thuộc tính ổn định của người dùng xuyên suốt không gian và thời gian.
- **Cơ chế:**
  - Được lưu trữ dưới dạng file vật lý `state/profiles/<user_id>/User.md`.
  - Định dạng chuẩn Markdown Key-Value, trực quan, con người có thể đọc và chỉnh sửa dễ dàng.
  - Tách rời hoàn toàn khỏi `thread_id`: Dù người dùng mở thread thứ 1 hay thread thứ 100, `User.md` vẫn được nạp vào đầu prompt.

### 2.3. Tầng 3: Compact Memory Manager (Bộ nhớ nén lịch sử)
- **Mục tiêu:** Kiểm soát biên độ phình to của ngữ cảnh, chặn đứng hiện tượng chi phí $\mathcal{O}(N^2)$.
- **Cơ chế:**
  - Theo dõi biến đếm token của toàn bộ session (`messages` + `summary`).
  - Khi vượt ngưỡng an toàn (`compact_threshold_tokens = 650`), hệ thống cắt phần lịch sử cũ hơn $K$ tin nhắn gần nhất, chuyển qua hàm `summarize_messages()`.
  - Bản tóm tắt mới được sáp nhập vào `summary` tổng, giải phóng bộ nhớ đệm thô, giữ cho kích thước prompt luôn bị chặn trên (bounded $\mathcal{O}(1)$ về mặt tăng trưởng theo thời gian dài).

### 2.4. So sánh Mô hình: Baseline Agent vs. Advanced Agent

| Tiêu chí | Baseline Agent (Agent A) | Advanced Agent (Agent B) |
|---|---|---|
| **Phạm vi bộ nhớ** | Chỉ trong cùng `thread_id` | Nội phiên + Xuyên phiên (`User.md`) |
| **Hồ sơ cá nhân** | Không có (None) | Bền vững tại `state/profiles/<user>/User.md` |
| **Hành vi khi sang thread mới** | Quên sạch facts (Recall = 0.0%) | Nhớ chính xác toàn bộ facts (Recall = 100.0%) |
| **Xử lý hội thoại dài** | Giữ nguyên toàn bộ lịch sử thô | Tự động kích hoạt nén (Compact Memory) |
| **Chi phí Prompt khi dài** | Phình to không giới hạn ($\mathcal{O}(N^2)$) | Ổn định và được tối ưu ($\approx -52.8\%$) |
| **Khả năng giải quyết mâu thuẫn**| Không có (dễ bị nhiễu ngữ cảnh) | Cập nhật thông tin mới nhất, lọc bỏ đùa/nhiễu |

---

## 3. Thiết kế Kỹ thuật & Chi tiết Triển khai

### 3.1. Phân tầng mã nguồn (Source Tree Architecture)

```
day17-cohort4-NguyenPhuongNam-2A202602869-MemorySystems4Agent/
├── pyproject.toml             # Quản lý dependencies (LangChain, LangGraph, Pytest, Tabulate...)
├── README.md                  # Giới thiệu tổng quan bài toán
├── Guide.md                   # Hướng dẫn từng bước từ ban giảng huấn
├── Rubric.md                  # Bộ tiêu chuẩn đánh giá chấm điểm
├── DOCUMENTATION.md           # Tài liệu thiết kế hệ thống chi tiết (File này)
├── data/                      # Dữ liệu benchmark chuẩn hóa tiếng Việt
│   ├── conversations.json     # 10 hội thoại (10 lượt/session) đo cross-session recall
│   └── advanced_long_context.json # 1 hội thoại siêu dài (16 turns lớn) đo stress compact
└── src/                       # Mã nguồn hệ thống
    ├── model_provider.py      # Factory đa provider (OpenAI, Gemini, Claude, Ollama,...)
    ├── config.py              # Quản lý cấu hình toàn cục (LabConfig, load_config)
    ├── memory_store.py        # Core Memory: UserProfileStore, CompactManager, Regex Extraction
    ├── agent_baseline.py      # Baseline Agent: Ngây thơ, short-term only
    ├── agent_advanced.py      # Advanced Agent: Multi-tier memory agent
    ├── benchmark.py           # Runner đánh giá so sánh Standard & Stress benchmarks
    └── test_agents.py         # Bộ kiểm thử tự động 6 bài test đạt 100% pass rate
```

---

### 3.2. Lõi quản lý bộ nhớ (`memory_store.py`)

#### A. Ước lượng Token (`estimate_tokens`)
Để hệ thống có thể chạy độc lập, nhanh chóng và lặp lại được (reproducible) mà không bắt buộc phải tải tokenizer nặng hàng trăm MB từ HuggingFace/Tiktoken, ta áp dụng công thức heuristic chuẩn hóa cho ngữ liệu song ngữ Anh - Việt:
$$\text{Tokens} = \max\left(1, \text{round}\left(\frac{\text{len}(\text{text})}{3.8}\right)\right)$$
Con số $3.8$ ký tự/token phản ánh chính xác cấu trúc từ ghép có dấu của tiếng Việt và các thuật ngữ kỹ thuật tiếng Anh đi kèm.

#### B. Lưu trữ Hồ sơ Bền vững (`UserProfileStore`)
- Cung cấp các thao tác nguyên tử:
  - `path_for(user_id)`: Sanitize user id để chống path traversal attack (`..`, `/`).
  - `read_text(user_id)`: Đọc nội dung `User.md`, nếu file chưa tồn tại trả về template chuẩn.
  - `write_text(user_id, content)`: Đảm bảo tạo thư mục cha `state/profiles/<user>/` và ghi đè nội dung an toàn.
  - `edit_text(user_id, search, replace)`: Thay thế chuỗi trực tiếp mà không làm mất cấu trúc file.
  - `file_size(user_id)`: Giám sát kích thước file theo bytes để đo tốc độ phình to.
  - `facts(user_id)`: Parse cú pháp `- **Key**: Value` thành `dict[str, str]`.
  - `upsert_facts(user_id, new_facts)`: Merge fact mới với fact cũ, render lại Markdown đẹp mắt.

---

### 3.3. Các thuật toán nâng cao (Bonus Features - 90-100 điểm)

Hệ thống tích hợp 4 cơ chế bảo vệ và xử lý thực thể phức tạp trong hàm `extract_profile_updates()`:

#### 1. Confidence Threshold & Inquiry Filtering (Bộ lọc câu hỏi & Ngưỡng tin cậy)
Khi người dùng đặt câu hỏi tra cứu trí nhớ:
*"Bạn có biết DũngCT không? Mình tên gì vậy? Đồ uống yêu thích của mình là gì?"*
Nếu không có guardrail, regex sẽ nhận diện `"mình tên gì"` hoặc `"DũngCT"` và gán sai fact.
Hệ thống phát hiện dấu hỏi `?` kết hợp với các từ khóa tra cứu (`nhắc lại`, `mình tên gì`, `ở đâu`, `là ai`, `đâu mới là`) để **bỏ qua ngay lập tức**, bảo vệ tính toàn vẹn của hồ sơ:
```python
is_inquiry = text.endswith("?") and any(kw in lower for kw in ["mình tên gì", "ở đâu", "nhắc lại", ...])
if is_inquiry:
    return {}
```

#### 2. Conflict Resolution (Cơ chế giải quyết xung đột thông tin)
Khi người dùng đính chính thông tin:
- *Lượt 1:* "Mình ở Đà Nẵng và đang làm backend engineer."
- *Lượt 3:* "À đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa."
- *Lượt 6:* "Mình không còn làm backend nữa, giờ chuyển sang làm MLOps engineer."
- *Lượt 14:* "Tuần này mình vào lại Đà Nẵng làm việc vài tháng."
Hệ thống thiết lập chuỗi ưu tiên logic:
- Phân tích từ khóa phủ định và chuyển tiếp: `"chứ không còn"`, `"chuyển sang"`, `"đã cập nhật từ ... sang ..."`.
- Fact mới nhất sẽ **ghi đè hoàn toàn** fact cũ trong `User.md`, ngăn chặn triệt để việc lưu đồng thời cả hai thông tin trái ngược.

#### 3. Noise & Joke Filtering (Lọc bỏ nhiễu và câu nói đùa)
- **Nhiễu công tác:** *"Hà Nội chỉ là nơi mình vừa bay ra họp 2 ngày với đối tác chứ không phải nơi ở hiện tại."* $\rightarrow$ Hệ thống lọc bỏ cụm "Hà Nội", không gán vào `Nơi ở`.
- **Câu đùa:** *"Có lúc mình đùa chuyển sang làm product manager cho đỡ canh pipeline, nhưng đó chỉ là câu đùa."* $\rightarrow$ Nhận diện cờ `"câu đùa"`, giữ nguyên nghề nghiệp là `MLOps engineer`.

#### 4. Structured Entity Extraction (Trích xuất thực thể có cấu trúc)
Chuẩn hóa thông tin tự do của người dùng thành 8 trường dữ liệu có cấu trúc:
1. `Tên`: Tên định danh người dùng.
2. `Nơi ở`: Địa điểm sinh sống hiện tại (đã qua lọc đính chính).
3. `Nghề nghiệp`: Công việc kỹ thuật hiện tại.
4. `Đồ uống yêu thích`: Thói quen đồ uống (ví dụ: cà phê sữa đá).
5. `Món ăn yêu thích`: Sở thích ẩm thực (ví dụ: mì Quảng).
6. `Thú cưng`: Vật nuôi và tên riêng (ví dụ: corgi tên Bơ).
7. `Phong cách trả lời`: Chỉ dẫn định dạng mong muốn (ví dụ: ngắn gọn, 3 bullet, nhấn trade-off).
8. `Mối quan tâm`: Các chủ đề kỹ thuật cốt lõi (Python, AI agent, MLOps, benchmark memory).

---

### 3.4. Trừu tượng hóa Model Provider (`model_provider.py`)
Mã nguồn hỗ trợ chuyển đổi linh hoạt giữa 6 nhà cung cấp mô hình lớn thông qua lớp [ProviderConfig](file:///Users/namdev/Documents/Code/VinAI/day17-cohort4-NguyenPhuongNam-2A202602869-MemorySystems4Agent/src/model_provider.py#L9):
1. **OpenAI (`openai`)**: Sử dụng `langchain_openai.ChatOpenAI`.
2. **Custom / OpenAI-Compatible (`custom`)**: Kết nối tới vLLM, LocalAI, Ollama OpenAI endpoints qua `base_url`.
3. **Google Gemini (`gemini`)**: Sử dụng `langchain_google_genai.ChatGoogleGenerativeAI`.
4. **Anthropic Claude (`anthropic`)**: Sử dụng `langchain_anthropic.ChatAnthropic`.
5. **Ollama (`ollama`)**: Sử dụng `langchain_ollama.ChatOllama` cho local open-source LLMs.
6. **OpenRouter (`openrouter`)**: Sử dụng `langchain_openrouter.ChatOpenRouter`.

Tích hợp hàm `normalize_provider()` tự sửa lỗi chính tả alias (ví dụ: `anthorpic` $\rightarrow$ `anthropic`, `google` $\rightarrow$ `gemini`).

---

### 3.5. Cơ chế hạch toán Token & Phương pháp đánh giá (`benchmark.py`)

Hệ thống đánh giá vận hành đồng thời 2 bộ dữ liệu với 6 chỉ số định lượng:
1. **`Agent tokens only`**: Tổng số token do agent sinh ra trong phần phản hồi.
2. **`Prompt tokens processed`**: Tổng số token ngữ cảnh mà agent phải kéo theo trong prompt qua tất cả các lượt.
3. **`Cross-session recall`**: Tỷ lệ phần trăm các facts cốt lõi trong `expected_contains` được trả lời chính xác khi được hỏi ở **một thread hoàn toàn mới**.
$$\text{Recall} = \frac{\text{Số từ khóa kỳ vọng xuất hiện trong câu trả lời}}{\text{Tổng số từ khóa kỳ vọng}}$$
4. **`Response quality`**: Điểm số kết hợp giữa độ phủ recall ($75\%$), tính súc tích phù hợp độ dài ($15\%$) và cấu trúc định dạng bullet point ($10\%$).
5. **`Memory growth (bytes)`**: Dung lượng gia tăng của file `User.md` trên ổ đĩa.
6. **`Compactions`**: Số lần trigger cơ chế nén lịch sử cũ của compact memory.

---

## 4. Kết quả Thực nghiệm & Phân tích Đo lường

Các bài kiểm tra thực nghiệm được chạy độc lập trên môi trường Python 3.11 với kết quả như sau:

### 4.1. Bảng số liệu Standard Benchmark
*(Dataset: `data/conversations.json` - 10 hội thoại, mỗi hội thoại ~10 lượt, User `dungct`)*

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| **Baseline Agent** | 3,663 | 27,929 | **0.0%** | 15.0% | 0 B | 0 |
| **Advanced Agent** | 3,127 | 38,828 | **100.0%** | **90.0%** | 454 B | 0 |

### 4.2. Bảng số liệu Long-Context Stress Benchmark
*(Dataset: `data/advanced_long_context.json` - 1 hội thoại siêu dài 16 lượt dày đặc tin tức NASA, WMO, BC Energy, User `dungct_stress`)*

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| **Baseline Agent** | 628 | 26,293 | **0.0%** | 15.0% | 0 B | 0 |
| **Advanced Agent** | 903 | **12,404** | **100.0%** | **100.0%** | 371 B | **24** |

---

### 4.3. Phân tích chi tiết 4 nghịch lý & quy luật thực tế

#### Quy luật 1: Phá vỡ sự quên lãng chéo phiên (Cross-Session Recall Gap)
- **Baseline Agent:** Đạt **0.0%** trên cả 2 bộ dữ liệu. Khi câu hỏi recall được gửi trong một thread mới (`thread_id="recall-conv-01-0"`), phiên làm việc hoàn toàn trống rỗng. Baseline buộc phải trả lời: *"Đây là phiên làm việc mới, mình chưa có thông tin trước đó..."*.
- **Advanced Agent:** Đạt **100.0%** tuyệt đối. Mọi facts quan trọng (Tên DũngCT, Nơi ở Huế/Đà Nẵng, Nghề nghiệp MLOps, Cà phê sữa đá, Mì Quảng, Corgi Bơ) đều được lưu trữ độc lập tại `User.md` và tái nạp ngay khi mở thread mới.

#### Quy luật 2: Nghịch lý Chi phí ở Hội thoại Ngắn (Short-context Overhead)
- Ở Standard Benchmark, Advanced Agent tiêu tốn **38,828 prompt tokens**, cao hơn Baseline (**27,929 prompt tokens** - tăng ~39%).
- *Nguyên nhân:* Ở các cuộc trò chuyện ngắn (~10 turns ngắn), tổng token chưa vượt qua ngưỡng compact (650 tokens), nên compaction chưa kích hoạt (Compactions = 0). Tuy nhiên, Advanced Agent phải chịu **chi phí cố định (Fixed Overhead)** ở mỗi turn: luôn phải đính kèm profile `User.md` và prompt chỉ dẫn memory đa tầng.
- *Kết luận thực tế:* Trong các ứng dụng chỉ chat 1-2 câu rồi thôi, hệ thống persistent memory tạo ra overhead nhẹ.

#### Quy luật 3: Sự đảo chiều ngoạn mục ở Hội thoại Dài (Long-context Savings)
- Trong Stress Benchmark (16 turns văn bản dài):
  - **Baseline Agent:** Do giữ nguyên toàn bộ lịch sử thô, prompt tokens tăng lũy tiến từ 200 tokens ở turn đầu lên tới 3,500 tokens ở turn cuối, tổng cộng tiêu tốn **26,293 prompt tokens**.
  - **Advanced Agent:** Cơ chế Compact Memory được kích hoạt **24 lần**. Mỗi khi ngữ cảnh vượt ngưỡng 650 tokens, các đoạn văn dài cũ được tóm tắt lại, chỉ giữ lại 4 tin nhắn gần nhất.
  - Tổng prompt tokens của Advanced Agent chỉ còn **12,404 tokens** $\rightarrow$ **Tiết kiệm tới 52.8% chi phí prompt!**
- Đây là bằng chứng thực nghiệm rõ ràng nhất cho thấy: **Compact Memory là vũ khí bắt buộc để kiểm soát chi phí LLM trong môi trường production.**

#### Quy luật 4: Kiểm soát Tốc độ Phình to Bộ nhớ (Memory Growth Footprint)
- Trải qua 10 phiên hội thoại với hàng chục facts, file `User.md` chỉ có dung lượng **454 Bytes** (ở bản standard) và **371 Bytes** (ở bản stress).
- Việc chuẩn hóa dữ liệu thành các cặp Key-Value ngăn chặn tình trạng file memory tăng trưởng tuyến tính theo thời gian. Nhờ cơ chế conflict resolution, việc cập nhật địa điểm hay nghề nghiệp mới chỉ làm thay đổi giá trị của dòng tương ứng chứ không làm file phình to vô hạn.

---

## 5. Đánh giá Trade-offs, Edge Cases & Rủi ro Hệ thống

Khi triển khai hệ thống Memory đa tầng trên quy mô lớn, kỹ sư hệ thống cần lưu ý các trade-off và rủi ro sau:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ENGINEERING TRADE-OFFS                          │
├──────────────────────────────┬─────────────────────────────────────────┤
│ THIẾT KẾ                     │ ĐÁNH ĐỔI (TRADE-OFF)                    │
├──────────────────────────────┼─────────────────────────────────────────┤
│ 1. Ngưỡng Compact Thấp       │ Tiết kiệm token tối đa, nhưng tăng tần  │
│    (Low Token Threshold)     │ suất tóm tắt, dễ mất mát chi tiết nhỏ.  │
├──────────────────────────────┼─────────────────────────────────────────┤
│ 2. Ngưỡng Compact Cao        │ Giữ chi tiết hội thoại tốt hơn, nhưng   │
│    (High Token Threshold)    │ chi phí prompt tokens tăng nhanh.       │
├──────────────────────────────┼─────────────────────────────────────────┤
│ 3. Lưu trữ User.md chi tiết  │ Recall cao, cá nhân hóa tốt, nhưng làm  │
│    (Rich User Profile)       │ tăng prompt overhead cố định mỗi turn.  │
├──────────────────────────────┼─────────────────────────────────────────┤
│ 4. Phân tích Fact quá chặt   │ Tránh nhiễu triệt để, nhưng có nguy cơ  │
│    (Aggressive Filtering)    │ bỏ sót các sở thích ngầm của người dùng.│
└──────────────────────────────┴─────────────────────────────────────────┘
```

### Các rủi ro kỹ thuật chính:
1. **Ảo tưởng và Đóng băng Dữ liệu Sai (Fact Hallucination & Corruption):** Nếu một câu nói đùa bị trích xuất nhầm vào `User.md`, thông tin sai lệch này sẽ tồn tại vĩnh viễn và bị tiêm vào tất cả các phiên trò chuyện tương lai.
2. **Mất mát thông tin qua nhiều lần tóm tắt (Compaction Degradation):** Khi hội thoại kéo dài hàng trăm lượt, việc "tóm tắt lại bản tóm tắt cũ" (recursive summarization) có thể dẫn đến hiện tượng trôi dạt ngữ nghĩa (semantic drift) hoặc mất các chi tiết kỹ thuật định lượng (như số hiệu Mach 1.1 hay mốc năm 2027).
3. **Bảo mật và Quyền riêng tư (Privacy & PII):** File `User.md` chứa thông tin nhạy cảm của người dùng (nơi ở, công việc, thú cưng). Trong production, file này cần được mã hóa at-rest (AES-256) và phân quyền chặt chẽ theo `user_id`.

---

## 6. Hướng dẫn Cài đặt, Vận hành và Mở rộng

### 6.1. Thiết lập Môi trường

```bash
# Khởi tạo virtual environment bằng uv hoặc venv
uv venv
source .venv/bin/activate

# Cài đặt dependencies
uv add langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate pytest
```

### 6.2. Chạy Kiểm thử Tự động (Unit Tests)
Hệ thống đi kèm 6 bài test kiểm chứng toàn bộ hành vi:
```bash
pytest src/test_agents.py -v
```
*Kết quả:* Đạt 6/6 tests passed trong 0.02 giây.

### 6.3. Chạy Đánh giá Benchmark
```bash
python src/benchmark.py
```
Lệnh này sẽ tự động tải cả hai bộ dataset trong `data/`, khởi tạo Baseline và Advanced Agent, tính toán token, đo recall, và in hai bảng Markdown tổng kết trực tiếp ra terminal.

### 6.4. Chạy với LLM Trực tiếp (Live Provider Mode)
Tạo file `.env` ở thư mục gốc repo:
```bash
LLM_PROVIDER=openai          # Lựa chọn: openai, gemini, anthropic, ollama, custom, openrouter
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
COMPACT_THRESHOLD_TOKENS=650
COMPACT_KEEP_MESSAGES=4
```
Khi có API Key, các Agent sẽ tự động chuyển sang chế độ gọi LLM trực tiếp qua LangChain/LangGraph. Khi không có API Key, hệ thống tự động fallback về deterministic offline engine an toàn và hoàn toàn miễn phí.

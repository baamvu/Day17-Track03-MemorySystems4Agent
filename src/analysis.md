# Phân tích kết quả Benchmark - Day 17: Memory Systems for AI Agent

## Kết quả Benchmark

### Standard Benchmark (10 conversations, 16 recall questions)

| Agent | Agent tokens | Prompt tokens | Recall | Quality | Memory (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 1985 | 16584 | 0.00 | 0.40 | 0 | 0 |
| Advanced | 2004 | 22223 | 0.64 | 0.77 | 278 | 0 |

### Long-Context Stress Benchmark (1 conversation, 16 turns dài, 3 recall questions)

| Agent | Agent tokens | Prompt tokens | Recall | Quality | Memory (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 352 | 22809 | 0.00 | 0.40 | 0 | 0 |
| Advanced | 263 | 19193 | 0.33 | 0.55 | 132 | 1 |

---

## 1. Vì sao Advanced có recall tốt hơn Baseline?

**Baseline = 0.00 recall, Advanced = 0.64 recall (standard)**

Baseline Agent chỉ có short-term memory trong cùng thread. Khi sang thread mới (recall thread), toàn bộ context trước đó bị mất. Baseline không có cơ chế nào để lưu trữ thông tin người dùng qua session.

Advanced Agent có `User.md` lưu trữ persistent memory. Khi người dùng cung cấp thông tin (tên, nghề nghiệp, nơi ở, sở thích), các fact được extract bằng regex và lưu vào file markdown. Khi recall ở thread mới, Advanced đọc lại `User.md` để trả lời.

Recall không đạt 1.0 vì:
- Confidence threshold yêu cầu fact xuất hiện >= 2 lần mới persist (tránh lưu sai)
- Một số recall questions đòi hỏi compound answer (nhiều fact cùng lúc) mà offline response logic không always cover hết
- Regex extraction có giới hạn với câu phức tạp

---

## 2. Vì sao Advanced tốn hơn ở hội thoại ngắn?

**Standard: Advanced = 22,223 prompt tokens vs Baseline = 16,584 prompt tokens (+34%)**

Ở hội thoại ngắn, Advanced mang theo nhiều ngữ cảnh hơn mỗi lượt:
- `User.md` content (profile text)
- Compact memory summary (nếu có)
- Recent messages

Baseline chỉ mang theo messages trong cùng thread, không có overhead từ profile.

Đây là **trade-off chính** của persistent memory: chi phí token tăng để đổi lấy khả năng recall qua session. Ở hội thoại ngắn (10 turns), overhead này chiếm ~34% tổng prompt tokens.

---

## 3. Vì sao Compact giúp Advanced có lợi thế ở hội thoại dài?

**Stress: Advanced = 19,193 prompt tokens vs Baseline = 22,809 prompt tokens (-16%)**

Khi hội thoại rất dài (16 turns với nội dung dài), Baseline phải mang theo toàn bộ history qua mỗi lượt. Không có cơ chế nén, prompt tokens tăng tuyến tính với số lượng message.

Advanced có `CompactMemoryManager` tự động nén old messages thành summary khi token vượt ngưỡng. Kết quả:
- 1 compaction xảy ra trong stress test
- Prompt tokens giảm 16% so với Baseline
- Thông tin cốt lõi (facts, recent context) vẫn được giữ lại trong summary

Compact chủ yếu tối ưu **prompt tokens processed** (ngữ cảnh đầu vào), không phải agent tokens (output). Đây là nơi tiết kiệm lớn nhất vì prompt tokens thường chiếm phần lớn chi phí trong hội thoại dài.

---

## 4. Memory growth và rủi ro

**Memory growth: 278 bytes (standard), 132 bytes (stress)**

File `User.md` tăng trưởng khi có thêm fact mới. Mỗi fact khoảng 30-50 bytes.

### Rủi ro:

1. **File phình to**: Nếu người dùng liên tục thêm fact mới mà không có cơ chế xóa old facts, file sẽ tăng vô hạn. Giải pháp: periodic cleanup, giới hạn số facts, hoặc archive old facts.

2. **Lưu sai fact**: Nếu extract_profile_updates match sai (ví dụ "mình thích Python" bị hiểu là drink), fact sai sẽ persist qua nhiều session. Confidence threshold giúp giảm rủi ro này (cần >= 2 lần match mới persist).

3. **Conflict khi correction**: Khi người dùng đính chính ("không còn ở Huế nữa, giờ ở Đà Nẵng"), nếu không detect correction keyword, cả 2 fact có thể cùng tồn tại. Conflict handling detect keywords "đính chính", "không còn", "chuyển sang" để resolve.

4. **Memory decay**: Thông tin cũ (tên, nghề nghiệp) có thể mất confidence theo thời gian nếu không được nhắc lại. Đây là feature hay rủi ro tùy ngữ cảnh: giúp quên thông tin sai, nhưng cũng có thể quên thông tin đúng.

---

## 5. Tóm tắt trade-off

| Metric | Baseline | Advanced | Đánh giá |
|---|---|---|---|
| Recall | 0.00 | 0.64 | Advanced nhớ qua session |
| Prompt cost (short) | 16,584 | 22,223 | Advanced tốn hơn 34% |
| Prompt cost (long) | 22,809 | 19,193 | Advanced tiết kiệm 16% |
| Memory growth | 0 bytes | 278 bytes | Chi phí lưu trữ |
| System complexity | Thấp | Cao | 3 lớp memory + confidence + decay |

**Kết luận**: Advanced mạnh hơn về recall và tiết kiệm ở hội thoại dài, nhưng phức tạp hơn và tốn hơn ở hội thoại ngắn. Lựa chọn phụ thuộc vào use case: nếu người dùng thường xuyên quay lại (multi-session), Advanced worth it. Nếu chỉ dùng one-shot, Baseline đủ.

---

## 6. Confidence Threshold - Ngăn ngừa lưu sai fact

### Vấn đề

Không có confidence threshold, bất kỳ lần mention nào cũng persist ngay lập tức. Điều này gây ra false facts:

- `"Hãy giải thích cho mình tên gọi MLOps"` → extract sai `name=gọi MLOps`
- `"Mình muốn hỏi về đồ uống yêu thích của người nổi tiếng"` → extract sai `drink=của người nổi tiếng`

### Kết quả đo lường

| Metric | Không có threshold | Có threshold (min=2) | Cải thiện |
|---|---|---|---|
| False facts persisted | 2 | 0 | **-100%** |
| True facts retained | 12 | 6 | -50% |

**Nhận xét**: Confidence threshold loại bỏ 100% false facts (2/2). Trade-off: true facts chỉ retain 50% vì mỗi fact cần xuất hiện >= 2 lần. Trong thực tế, người dùng thường nhắc lại thông tin quan trọng nhiều lần qua các session, nên true facts sẽ được persist dần.

### Cách hoạt động

```
Turn 1: "Mình tên là Dũng" → extract name=Dũng, occurrences=1 → KHÔNG persist
Turn 2: "Mình tên là Dũng" → occurrences=2 → PERSIST vào User.md
```

Giá trị `min_occurrences=2` là heuristic hợp lý: balance giữa tránh false positive và không quá khắt khe với true facts.

---

## 7. Memory Decay - Confidence giảm theo thời gian

### Vấn đề

Thông tin cũ có thể trở nên không chính xác (người dùng đổi nghề, chuyển nhà). Nếu fact không được nhắc lại, confidence nên giảm dần.

### Công thức

```
confidence = frequency_score × decay_factor
frequency_score = min(1.0, occurrences / 3.0)
decay_factor = 0.5 ^ (age_hours / half_life_hours)
```

### Kết quả mô phỏng (với occurrences=3, half_life=24h)

| Thời điểm | Confidence | Ý nghĩa |
|---|---|---|
| Hôm nay (seen 3 lần) | 1.00 | Chắc chắn đúng |
| Sau 24h không nhắc | 0.50 | Bớt chắc chắn |
| Sau 48h không nhắc | 0.25 | Cần xác minh lại |
| Sau 72h không nhắc | 0.125 | Gần như không tin cậy |

### Ưu điểm

- Fact đúng được nhắc lại nhiều lần → confidence luôn cao
- Fact sai chỉ mention 1 lần → confidence thấp, dễ bị loại bỏ
- Tự động adapt theo hành vi người dùng

### Rủi ro

- Fact quan trọng nhưng ít nhắc (ngày sinh, số điện thoại) có thể bị decay
- Cần threshold hợp lý để không quên thông tin cần thiết

---

## 8. Conflict Handling - Xử lý correction

### Vấn đề

Khi người dùng đính chính ("Mình không còn ở Huế nữa, giờ ở Đà Nẵng"), nếu không detect correction, cả 2 fact có thể cùng tồn tại trong User.md.

### Kết quả đo lường

| Metric | Giá trị |
|---|---|
| Correction detection accuracy | **93% (13/14)** |
| Test cases | 14 (7 corrections, 7 non-corrections) |

### Keywords detect correction

```python
_CORRECTION_KEYWORDS = [
    "đính chính", "không còn", "chuyển sang",
    "không phải", "thực ra", "chứ không"
]
```

### Cách hoạt động

```
1. User: "Mình đính chính, mình đang ở Đà Nẵng"
2. is_correction_message() → True
3. detect_conflict("location", "Đà Nẵng") → True (vì đã có "Huế")
4. resolve_conflict() → ghi đè "Huế" thành "Đà Nẵng"
5. User.md: - **location**: Đà Nẵng
```

### Trường hợp fail

- `"Đó là câu đùa thôi, nghề mình vẫn là MLOps"` — chứa "vẫn là" nhưng không có correction keyword rõ ràng. Giải pháp: thêm pattern matching cho "câu đùa", "thực chất", "thật ra".

---

## 9. Stress Test - Prompt Token Growth Curve

### Per-turn chi tiết (16 turns)

| Turn | Baseline | Advanced | Diff | Ghi chú |
|---|---|---|---|---|
| 0 | 186 | 199 | -13 | Gần bằng |
| 4 | 875 | 862 | +13 | Gần bằng |
| 8 | 1,499 | 1,454 | +45 | Advanced bắt đầu tiết kiệm |
| 11 | 1,966 | 1,893 | +73 | Chênh lệch tăng |
| 12 | 2,129 | 1,116 | +1,013 | **Compaction xảy ra** |
| 15 | 2,596 | 1,534 | +1,062 | Tiết kiệm rõ rệt |

### Tổng hợp

| Metric | Baseline | Advanced | Chênh lệch |
|---|---|---|---|
| First half (turns 0-7) | 6,267 | 6,161 | +106 (+1.7%) |
| Second half (turns 8-15) | 16,335 | 11,963 | +4,372 (+26.8%) |
| **Total** | **22,602** | **18,338** | **+4,264 (19.8%)** |
| Growth (first→second half) | +161% | +94% | -67 pts |

### Nhận xét

- **Compaction xảy ra ở turn 12**, ngay sau khi tổng token vượt ngưỡng 2000
- Sau compaction, mỗi turn tiết kiệm ~1,000 tokens so với Baseline
- Baseline prompt growth +161% (gấp 2.6 lần) vs Advanced chỉ +94% (gấp 1.9 lần)
- Compact chủ yếu tối ưu **prompt tokens processed** — phần chi phí lớn nhất trong hội thoại dài

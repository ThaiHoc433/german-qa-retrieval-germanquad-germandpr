# future_RAG — Nhánh sinh câu trả lời + trích dẫn (V2)

Thư mục này chứa thiết kế và implementation plan cho nhánh **V2**: thêm bộ trả lời sinh
câu (LLM) có trích dẫn nguồn, cạnh Reader trích xuất của V1, cộng một web app local để
kiểm thử trực quan.

- **Không thuộc cổng hoàn tất V1.** Module spec V1.1
  (`../docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md`) vẫn là
  chuẩn kỹ thuật cho V1 và **không bị sửa** bởi bất kỳ file nào trong thư mục này.
- Nhánh này phụ thuộc đầu ra của V1 (candidate top-20 đã khóa, corpus các tier, run infra),
  nên thực nghiệm chính chạy sau khi V1 qua Chặng 9. Phần không phụ thuộc (prompt, parser,
  bộ kiểm chứng trích dẫn, smoke test) làm được sớm trên fixture.

## Cấu trúc

Mô phỏng đúng quy ước `docs/superpowers/{specs,plans}/` của project chính, để tách biệt
tài liệu nhánh V2 khỏi V1 trong lúc còn brainstorm/chưa duyệt production:

```
future_RAG/
├── README.md                     (file này)
└── superpowers/
    ├── specs/
    │   └── 2026-09-12-rag-answer-generation-design.md   — design doc đã duyệt
    └── plans/
        └── 2026-09-12-rag-answer-generation-plan.md     — implementation plan
```

## Trạng thái

- Design doc: đã brainstorm và duyệt qua nhiều vòng (2026-09-12).
- Implementation plan: xem `superpowers/plans/`.
- Code: **chưa có**. Khi triển khai, code thật (`src/`, `app/`, `tests/`) nằm ở vị trí
  chuẩn trong repo, không nằm trong `future_RAG/` — thư mục này chỉ giữ tài liệu thiết
  kế/kế hoạch của nhánh V2.

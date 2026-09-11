# German Open-Domain Extractive QA — Website tĩnh

Trang giới thiệu tiếng Việt dành cho người đọc phổ thông và kỹ thuật. HTML semantic,
CSS, JavaScript thuần; không có framework, backend, API key hoặc bước build.

## Chạy local

Mở `index.html` bằng trình duyệt. Tất cả đường dẫn là tương đối; không dùng fetch,
CDN hoặc font ngoài nên trang hoạt động qua `file://` và khi offline.

Hoặc chạy từ thư mục `website` nếu đã cài Python:

```sh
python -m http.server 8000
```

Mở `http://localhost:8000`. Không cần npm để xem website.

## Deploy GitHub Pages

**Cách đơn giản: repo riêng cho website.** Đưa `index.html`, `styles.css`, `script.js`,
`README.md` và `.nojekyll` vào thư mục gốc repo, commit và push. Trên GitHub:

1. Settings → Pages → Build and deployment → Source: Deploy from a branch.
2. Chọn branch `main`, thư mục `/ (root)` rồi Save.
3. Đợi GitHub báo deploy thành công, mở URL Pages được cung cấp.

**Nếu giữ trong repo NLP:** dùng GitHub Actions để upload **chỉ thư mục `website`**
bằng `actions/upload-pages-artifact` rồi deploy bằng `actions/deploy-pages`.
Hoặc xuất riêng nội dung thư mục website sang root của branch `gh-pages` và chọn
branch đó trong Pages. Chế độ Deploy from a branch không cho chọn thư mục tùy ý
`/website` (chỉ root hoặc `/docs`). Không ghi đè thư mục docs đang chứa spec.

Không cần thêm base URL. Các anchor và asset tương đối hoạt động cả với project
Pages dạng `https://<owner>.github.io/<repo>/`. Website này chưa được push/deploy tự động.

Tài liệu GitHub: https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site

## Nguồn nội dung và trạng thái

- Nguồn: **Module Implementation Specs V1.1**, frozen **2026-09-11**, D1–D6/P1–P4.
- File đã đọc toàn bộ: `../docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md`.
- File được yêu cầu có hậu tố `(3)` không có trong workspace tại thời điểm xây dựng.
  File hiện có tự xác định là V1.1 và được dùng làm nguồn. Cần đối chiếu thêm nếu
  bản `(3)` ở nơi khác có nội dung khác.
- Không sửa, không nhúng toàn bộ Markdown nguồn vào trang. Chi tiết được biên tập
  lại, mỗi module chỉ mở rộng schema, quy tắc, cấu hình, kiểm thử và artifact liên quan.
- Base dự kiến khoảng 7,5K; mid tổng 50.000; large tổng 150.000. Base phải thay bằng
  số thật sau nghiệm thu build V1.1; mid/large là mốc thiết kế, chưa là thành quả.
- Không đưa metric benchmark cũ vào trang. Các phép tính rank/EM/F1 và ví dụ Einstein
  chỉ để giải thích, được gắn nhãn minh họa. Website không thực hiện inference.
- V2 hybrid được đề cập trong spec. Hard-negative mining/Haystack là hướng mở rộng
  theo brief website, không tuyên bố có chi tiết hay đã triển khai trong V1.1.

## Cập nhật khi spec đổi

1. Đọc bản mới và decision log; không sửa bản V1.1 frozen để phù hợp website.
2. Cập nhật version/ngày ở hero, fact band, technical section và README này.
3. Đối chiếu các mục: §1 schemas; §2 corpus/split/chunking; §3 metric, N_eval và
   buckets; §4–7 cấu hình/checkpoint; §8 latency/run; §9 quality gates.
4. Sửa nội dung trong `index.html`. Logic walkthrough nằm trong HTML; JavaScript
   chỉ đổi bước và đánh dấu menu. Kiểu hiển thị nằm trong `styles.css`.
5. Chỉ thêm kết quả sau khi có artifact được nghiệm thu. Ghi rõ run_id, dataset,
   split, fingerprint corpus, N_eval, thiết bị, n_latency và nguồn metric.
6. Kiểm tra 375px / 768px / 1440px, mọi details bằng Enter/Space, walkthrough,
   menu anchor, focus, reduced motion và chế độ JavaScript tắt. Đừng đưa lại các
   quy mô đã bị thay thế từ phiên bản cũ.

## Khả năng tiếp cận

Native `details/summary` dùng được bằng bàn phím và không cần JS. Walkthrough mặc
định hiển thị toàn bộ bước; JS chỉ tăng cường với Next/Previous và nút chọn bước.
Có skip link, focus rõ, sticky navigation, trạng thái bước qua aria-live, reduced
motion, SVG có tên truy cập và nhãn chữ song song với màu sắc. Font chính từ 16px.

## File

- `index.html`: toàn bộ nội dung đã chọn lọc, diagrams và accordions.
- `styles.css`: theme, responsive, print và reduced motion.
- `script.js`: walkthrough và active navigation.
- `.nojekyll`: bỏ xử lý Jekyll khi dùng Pages.
- `qa/`: bằng chứng kiểm tra và screenshots (nếu có), không cần để chạy trang.

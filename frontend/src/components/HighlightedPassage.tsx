import { splitByUtf16 } from "../lib/highlight";

/**
 * Đoạn nguồn với đáp án được tô sáng.
 * An toàn: nội dung corpus chỉ đi vào JSX như chuỗi (React tự thoát ký tự), không bao giờ chèn HTML thô.
 * `white-space: pre-wrap` (trong CSS) giữ xuống dòng của corpus (ví dụ "Tiêu đề\n\n=== Mục ===").
 */
export function HighlightedPassage({ text, startUtf16, endUtf16 }: { text: string; startUtf16: number; endUtf16: number }) {
  const p = splitByUtf16(text, startUtf16, endUtf16);
  return (
    <p className="passage" lang="de">
      {p.before}
      {p.highlighted && <mark data-testid="answer-mark">{p.answer}</mark>}
      {p.after}
    </p>
  );
}

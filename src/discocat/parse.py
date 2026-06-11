"""Wrapper lambeq parsers/readers để chuyển phrase → string diagram.

Support nhiều "reader" của lambeq:
    - bobcat   : BobcatParser, CCG đầy đủ, CẦN tải model 500 MB
                 (hiện URL gốc cambridgequantum đã chết, cần model local)
    - spiders  : SpidersReader, không cần tải, mọi từ là atomic noun.
                 Common QNLP baseline (Lorenz 2021).
    - cups     : CupsReader, cups composition đơn giản
    - linear   : LinearReader, sequential left-to-right

DisCoCat pipeline tổng quát:
    text → Reader → string diagram → (Rewriter nếu có) → final diagram

Lưu ý:
    - Rewriter (prepositional_phrase, determiner, ...) chỉ ý nghĩa với
      diagram CCG (Bobcat). Với spiders/cups/linear → bỏ qua rewriter.
    - Một số phrase có thể parse fail. Ghi vào failures, không raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from tqdm import tqdm


ReaderType = Literal["bobcat", "spiders", "cups", "linear"]


# Default rewriter rules (chỉ áp dụng cho Bobcat).
DEFAULT_REWRITE_RULES: tuple[str, ...] = (
    "prepositional_phrase",
    "determiner",
    "auxiliary_verb",
    "connector",
    "coordination",
)


@dataclass
class ParseResult:
    """Kết quả parse 1 batch phrase."""

    diagrams: list[Any]
    texts: list[str]
    labels: list[int]
    source_ids: list[int]
    failures: list[dict[str, Any]] = field(default_factory=list)
    reader_type: str = "unknown"

    @property
    def n_total(self) -> int:
        return len(self.texts)

    @property
    def n_success(self) -> int:
        return sum(1 for d in self.diagrams if d is not None)

    @property
    def n_failed(self) -> int:
        return self.n_total - self.n_success


def _resolve_reader(class_name: str, instance_name: str):
    """Tìm reader trong lambeq với cả 2 API pattern (class hoặc instance).

    lambeq 0.4 dùng class CamelCase: ``SpidersReader()``.
    lambeq 0.5+ dùng instance snake_case: ``spiders_reader`` (đã instantiate sẵn).
    Class top-level trong 0.5 thường là abstract → KHÔNG instantiate trực tiếp.

    Trả về một object có method ``sentence2diagram``.
    """
    import lambeq

    # 1. Ưu tiên instance snake_case top-level (lambeq 0.5 canonical):
    #    lambeq.spiders_reader → đã là instance, dùng luôn
    attr = getattr(lambeq, instance_name, None)
    if attr is not None and hasattr(attr, "sentence2diagram"):
        return attr

    # 2. Thử class CamelCase top-level (lambeq 0.4):
    #    lambeq.SpidersReader() — phòng khi class concrete
    cls = getattr(lambeq, class_name, None)
    if cls is not None and isinstance(cls, type):
        try:
            return cls()
        except TypeError:
            # ABCMeta abstract — không instantiate được
            pass

    # 3. Submodule lambeq.text2diagram (cả 2 dạng):
    try:
        t2d = __import__("lambeq.text2diagram", fromlist=[class_name, instance_name])
        attr = getattr(t2d, instance_name, None)
        if attr is not None and hasattr(attr, "sentence2diagram"):
            return attr
        cls = getattr(t2d, class_name, None)
        if cls is not None and isinstance(cls, type):
            try:
                return cls()
            except TypeError:
                pass
    except ImportError:
        pass

    # 4. Submodule riêng: lambeq.text2diagram.spiders_reader
    try:
        sub = __import__(f"lambeq.text2diagram.{instance_name}", fromlist=[class_name])
        cls = getattr(sub, class_name, None)
        if cls is not None and isinstance(cls, type):
            try:
                return cls()
            except TypeError:
                pass
    except ImportError:
        pass

    raise ImportError(
        f"Không tìm được '{instance_name}' hoặc '{class_name}' trong lambeq. "
        f"Chạy: python -c 'import lambeq; print(sorted(dir(lambeq)))'"
    )


def make_parser(
    reader_type: ReaderType = "bobcat",
    model_path: str | None = None,
    verbose: str = "suppress",
):
    """Tạo reader/parser theo type.

    Parameters
    ----------
    reader_type : str
        "bobcat", "spiders", "cups", hoặc "linear".
    model_path : str, optional
        Chỉ dùng cho bobcat — đường dẫn local model nếu không tải được online.
    verbose : str
        Chỉ áp dụng cho bobcat ("suppress", "text", "progress").
    """
    if reader_type == "bobcat":
        from lambeq import BobcatParser

        kwargs: dict[str, Any] = {"verbose": verbose}
        if model_path is not None:
            kwargs["model_name_or_path"] = model_path
        return BobcatParser(**kwargs)

    elif reader_type == "spiders":
        return _resolve_reader("SpidersReader", "spiders_reader")

    elif reader_type == "cups":
        return _resolve_reader("CupsReader", "cups_reader")

    elif reader_type == "linear":
        return _resolve_reader("LinearReader", "linear_reader")

    else:
        raise ValueError(
            f"reader_type không hỗ trợ: {reader_type!r}. "
            f"Chọn 1 trong: bobcat, spiders, cups, linear"
        )


def make_rewriter(rules: tuple[str, ...] = DEFAULT_REWRITE_RULES):
    """Tạo Rewriter chain. Chỉ hữu ích cho diagram của Bobcat."""
    from lambeq import Rewriter

    return Rewriter(list(rules))


def parse_one(parser, rewriter, text: str):
    """Parse 1 phrase. Trả về diagram hoặc raise."""
    diagram = parser.sentence2diagram(text)
    if rewriter is not None:
        diagram = rewriter(diagram)
    return diagram


def parse_batch(
    texts: list[str],
    labels: list[int],
    source_ids: list[int],
    parser=None,
    rewriter=None,
    rules: tuple[str, ...] = DEFAULT_REWRITE_RULES,
    apply_rewriter: bool = True,
    reader_type: ReaderType = "bobcat",
    model_path: str | None = None,
    verbose: bool = True,
    desc: str = "parsing",
) -> ParseResult:
    """Parse list phrase. Skip failure, ghi nhận vào failures.

    Parameters
    ----------
    reader_type
        Loại reader cần dùng nếu parser chưa được truyền sẵn.
    model_path
        Path local cho bobcat (nếu URL online dead).
    """
    if parser is None:
        parser = make_parser(reader_type=reader_type, model_path=model_path)

    # Rewriter chỉ ý nghĩa với bobcat (CCG diagram).
    effective_apply = apply_rewriter and reader_type == "bobcat"
    if effective_apply and rewriter is None:
        rewriter = make_rewriter(rules)
    if not effective_apply:
        rewriter = None

    diagrams: list[Any] = []
    failures: list[dict[str, Any]] = []

    iterator = enumerate(texts)
    if verbose:
        iterator = tqdm(iterator, total=len(texts), desc=desc, unit="phrase")

    for idx, text in iterator:
        try:
            diagram = parse_one(parser, rewriter, text)
            diagrams.append(diagram)
        except Exception as e:
            diagrams.append(None)
            failures.append(
                {
                    "idx": idx,
                    "source_id": source_ids[idx],
                    "text": text,
                    "label": labels[idx],
                    "error_type": type(e).__name__,
                    "error_msg": str(e)[:200],
                }
            )

    return ParseResult(
        diagrams=diagrams,
        texts=list(texts),
        labels=list(labels),
        source_ids=list(source_ids),
        failures=failures,
        reader_type=reader_type,
    )


def filter_successful(result: ParseResult) -> ParseResult:
    """Trả về ParseResult chỉ chứa các phrase parse thành công."""
    keep_idx = [i for i, d in enumerate(result.diagrams) if d is not None]
    return ParseResult(
        diagrams=[result.diagrams[i] for i in keep_idx],
        texts=[result.texts[i] for i in keep_idx],
        labels=[result.labels[i] for i in keep_idx],
        source_ids=[result.source_ids[i] for i in keep_idx],
        failures=result.failures,
        reader_type=result.reader_type,
    )

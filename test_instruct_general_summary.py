from data_pipeline.instruct_formatter import InstructFormatter


def test_general_summary_removes_pdf_noise_and_avoids_raw_echo():
    formatter = InstructFormatter(template="alpaca")
    raw_text = (
        "06/03/2026, 20:52 DataPipe (Project: DataClean1)\n"
        "DataPipe (Project: DataClean1)\n"
        "High-Dimensional AutoML, Generative AI Pipelines, and Distributed Ecosystem Architecture\n"
        "Abstract\n"
        "The DataClean1 project represents a fundamental intersection between deterministic data sanitization, "
        "non-parametric ML, and large-scale LLM preprocessing.\n"
        "https://md2pdf.netlify.app 1/13\n"
    )

    pairs = formatter.format_chunks(
        [{"text": raw_text, "source_type": "text"}],
        domain="general",
        generate_qa=False,
        pairs_per_chunk=1,
    )

    assert pairs
    output = pairs[0]["output"]

    assert output != raw_text
    assert output.lower().startswith("key concepts include:")
    assert "https://md2pdf.netlify.app" not in output.lower()
    assert "06/03/2026" not in output


def test_general_qa_does_not_use_product_data_phrase_or_full_chunk_echo():
    formatter = InstructFormatter(template="alpaca")
    raw_text = (
        "Distributed Ecosystem Architecture enables scaling across services.\n"
        "The DataClean1 architecture uses layered design with web, ML, and LLM pipelines.\n"
        "It supports asynchronous workflows and cloud-native deployment patterns.\n"
    )

    pairs = formatter.format_chunks(
        [{"text": raw_text, "source_type": "text"}],
        domain="general",
        generate_qa=True,
        pairs_per_chunk=1,
    )

    qa_pairs = [p for p in pairs if "?" in p.get("instruction", "")]
    assert qa_pairs
    for p in qa_pairs:
        assert "product data" not in p["instruction"].lower()
        assert p["output"].strip() != raw_text.strip()


def test_summary_point_cleanup_avoids_double_period_and_trailing_colon():
    formatter = InstructFormatter(template="alpaca")
    raw_text = (
        "Instead, DataClean1 uses Lazy Loading.\n"
        "@app.route('/process', methods=['POST'])\n"
        "Header.Payload.Signature\n"
        "Signature generated using:\n"
        "HMAC-SHA256(secret_key, payload)\n"
    )
    pairs = formatter.format_chunks(
        [{"text": raw_text, "source_type": "text"}],
        domain="general",
        generate_qa=False,
        pairs_per_chunk=1,
    )
    assert pairs
    out = pairs[0]["output"]
    assert ".." not in out
    assert not out.rstrip().endswith(":")

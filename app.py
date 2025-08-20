import json
import streamlit as st
from utils import generate_from_text

st.set_page_config(page_title="Intelligent Question Generator", page_icon="📝")

with st.sidebar:
    st.header("ℹ️ About App")
    st.write(
        """
        This app automatically generates different types of questions 
        (MCQs, Fill-in-the-Blanks, True/False, Short Answer) 
        from any given text using NLP techniques.  

        👉 You can paste text or upload a `.txt` file.  
        👉 Export results as **JSON, CSV, or PDF**.  
        👉 Useful for **teachers, trainers, and students**.  
        """
    )

    st.header("👨‍💻 About Me")
    st.write(
        """
        **Developer:** Nitish Kumar Gupta  
        **Role:** Computer Vision Engineer & AI Enthusiast  
        Passionate about AI, NLP, and building intelligent applications.  

        📧 Contact: kumarknitish@gmail.com  
        🌐 GitHub: [github.com/nitish](https://github.com/nitishkrgupta) 
        🌐 Linkedin: (https://www.linkedin.com/in/nitish-kumar-gupta-1b4274269/) 
        """
    )

st.title("📝 Intelligent Question Generator")
st.caption("Generate MCQs, Fill-in-the-Blanks,and True/False questions from any text.")

# ------------------ Settings ------------------
with st.expander("⚙️ Settings", expanded=True):
    col1, col2, col3 = st.columns(3)
    max_sents = col1.number_input("Max sentences", min_value=1, max_value=50, value=8, step=1)
    num_qs = col2.number_input("Q candidates per sentence", min_value=1, max_value=10, value=3, step=1)
    types = col3.multiselect(
        "Question Types",
        ["mcq", "fill_blank", "true_false", "short_answer"],
        default=["mcq", "fill_blank", "true_false", "short_answer"],
    )

# ------------------ Input ------------------
tab1, tab2 = st.tabs(["✍️ Paste Text", "📄 Upload .txt"])
with tab1:
    input_text = st.text_area("Paste source content here:", height=220)
with tab2:
    uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded is not None:
        input_text = uploaded.read().decode("utf-8")

# ------------------ Generate ------------------
if st.button("Generate"):
    if not input_text or not input_text.strip():
        st.warning("Please provide some text.")
        st.stop()

    with st.spinner("Generating questions..."):
        questions = generate_from_text(
            input_text,
            max_sents=int(max_sents),
            num_q_per_sent=int(num_qs),
            include_types=tuple(types),
        )

    st.success(f"Generated {len(questions)} questions 🎉")

    # ------------------ Render results ------------------
    for i, q in enumerate(questions, 1):
        st.markdown("---")
        st.markdown(
            f"**Q{i}. Type:** `{q.get('type')}`  |  "
            f"**Difficulty:** `{q.get('difficulty')}`  |  "
            f"**Bloom:** `{q.get('bloom')}`"
        )

        if q["type"] == "mcq":
            st.write(q["question"])
            for idx, opt in enumerate(q["options"]):
                st.write(f"- ({chr(65+idx)}) {opt}")
            with st.expander("Answer"):
                st.write(f"✅ {q.get('answer')} (Option {chr(65+q['answer_index'])})")
            with st.expander("Source sentence"):
                st.write(q.get("source"))

        elif q["type"] == "fill_blank":
            st.write(q["question"])
            with st.expander("Answer"):
                st.write(f"✅ {q.get('answer')}")
            with st.expander("Source sentence"):
                st.write(q.get("source"))

        elif q["type"] == "true_false":
            st.write(q["statement"])
            with st.expander("Answer"):
                st.write(f"✅ {q.get('answer')}")
            with st.expander("Source sentence"):
                st.write(q.get("source"))

        elif q["type"] == "short_answer":
            st.write(q["question"])
            with st.expander("Sample Answer (extracted key)"):
                st.write(f"✅ {q.get('answer')}")
            with st.expander("Source sentence"):
                st.write(q.get("source"))

    # ------------------ Export ------------------
    with st.expander("📥 Export Results"):
        # JSON export
        as_json = json.dumps(questions, indent=2, ensure_ascii=False)
        st.download_button(
            "Download JSON",
            data=as_json,
            file_name="questions.json",
            mime="application/json",
        )

        # CSV export
        import csv, io
        keys = sorted(set().union(*[q.keys() for q in questions])) if questions else []
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=keys)
        writer.writeheader()
        for q in questions:
            row = {
                k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                for k, v in q.items()
            }
            writer.writerow(row)

        st.download_button(
            "Download CSV",
            data=csv_buf.getvalue(),
            file_name="questions.csv",
            mime="text/csv",
        )

        # PDF export (Only Questions + Options)
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter
        y = height - 50

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Generated Questions")
        y -= 30
        c.setFont("Helvetica", 11)

        for i, q in enumerate(questions, 1):
            if y < 80:  # new page if space is low
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 11)

            if q["type"] == "mcq":
                c.drawString(50, y, f"Q{i}. {q['question']}")
                y -= 20
                for idx, opt in enumerate(q["options"]):
                    c.drawString(70, y, f"({chr(65+idx)}) {opt}")
                    y -= 20
            else:
                # For non-MCQ questions
                text = q.get("question") or q.get("statement", "")
                c.drawString(50, y, f"Q{i}. {text}")
                y -= 30

        c.save()
        pdf_buffer.seek(0)

        st.download_button(
            "Download PDF",
            data=pdf_buffer,
            file_name="questions.pdf",
            mime="application/pdf",
        )
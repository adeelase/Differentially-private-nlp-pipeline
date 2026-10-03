import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and render total page numbers
    along with running headers and footers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Suppress running header on Page 1 (Cover / Header area)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "Task 1: Differentially Private NLP Pipeline — Technical Interview Guide")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)
        
        # Running Footer on all pages
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 36, page_str)
        self.drawString(54, 36, "CONFIDENTIAL & PROPRIETARY — PREPARED FOR TECHNICAL INTERVIEW")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 8.5 * inch - 54, 48)
        
        self.restoreState()

def build_pdf(filename="Task1_Interview_Guide.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#1A365D")   # Deep Navy
    SECONDARY = colors.HexColor("#2B6CB0") # Slate Blue
    NEUTRAL_DARK = colors.HexColor("#2D3748") # Dark Charcoal
    BG_LIGHT = colors.HexColor("#F7FAFC")  # Soft Grey Background
    BORDER_COLOR = colors.HexColor("#E2E8F0")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=PRIMARY,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=SECONDARY,
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=SECONDARY,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=NEUTRAL_DARK,
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        bulletIndent=5,
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#2C5282")
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=body_style,
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=body_style,
        fontSize=8,
        leading=11,
        spaceAfter=0
    )

    story = []

    # Title Banner Block
    story.append(Paragraph("Technical Deep-Dive & Interview Guide", title_style))
    story.append(Paragraph("Task 1: Differentially Private NLP Text Obfuscation Pipeline", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=12))

    # Section 1
    story.append(Paragraph("1. Executive Summary & Core Premise", h1_style))
    story.append(Paragraph(
        "Unstructured text datasets (such as IMDB movie reviews, clinical records, and customer communications) "
        "contain rich semantic information vital for machine learning models. However, raw text frequently exposes "
        "Personally Identifiable Information (PII) and sensitive contextual markers vulnerable to adversarial re-identification.",
        body_style
    ))
    story.append(Paragraph(
        "Traditional sanitization methods—such as Named Entity Recognition (NER) masking or regex redaction—fail to prevent "
        "linkage and membership inference attacks. This project implements a <b>Local Differential Privacy (LDP)</b> framework "
        "operating directly in dense vector space (R<sup>d</sup>). Continuous 50-dimensional GloVe word embeddings are perturbed using "
        "calibrated <b>Multivariate Laplacian Noise</b> derived from differential privacy metrics, guaranteeing provable mathematical "
        "privacy bounds while maintaining high downstream semantic utility.",
        body_style
    ))

    # Section 2
    story.append(Paragraph("2. Literature Review & Theoretical Foundations", h1_style))
    story.append(Paragraph("2.1 Comparative Analysis of Text Privacy Paradigms", h2_style))

    # Comparison Table
    table_data = [
        [
            Paragraph("Privacy Model", table_header_style),
            Paragraph("Mechanism", table_header_style),
            Paragraph("Advantages", table_header_style),
            Paragraph("Critical Vulnerabilities", table_header_style)
        ],
        [
            Paragraph("K-Anonymity & Masking", table_cell_style),
            Paragraph("Rule-based entity removal / Regex substitution", table_cell_style),
            Paragraph("Simple to implement, human-readable text", table_cell_style),
            Paragraph("Vulnerable to background knowledge and high-dimensional linkage attacks.", table_cell_style)
        ],
        [
            Paragraph("Global DP (Centralized)", table_cell_style),
            Paragraph("Noise added to query results or gradients (DP-SGD)", table_cell_style),
            Paragraph("Strong mathematical guarantees", table_cell_style),
            Paragraph("Requires an absolute trusted central server; raw data exposed at collection.", table_cell_style)
        ],
        [
            Paragraph("Local DP (Discrete)", table_cell_style),
            Paragraph("Client-side discrete randomized response on tokens", table_cell_style),
            Paragraph("Zero-trust required in central server", table_cell_style),
            Paragraph("Catastrophic utility loss over large vocabularies (random output tokens).", table_cell_style)
        ],
        [
            Paragraph("Metric LDP (This Project)", table_cell_style),
            Paragraph("Multivariate Laplacian noise injected into vector space R<sup>d</sup>", table_cell_style),
            Paragraph("Preserves local semantic neighborhood and utility", table_cell_style),
            Paragraph("Requires careful parameter calibration (epsilon vs. distance metric).", table_cell_style)
        ]
    ]

    col_widths = [105, 120, 125, 154]
    comp_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("2.2 Theoretical Foundations of Local Differential Privacy", h2_style))
    story.append(Paragraph(
        "A randomized algorithm <i>M</i> provides <b>&epsilon;-Differential Privacy</b> if, for all neighboring inputs <i>d, d'</i> "
        "and all output subsets <i>S</i>:",
        body_style
    ))
    story.append(Paragraph("<b>Pr[M(d) &isin; S] &le; exp(&epsilon;) &times; Pr[M(d') &isin; S]</b>", bullet_style))
    story.append(Paragraph(
        "Where <b>&epsilon; (Privacy Budget)</b> controls the privacy-utility trade-off. Lower &epsilon; values produce stronger privacy guarantees "
        "by increasing the variance of the injected noise. In continuous embedding space R<sup>d</sup>, noise vectors are drawn from a spherical "
        "distribution where direction is uniformly distributed on the unit sphere and magnitude follows a continuous Gamma distribution.",
        body_style
    ))

    # Section 3
    story.append(Paragraph("3. Deep-Dive Implementation & Code Architecture", h1_style))
    
    code_blocks = [
        ("3.1 Text Preprocessing & Cleaning Pipeline",
"""import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text)).lower()
    tokens = word_tokenize(text)
    return [
        lemmatizer.lemmatize(w) for w in tokens 
        if w not in stop_words and len(w) > 1
    ]"""),

        ("3.2 Multivariate Laplacian Noise Generation in R^d Space",
"""import numpy as np

def generate_laplacian_noise_vector(dim=50, sensitivity=1.0, epsilon=25.0):
    # Step 1: Uniform random direction vector via standard Gaussian sampling
    gaussian_samples = np.random.normal(0, 1, dim)
    norm = np.linalg.norm(gaussian_samples)
    direction = gaussian_samples / norm if norm != 0 else gaussian_samples
    
    # Step 2: Radial magnitude from Gamma distribution
    scale = sensitivity / epsilon
    magnitude = np.random.gamma(shape=dim, scale=scale)
    
    return direction * magnitude"""),

        ("3.3 Obfuscation & Stochastic Top-k Semantic Replacement",
"""from gensim.models import KeyedVectors

def replace_word(word, model, epsilon=25.0, sensitivity=1.0, top_k=5):
    if word not in model:
        return word  # OOV Fallback
    
    original_vec = model[word]
    noise = generate_laplacian_noise_vector(dim=model.vector_size, sensitivity=sensitivity, epsilon=epsilon)
    noisy_vec = original_vec + noise
    
    try:
        similar_words = model.most_similar(positive=[noisy_vec], topn=top_k)
        candidates = [w for w, sim in similar_words]
        return np.random.choice(candidates) # Stochastic selection
    except Exception:
        return word""")
    ]

    for title, code in code_blocks:
        block = []
        block.append(Paragraph(title, h2_style))
        
        # Render code in boxed container
        code_p = Paragraph(code.replace("\n", "<br/>").replace(" ", "&nbsp;"), code_style)
        code_table = Table([[code_p]], colWidths=[504])
        code_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
            ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        block.append(code_table)
        block.append(Spacer(1, 8))
        story.append(KeepTogether(block))

    # Section 4
    story.append(Paragraph("4. Key Technical Interview Q&A", h1_style))

    qa_list = [
        ("Q1: Why apply Local Differential Privacy in vector embedding space rather than on discrete tokens?",
         "Discrete token perturbation on large vocabularies yields essentially random output tokens, causing catastrophic utility loss. "
         "By projecting text into continuous metric space (R<sup>d</sup>), distance corresponds to semantic similarity. "
         "Injecting noise into embeddings allows us to sample replacement tokens from the local semantic neighborhood, preserving grammar and utility."),
        
        ("Q2: How does the pipeline handle Out-Of-Vocabulary (OOV) tokens during processing?",
         "OOV tokens are intercepted gracefully via lookup checks and try-except error handling. "
         "Unseen words are preserved without modification, preventing runtime exceptions and ensuring uninterrupted batch streaming on large datasets."),

        ("Q3: Why use stochastic selection among top-k nearest neighbors instead of always choosing k=1?",
         "Deterministic nearest-neighbor replacement (k=1) creates a fixed mapping function. An adversary with access to the vocabulary "
         "embedding matrix could perform deterministic spatial inversion attacks. Random sampling among top-k candidates introduces non-deterministic entropy, "
         "breaking deterministic mapping attacks."),

        ("Q4: How would you optimize this architecture for high-throughput production workloads?",
         "We can replace linear vector scans (O(V &middot; d)) with GPU-accelerated Approximate Nearest Neighbor (ANN) indexes like FAISS or Annoy (O(log V)). "
         "Additionally, static GloVe embeddings can be upgraded to contextual Transformer representations (e.g., RoBERTa/BERT embeddings) processed in batched GPU memory chunks.")
    ]

    for q, a in qa_list:
        qa_block = []
        qa_block.append(Paragraph(f"<b>{q}</b>", ParagraphStyle('QStyle', parent=body_style, fontName='Helvetica-Bold', textColor=PRIMARY)))
        qa_block.append(Paragraph(f"<i>Answer:</i> {a}", ParagraphStyle('AStyle', parent=body_style, leftIndent=8)))
        qa_block.append(Spacer(1, 4))
        story.append(KeepTogether(qa_block))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] PDF successfully generated: {os.path.abspath(filename)}")

if __name__ == "__main__":
    build_pdf()
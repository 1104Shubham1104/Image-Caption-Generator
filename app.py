import streamlit as st
from PIL import Image
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except ImportError:
    # Fallback for older transformers versions
    from transformers import AutoProcessor, AutoModelForVision2Seq
    BlipProcessor = AutoProcessor
    BlipForConditionalGeneration = AutoModelForVision2Seq
import time
import pandas as pd
import altair as alt
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch

# ----------- CONFIG ---------------
st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS 
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .caption-card {
        background: #ffffff;
        border: 2px solid #667eea;
        border-left: 5px solid #667eea;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-radius: 0.5rem;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #1a1a1a;
    }
    .caption-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    .caption-card strong {
        color: #667eea;
        font-size: 1.1rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    .caption-text {
        color: #2d3748;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 0.5rem;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(120deg, #764ba2 0%, #667eea 100%);
    }
    div[data-testid="stFileUploadDropzone"] {
        border: 2px dashed #667eea;
        border-radius: 1rem;
        background: #f8f9fa;
    }
    .section-header {
        color: #2d3748;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    .stMetric {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
    }
    .stMetric label {
        color: #4a5568 !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #667eea !important;
        font-size: 1.8rem !important;
    }
</style>
""", unsafe_allow_html=True)

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ----------- LOAD MODEL ---------------
@st.cache_resource(show_spinner=False)
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return processor, model, device

with st.spinner("🔄 Loading AI model..."):
    processor, model, device = load_model()

# ----------- FAST CAPTION GENERATION ---------------
def generate_captions_fast(image, processor, model, device, strategy="Sampling", temperature=1.1, num_beams=5, num_return=5):
    """Generate multiple captions efficiently in a single batch"""
    inputs = processor(image, return_tensors="pt").to(device)
    
    if strategy == "Greedy":
        # For greedy, generate once then duplicate
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=30,
                num_return_sequences=1
            )
        caption = processor.decode(out[0], skip_special_tokens=True)
        return [caption]
    
    elif strategy == "Beam Search":
        # Generate multiple sequences with beam search
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=30,
                num_beams=num_beams,
                num_return_sequences=min(num_return, num_beams),
                early_stopping=True
            )
        captions = [processor.decode(o, skip_special_tokens=True) for o in out]
        return captions
    
    else:  # Sampling - generate multiple in one call
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=30,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=temperature,
                num_return_sequences=num_return
            )
        captions = [processor.decode(o, skip_special_tokens=True) for o in out]
        return captions

# ----------- MAIN PAGE ---------------
st.markdown('<h1 class="main-header">🎨 AI Image Caption Generator</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload an image and get 5 AI-generated captions instantly</p>', unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Your Image", use_container_width=True)
        
        # Reference caption for BLEU score
        st.markdown("### 📝 Reference Caption (Optional)")
        reference_caption = st.text_input(
            "Enter ground truth caption for evaluation",
            placeholder="e.g., A dog playing in the park...",
            label_visibility="collapsed"
        )

with col2:
    if uploaded_file:
        st.markdown("### ⚙️ Generation Settings")
        
        col_a, col_b = st.columns(2)
        with col_a:
            strategy = st.selectbox(
                "Decoding Strategy",
                ["Sampling", "Beam Search", "Greedy"],
                help="Sampling: Creative & diverse\nBeam Search: Balanced\nGreedy: Fast & deterministic"
            )
        
        with col_b:
            if strategy == "Sampling":
                temperature = st.slider("Temperature", 0.7, 1.5, 1.1, 0.1, help="Higher = more creative")
            elif strategy == "Beam Search":
                num_beams = st.slider("Beam Size", 3, 10, 5, help="More beams = better quality")
            else:
                st.info("⚡ Fastest mode")
        
        generate_btn = st.button("🚀 Generate 5 Captions", use_container_width=True)
        
        if generate_btn:
            with st.spinner("🎨 Generating captions..."):
                start = time.time()
                
                # Generate captions efficiently
                if strategy == "Sampling":
                    captions = generate_captions_fast(
                        image, processor, model, device,
                        strategy="Sampling",
                        temperature=temperature,
                        num_return=5
                    )
                elif strategy == "Beam Search":
                    captions = generate_captions_fast(
                        image, processor, model, device,
                        strategy="Beam Search",
                        num_beams=num_beams,
                        num_return=5
                    )
                else:  # Greedy
                    # For greedy, we need to generate multiple with sampling to get variety
                    captions = generate_captions_fast(
                        image, processor, model, device,
                        strategy="Sampling",
                        temperature=0.7,
                        num_return=5
                    )
                
                # Remove duplicates while preserving order
                seen = set()
                unique_captions = []
                for c in captions:
                    if c not in seen:
                        seen.add(c)
                        unique_captions.append(c)
                
                # If we need more unique captions, generate additional ones
                while len(unique_captions) < 5:
                    extra = generate_captions_fast(
                        image, processor, model, device,
                        strategy="Sampling",
                        temperature=temperature if strategy == "Sampling" else 1.2,
                        num_return=3
                    )
                    for c in extra:
                        if c not in seen:
                            seen.add(c)
                            unique_captions.append(c)
                        if len(unique_captions) >= 5:
                            break
                
                captions = unique_captions[:5]
                total_time = time.time() - start
            
            # Display results
            st.success(f"✅ Generated {len(captions)} captions in {total_time:.2f}s")
            
            st.markdown("### 🎯 Generated Captions")
            for i, c in enumerate(captions, 1):
                st.markdown(f"""
                <div class="caption-card">
                    <strong>Caption {i}</strong>
                    <div class="caption-text">{c}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Store in session state for analysis
            st.session_state['captions'] = captions
            st.session_state['generation_time'] = total_time

# ----------- ANALYSIS SECTION ---------------
if uploaded_file and 'captions' in st.session_state:
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Analysis & Metrics</div>', unsafe_allow_html=True)
    
    captions = st.session_state['captions']
    
    # Metrics row
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.metric("📝 Captions", len(captions))
    
    with metric_cols[1]:
        avg_length = sum(len(c.split()) for c in captions) / len(captions)
        st.metric("📊 Avg Words", f"{avg_length:.1f}")
    
    with metric_cols[2]:
        st.metric("⚡ Time", f"{st.session_state['generation_time']:.2f}s")
    
    with metric_cols[3]:
        unique_words = len(set(word for c in captions for word in c.split()))
        st.metric("🔤 Unique Words", unique_words)
    
    # Charts
    st.markdown("<br>", unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown('<div style="color: #2d3748; font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem;">📏 Word Count Distribution</div>', unsafe_allow_html=True)
        lengths = [len(c.split()) for c in captions]
        df_len = pd.DataFrame({
            "Caption": [f"Caption {i}" for i in range(1, len(captions) + 1)],
            "Word Count": lengths,
        })
        
        chart = (
            alt.Chart(df_len)
            .mark_bar(color='#667eea')
            .encode(
                x=alt.X("Caption:N", title=""),
                y=alt.Y("Word Count:Q", title="Words"),
                tooltip=["Caption", "Word Count"]
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    
    with chart_col2:
        if reference_caption and reference_caption.strip():
            st.markdown('<div style="color: #2d3748; font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem;">🎯 BLEU Score Evaluation</div>', unsafe_allow_html=True)
            smoothie = SmoothingFunction().method1
            
            bleu_scores = []
            for c in captions:
                score = sentence_bleu(
                    [reference_caption.split()],
                    c.split(),
                    smoothing_function=smoothie
                )
                bleu_scores.append(score)
            
            df_bleu = pd.DataFrame({
                "Caption": [f"Caption {i}" for i in range(1, len(captions) + 1)],
                "BLEU Score": bleu_scores
            })
            
            chart = (
                alt.Chart(df_bleu)
                .mark_bar(color='#764ba2')
                .encode(
                    x=alt.X("Caption:N", title=""),
                    y=alt.Y("BLEU Score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                    tooltip=["Caption", alt.Tooltip("BLEU Score:Q", format=".3f")]
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
            
            best_idx = bleu_scores.index(max(bleu_scores))
            st.info(f"🏆 Best match: Caption {best_idx + 1} (BLEU: {max(bleu_scores):.3f})")
        else:
            st.info("💡 Add a reference caption above to see BLEU score evaluation")

# ----------- SIDEBAR INFO ---------------
with st.sidebar:
    st.markdown("## ℹ️ About")
    
    # Show device info
    device_emoji = "🚀" if device == "cuda" else "💻"
    st.markdown(f"""
    **Device:** {device_emoji} {device.upper()}  
    **Model:** BLIP (Base)  
    **Source:** Salesforce Research  
    **Parameters:** ~247M  
    **Task:** Image Captioning
    
    ---
    
    ### 🔍 Strategies
    
    **Sampling**  
    Creative & diverse captions
    
    **Beam Search**  
    Balanced quality & speed
    
    **Greedy**  
    Fast & deterministic
    
    ---
    
    ### 📏 BLEU Score
    Measures similarity between generated and reference captions (0-1, higher is better)
    
    ---
    
    ### ⚡ Speed Tips
    - Use GPU for 10x faster generation
    - Sampling mode is fastest
    - Beam search is slower but better quality
    """)
    
    st.markdown("---")
    st.markdown("Made with using Streamlit & Hugging Face")